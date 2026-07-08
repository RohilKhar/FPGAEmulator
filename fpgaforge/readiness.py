"""First-pass FPGA readiness gate.

Fuses every signal the platform produces -- synthesis, place & route, timing
margin, resource headroom, I/O fit, design-rule risks, and virtual bring-up --
into a single verdict for any input design: can it get to the FPGA on the first
shot, and if not, what is blocking it and how to fix it.

The check-evaluation logic is pure (data in, checks out) so it is fully
unit-testable without any tools; `assess()` orchestrates the real runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# Verdicts, in increasing order of concern.
READY = "READY"
AT_RISK = "AT_RISK"
BLOCKED = "BLOCKED"

_STATUS_RANK = {"pass": 0, "warn": 1, "fail": 2}

# Approximate usable logic-cell and bonded-I/O counts per default package.
from .devices import all_devices as _all_devices

# Capacity tables derived from the central device registry (fpgaforge/devices.py).
DEVICE_LUT_CAPACITY = {d.target: d.luts for d in _all_devices()}
DEVICE_IO_CAPACITY = {d.target: d.io for d in _all_devices()}

# Reset-like names excluded from clock-domain counting.
_RESET_TOKENS = {"rst", "reset", "rst_n", "resetn", "nrst", "rstn", "reset_n", "arst", "arst_n"}

# Timing headroom below which "met" timing is still considered risky.
_TIGHT_TIMING_RATIO = 1.15
_HIGH_UTIL_RATIO = 0.80


@dataclass
class Check:
    name: str
    status: str          # "pass" | "warn" | "fail"
    message: str
    recommendation: str | None = None

    @property
    def icon(self) -> str:
        return {"pass": "[ok]", "warn": "[warn]", "fail": "[FAIL]"}[self.status]


@dataclass
class ReadinessReport:
    design_id: str
    verdict: str
    score: int
    checks: list[Check] = field(default_factory=list)
    simulated: bool = False   # True if run against the mock backend (no real tools)
    fmax_mhz: float = 0.0
    target_mhz: float = 0.0
    target_device: str = ""
    vendor: str = ""
    equivalence_tier: str = ""   # "bitstream" | "netlist" | "none" (see devices.py)
    bitstream_path: str | None = None
    diagnostics: list[str] = field(default_factory=list)   # real tool errors
    tool_warnings: list[str] = field(default_factory=list)  # real tool warnings
    # Raw facts for downstream consumers (e.g. an RL reward function): continuous
    # metrics that the pass/warn/fail checks were derived from.
    metrics: dict = field(default_factory=dict)

    @property
    def recommendations(self) -> list[str]:
        return [c.recommendation for c in self.checks if c.recommendation]

    def summary(self) -> str:
        headline = {
            READY: "READY - expected to reach the FPGA first shot",
            AT_RISK: "AT RISK - can build, but review the warnings below",
            BLOCKED: "BLOCKED - will not reach the FPGA as-is",
        }[self.verdict]
        lines = [
            f"first-pass readiness: {self.verdict}  (confidence {self.score}/100)",
            headline,
            f"design : {self.design_id}",
        ]
        if self.target_device:
            tier = {
                "bitstream": "bit-level (proves the flashed bitstream)",
                "netlist": "netlist-level (proves the post-impl netlist)",
                "none": "none (no formal equivalence path on this silicon)",
            }.get(self.equivalence_tier, self.equivalence_tier)
            vend = f" [{self.vendor}]" if self.vendor else ""
            lines.append(f"device : {self.target_device}{vend} - equivalence tier: {tier}")
        if self.fmax_mhz or self.target_mhz:
            lines.append(
                f"timing : {self.fmax_mhz:.1f} MHz achieved vs {self.target_mhz:.1f} MHz target"
            )
        if self.bitstream_path:
            lines.append(f"bitstream: {self.bitstream_path}")
        if self.simulated:
            lines.append("note   : simulated backend (install FPGA tools for a real verdict)")
        lines.append("checks :")
        for c in self.checks:
            lines.append(f"  {c.icon:7} {c.name}: {c.message}")
        if self.diagnostics:
            lines.append("tool errors:")
            for d in self.diagnostics:
                lines.append(f"  {d}")
        if self.tool_warnings:
            lines.append("tool warnings:")
            for w in self.tool_warnings:
                lines.append(f"  {w}")
        if self.recommendations:
            lines.append("recommended fixes:")
            for r in self.recommendations:
                lines.append(f"  - {r}")
        return "\n".join(lines)


def count_clock_domains(rtl_text: str) -> int:
    """Heuristic count of distinct clock signals (edge-sensitive, non-reset)."""
    text = re.sub(r"//.*", "", rtl_text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    signals = set(re.findall(r"(?:pos|neg)edge\s+(\w+)", text))
    clocks = {s for s in signals if s.lower() not in _RESET_TOKENS}
    return len(clocks)


def _has_latch(synth_log: str) -> bool:
    low = synth_log.lower()
    return "dlatch" in low or "inferring latch" in low or "proc_dlatch" in low


def _has_comb_loop(synth_log: str) -> bool:
    low = synth_log.lower()
    return "logic loop" in low or "combinational loop" in low


def evaluate(
    *,
    synthesized: bool,
    routed_ok: bool,
    fmax_mhz: float,
    target_mhz: float,
    luts: int,
    lut_capacity: int,
    io_count: int,
    io_capacity: int,
    bringup_status: str,       # "up" | "down" | "skipped"
    synth_log: str = "",
    rtl_text: str = "",
    bitstream_equivalence: str = "skipped",  # see _equivalence_check
    equivalence_detail: str = "",
    equivalence_tier_used: str = "bitstream",  # "bitstream" | "netlist"
    pin_status: str = "skipped",   # "valid" | "invalid" | "missing" | "skipped"
    pin_detail: str = "",
    cdc_report=None,          # optional cdc.CDCReport for structural CDC analysis
) -> list[Check]:
    """Build the full list of readiness checks from run facts. Pure."""
    checks: list[Check] = []

    # 1. Synthesis
    if synthesized:
        checks.append(Check("synthesis", "pass", "RTL synthesized to FPGA primitives"))
    else:
        checks.append(
            Check(
                "synthesis", "fail", "synthesis failed",
                "Fix RTL errors reported by the synthesizer before proceeding.",
            )
        )
        return checks  # nothing downstream is meaningful

    # 2. Place & route / fit
    if routed_ok:
        checks.append(Check("place_and_route", "pass", "placed and routed successfully"))
    else:
        checks.append(
            Check(
                "place_and_route", "fail", "placement/routing failed",
                "Reduce resource/I/O usage or pick a larger device/package.",
            )
        )

    # 3. Timing
    if not routed_ok or fmax_mhz <= 0:
        checks.append(
            Check("timing", "fail", "no timing result (design did not route)",
                  "Achieve routing first, then re-check timing.")
        )
    elif fmax_mhz < target_mhz:
        checks.append(
            Check(
                "timing", "fail",
                f"misses timing: {fmax_mhz:.1f} MHz < {target_mhz:.1f} MHz target",
                "Enable retiming, add a pipeline stage on the critical path, or "
                "relax the target clock.",
            )
        )
    elif fmax_mhz < target_mhz * _TIGHT_TIMING_RATIO:
        margin = 100.0 * (fmax_mhz / target_mhz - 1.0)
        checks.append(
            Check(
                "timing", "warn",
                f"tight margin: {fmax_mhz:.1f} MHz is only +{margin:.0f}% over target",
                "Add timing headroom (pipelining/retiming); little margin for "
                "board-level and PVT variation.",
            )
        )
    else:
        margin = 100.0 * (fmax_mhz / target_mhz - 1.0)
        checks.append(
            Check("timing", "pass", f"meets timing with +{margin:.0f}% margin")
        )

    # 4. Resource headroom
    if lut_capacity > 0 and luts > 0:
        util = luts / lut_capacity
        if util > 1.0:
            checks.append(
                Check("resource_headroom", "fail",
                      f"over capacity: {luts}/{lut_capacity} LUTs ({util*100:.0f}%)",
                      "Use a larger device or reduce logic.")
            )
        elif util > _HIGH_UTIL_RATIO:
            checks.append(
                Check("resource_headroom", "warn",
                      f"high utilization: {luts}/{lut_capacity} LUTs ({util*100:.0f}%)",
                      "High utilization raises routing congestion and timing risk; "
                      "consider a larger device.")
            )
        else:
            checks.append(
                Check("resource_headroom", "pass",
                      f"{luts}/{lut_capacity} LUTs ({util*100:.0f}%)")
            )

    # 5. I/O fit for the package
    if io_capacity > 0 and io_count > 0:
        if io_count > io_capacity:
            checks.append(
                Check("io_fit", "fail",
                      f"needs {io_count} I/O but package has ~{io_capacity}",
                      "Choose a larger package, or reduce/serialize top-level I/O.")
            )
        elif io_count > io_capacity * 0.9:
            checks.append(
                Check("io_fit", "warn",
                      f"I/O nearly full: {io_count}/~{io_capacity}",
                      "Little I/O headroom; confirm a pin-constraint file exists.")
            )
        else:
            checks.append(
                Check("io_fit", "pass", f"{io_count}/~{io_capacity} package I/O")
            )

    # 6. Design-rule risks
    if _has_comb_loop(synth_log):
        checks.append(
            Check("no_comb_loops", "fail", "combinational loop detected",
                  "Break the combinational feedback loop in the RTL.")
        )
    else:
        checks.append(Check("no_comb_loops", "pass", "no combinational loops"))

    if _has_latch(synth_log):
        checks.append(
            Check("latch_free", "warn", "inferred latch(es) detected",
                  "Add default assignments / complete if-else to avoid latches.")
        )
    else:
        checks.append(Check("latch_free", "pass", "no inferred latches"))

    # 7. Clock-domain crossing analysis (structural when a netlist was available,
    # else a name-count heuristic).
    if cdc_report is not None and cdc_report.n_domains >= 1:
        if cdc_report.unsynchronized:
            c = cdc_report.unsynchronized[0]
            checks.append(
                Check("clock_domain_crossing", "fail",
                      f"{len(cdc_report.unsynchronized)} unsynchronized crossing(s), "
                      f"e.g. {c.from_domain}->{c.to_domain}",
                      "Add a 2-flop synchronizer (or async FIFO for buses); do not "
                      "route a raw cross-domain signal through combinational logic.")
            )
        elif cdc_report.single_flop:
            checks.append(
                Check("clock_domain_crossing", "warn",
                      f"{len(cdc_report.single_flop)} single-flop crossing(s)",
                      "Single-flop capture leaves residual metastability risk; use "
                      "a two-flop synchronizer.")
            )
        elif cdc_report.crossings:
            checks.append(Check("clock_domain_crossing", "pass",
                                f"{len(cdc_report.crossings)} crossing(s), all synchronized"))
        else:
            checks.append(Check("clock_domain_crossing", "pass",
                                f"{cdc_report.n_domains} clock domain(s), no crossings"))
    else:
        domains = count_clock_domains(rtl_text) if rtl_text else 0
        if domains > 1:
            checks.append(
                Check("clock_domains", "warn",
                      f"{domains} clock domains detected (name heuristic)",
                      "Multiple clocks imply clock-domain crossings; ensure proper "
                      "synchronizers (netlist CDC analysis unavailable).")
            )
        elif domains == 1:
            checks.append(Check("clock_domains", "pass", "single clock domain"))

    # 8. Functional virtual bring-up
    if bringup_status == "up":
        checks.append(Check("functional_bringup", "pass", "design came up and behaved in the virtual fabric"))
    elif bringup_status == "down":
        checks.append(
            Check("functional_bringup", "fail", "virtual bring-up failed (bad behavior/hang)",
                  "Fix functional bugs found in the virtual FPGA before building.")
        )
    else:  # skipped
        checks.append(
            Check("functional_bringup", "warn", "virtual bring-up not run",
                  "Run bring-up (needs iverilog/vvp) to verify behavior pre-hardware.")
        )

    # 9. Bitstream equivalence (the strongest evidence: does the flashed image
    # actually implement the RTL?).
    eq = _equivalence_check(bitstream_equivalence, equivalence_detail,
                            tier=equivalence_tier_used)
    if eq is not None:
        checks.append(eq)

    # 10. Pin constraints (board-level reality: a wrong/missing pin map is the
    # classic first-shot killer no functional check can catch).
    if pin_status == "valid":
        checks.append(Check("pin_constraints", "pass",
                            f"pin map validated: {pin_detail}".rstrip(": ")))
    elif pin_status == "invalid":
        checks.append(Check(
            "pin_constraints", "fail",
            f"pin constraints are wrong: {pin_detail}",
            "Fix the pin map before flashing -- a wrong pin assignment is the "
            "#1 cause of first-shot bench failures."))
    elif pin_status == "missing":
        checks.append(Check(
            "pin_constraints", "fail",
            "no pin-constraint file provided; I/O will be auto-placed on pins "
            "your board is not wired to",
            "Write a .pcf/.xdc/.qsf/.lpf mapping every top-level port to the "
            "board's pins and pass it via --pins."))
    # "skipped" -> the user did not engage pin checking; emit nothing.

    return checks


def _equivalence_check(status: str, detail: str, tier: str = "bitstream") -> Check | None:
    """Turn an equivalence outcome into a readiness check.

    ``tier`` is ``"bitstream"`` (RTL == flashed bits, open silicon) or
    ``"netlist"`` (RTL == vendor post-impl netlist, vendor-locked silicon).
    """
    name = "netlist_equivalence" if tier == "netlist" else "bitstream_equivalence"
    subject = "post-implementation netlist" if tier == "netlist" else "flashed bitstream"
    if status == "proved_all":
        return Check(name, "pass",
                     f"{subject} formally equivalent to RTL (all inputs, all time)")
    if status == "proved_bounded":
        return Check(name, "pass",
                     f"{subject} formally equivalent to RTL {detail}".rstrip())
    if status == "verified":
        return Check(name, "pass",
                     f"{subject} matches RTL {detail}".rstrip())
    if status == "differ":
        return Check(name, "fail",
                     f"{subject} differs from RTL: {detail}",
                     "Toolchain miscompiled the design; do not flash. Report the "
                     "counterexample.")
    if status == "inconclusive":
        return Check(name, "warn",
                     f"{tier} equivalence not established (proof inconclusive)",
                     "Design likely too large/memory-heavy for the prover; rely "
                     "on cycle-accurate verify and extend its coverage.")
    return None  # skipped -> omit the check entirely


def _run_equivalence(rtl_files, top, target_fpga, clock_ns, cycles, rtl_hash):
    """Prove (or, failing that, verify) that the bitstream implements the RTL.

    Returns ``(status, detail, bitstream_path, confidence)`` where ``confidence``
    is a float in [0,1] for verified campaigns, else ``None``.
    """
    from .backends.base import Design
    from .emulator import Emulator
    from .virtual.board import BringUpConfig

    clock_mhz = 1000.0 / clock_ns if clock_ns > 0 else 50.0
    eng = Emulator()
    design = Design(rtl_files=tuple(rtl_files), top=top, target=target_fpga)
    try:
        p = eng.prove_equivalence(
            design, clock_mhz=clock_mhz, depth=min(max(cycles, 8), 24),
            unbounded=True, workdir=Path(".runs") / "assess_prove" / rtl_hash,
        )
    except Exception as exc:  # noqa: BLE001 - never let this break the gate
        return "inconclusive", str(exc), None, None

    if p.equivalent is True:
        if p.unbounded:
            return "proved_all", "", p.bitstream_path, 1.0
        return ("proved_bounded", f"over {p.depth} cycles from reset (all inputs)",
                p.bitstream_path, 1.0)
    if p.equivalent is False:
        return "differ", p.counterexample or "inputs exist where they differ", p.bitstream_path, 0.0

    # Proof inconclusive (e.g. memory-heavy) -> fall back to a multi-seed,
    # coverage-measured verification campaign (the strongest evidence short of
    # a formal proof).
    try:
        v = eng.verify_bitstream(
            design, cycles=max(cycles, 128), clock_mhz=clock_mhz,
            adaptive=True,
            config=BringUpConfig(cycles=max(cycles, 128), stimulus="random"),
            workdir=Path(".runs") / "assess_verify" / rtl_hash,
        )
    except Exception:  # noqa: BLE001
        return "inconclusive", "", p.bitstream_path, None
    if v.error:
        return "inconclusive", v.error, p.bitstream_path, None
    if not v.matches:
        return "differ", v.first_mismatch or "cycle mismatch", v.bitstream_path, 0.0
    if v.total_compared == 0:
        # Matched only don't-care cycles -> no real evidence gathered.
        return "inconclusive", "no concrete output cycles were exercised", v.bitstream_path, None
    sat = ", coverage saturated" if v.coverage_saturated else ""
    detail = (
        f"across {v.total_compared} concrete cycles over {len(v.seeds)} random seed(s), "
        f"{v.toggle_coverage:.0%} output-bit coverage{sat} "
        f"(empirical confidence {v.confidence:.0%})"
    )
    return "verified", detail, v.bitstream_path, float(v.confidence)


def _run_netlist_equivalence(rtl_files, top, target_fpga, netlist_path, clock_ns,
                             cycles, rtl_hash):
    """Prove RTL == a vendor post-implementation netlist (netlist tier).

    Used for vendor-locked silicon whose bitstream cannot be reconstructed but
    whose backend emitted a gate-level Verilog netlist. Returns the same
    ``(status, detail, artifact, confidence)`` shape as ``_run_equivalence``.
    """
    from .backends.base import Design
    from .emulator import Emulator

    clock_mhz = 1000.0 / clock_ns if clock_ns > 0 else 50.0
    eng = Emulator()
    design = Design(rtl_files=tuple(rtl_files), top=top, target=target_fpga)
    try:
        p = eng.prove_equivalence(
            design, clock_mhz=clock_mhz, depth=min(max(cycles, 8), 24),
            unbounded=True, netlist=netlist_path,
            workdir=Path(".runs") / "assess_prove_nl" / rtl_hash,
        )
    except Exception as exc:  # noqa: BLE001 - never let this break the gate
        return "inconclusive", str(exc), None, None

    if p.equivalent is True:
        if p.unbounded:
            return "proved_all", "against the post-implementation netlist", str(netlist_path), 1.0
        return ("proved_bounded",
                f"against the post-impl netlist over {p.depth} cycles from reset",
                str(netlist_path), 1.0)
    if p.equivalent is False:
        return "differ", p.counterexample or "RTL and netlist differ", str(netlist_path), 0.0
    return "inconclusive", p.error or "", str(netlist_path), None


def _run_pin_check(pins, run, top, dev, board_file, clock_ns):
    """Validate pin constraints against the design's real ports/package/board.

    Returns ``(status, detail)`` where status is one of ``"valid"``,
    ``"invalid"``, ``"missing"``, ``"skipped"``.
    """
    if pins is None:
        return "missing", ""

    from .pins import check_pins, load_pins

    try:
        pc = load_pins(pins)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the gate
        return "invalid", f"could not parse {pins}: {exc}"

    # Real top-level ports come from the synthesized netlist.
    ports = None
    if run and run.workdir:
        nl_file = Path(run.workdir) / "netlist.json"
        if nl_file.exists():
            try:
                import json as _json

                from .virtual.vfpga import _ports_from_netlist

                ports = _ports_from_netlist(_json.loads(nl_file.read_text()), top)
            except Exception:  # noqa: BLE001
                ports = None
    if not ports:
        return "skipped", "could not extract top-level ports to validate pins"

    # Package pin database (iCE40 via the IceStorm chipdb, when installed).
    valid_pins = None
    if dev is not None and dev.chipdb_tag:
        try:
            from .emulator import netlist as _nl

            chipdb = _nl.find_chipdb(dev.chipdb_tag)
            if chipdb is not None:
                valid_pins = set(_nl.parse_package_pins(chipdb, dev.package))
        except Exception:  # noqa: BLE001
            valid_pins = None

    board = None
    if board_file is not None:
        try:
            import json as _json

            board = _json.loads(Path(board_file).read_text())
        except Exception as exc:  # noqa: BLE001
            return "invalid", f"could not parse board spec {board_file}: {exc}"

    from .virtual.board import detect_clock

    clock_port = detect_clock(ports, None)
    rep = check_pins(
        pc, ports, valid_pins=valid_pins,
        io_capacity=dev.io if dev else None,
        board=board, clock_port=clock_port.name if clock_port else None,
        clock_ns=clock_ns,
    )
    if not rep.ok:
        return "invalid", "; ".join(rep.errors[:4])
    detail = (f"{rep.constrained_ports}/{rep.total_port_bits} port bits pinned, "
              f"validated against the package"
              + (" and board spec" if board else ""))
    if rep.warnings:
        detail += "; " + "; ".join(rep.warnings[:2])
    return "valid", detail


def score_and_verdict(checks: list[Check]) -> tuple[int, str]:
    fails = sum(1 for c in checks if c.status == "fail")
    warns = sum(1 for c in checks if c.status == "warn")
    score = max(0, 100 - 35 * fails - 12 * warns)
    if fails:
        verdict = BLOCKED
        score = min(score, 40)
    elif warns:
        verdict = AT_RISK
        score = min(score, 85)
    else:
        verdict = READY
    return score, verdict


def assess(
    rtl: str | Sequence[str],
    top: str,
    target_fpga: str = "ice40_up5k",
    clock_ns: float = 10.0,
    cycles: int = 64,
    iterations: int = 6,
    optimize: bool = True,
    testbench: str | Path | None = None,
    prove_equivalence: bool = True,
    sdc: str | Path | None = None,
    pins: str | Path | None = None,
    require_pins: bool = False,
    board_file: str | Path | None = None,
    backend=None,
) -> ReadinessReport:
    """Assess whether a design can reach the FPGA first shot.

    Runs (optionally optimizing) implementation, a virtual bring-up,
    design-rule checks, structural clock-domain-crossing analysis, and -- when
    feasible -- a bitstream-level equivalence check (formal proof, falling back
    to cycle-accurate verify), then fuses them into a verdict + confidence +
    fixes. Pass ``sdc`` to honor real timing constraints (multiple clocks,
    false/multicycle paths).

    Pass ``pins`` (a .pcf/.xdc/.qsf/.lpf file) to validate the board pin map
    against the design's real ports and package -- and, on iCE40, to constrain
    the actual place & route to those pins. ``require_pins=True`` makes a
    missing pin map a hard failure. ``board_file`` additionally cross-checks
    the clock source and voltage rails from a board spec JSON.
    """
    from .backends.base import Design, FlowOptions
    from .optimizer import default_backend, optimize as run_optimize
    from .virtual.vfpga import VirtualFPGA, bringup as run_bringup

    rtl_files = [rtl] if isinstance(rtl, str) else list(rtl)
    pcf = str(pins) if pins is not None and str(pins).endswith(".pcf") else None
    design = Design(rtl_files=tuple(rtl_files), top=top, target=target_fpga,
                    clock_ns=clock_ns, pcf=pcf)

    if backend is None:
        backend = default_backend(target_fpga)
    simulated = getattr(backend, "name", "") == "mock"

    # ---- Implementation (optimize to auto-fix where possible) ----
    if optimize:
        opt = run_optimize(
            rtl=rtl_files, top=top, target_fpga=target_fpga,
            objective="maximize_fmax", iterations=iterations,
            clock_ns=clock_ns, backend=backend, pcf=pcf,
        )
        run = opt.best
    else:
        workdir = Path(".runs") / "assess" / design.rtl_hash()
        run = backend.run(design, FlowOptions(), workdir)

    # ---- Virtual bring-up ----
    vf = VirtualFPGA()
    bringup_status = "skipped"
    bu = None
    if vf.is_available():
        bu = run_bringup(
            rtl=rtl_files, top=top, target_fpga=target_fpga,
            cycles=cycles, testbench=testbench,
            workdir=Path(".runs") / "assess_bringup" / design.rtl_hash(),
            engine=vf,
        )
        bringup_status = "up" if bu.success else "down"
        io_count = sum(p.width for p in bu.ports)
    else:
        io_count = 0

    # ---- Bitstream-level equivalence (real tools, when the design fits) ----
    metrics = run.metrics if run else None
    equivalence_status = "skipped"
    equivalence_detail = ""
    equivalence_bitstream: str | None = None
    equivalence_confidence: float | None = None
    equivalence_tier_used = "bitstream"
    from .devices import get as _dev_get
    from .emulator.reconstruct import achievable_tier, reconstructor_for

    dev = _dev_get(target_fpga)
    # Bit-level bring-up needs the device format to be open AND the decoding
    # tools installed (IceStorm / Project X-Ray / Apicula).
    recon = reconstructor_for(target_fpga)
    reconstructable = bool(dev and dev.reconstructable and recon.available)
    if (
        prove_equivalence
        and not simulated
        and metrics is not None
        and metrics.routed_ok
        and reconstructable
        and target_fpga in DEVICE_IO_CAPACITY
        and 0 < io_count <= DEVICE_IO_CAPACITY[target_fpga]
    ):
        (equivalence_status, equivalence_detail, equivalence_bitstream,
         equivalence_confidence) = _run_equivalence(
            rtl_files, top, target_fpga, clock_ns, cycles, design.rtl_hash()
        )
    elif prove_equivalence and not simulated and dev is not None and not reconstructable:
        # Vendor-locked silicon: bit-level bring-up is impossible; the strongest
        # tier is netlist-level equivalence. If the backend emitted a gate-level
        # Verilog netlist and we know the vendor sim library, prove RTL == it.
        nl = getattr(run, "routed_netlist_path", None) if run else None
        nl_ok = (
            nl is not None and dev.sim_lib
            and str(nl).lower().endswith((".v", ".vo", ".vg"))
            and Path(nl).exists()
        )
        if nl_ok:
            equivalence_tier_used = "netlist"
            (equivalence_status, equivalence_detail, equivalence_bitstream,
             equivalence_confidence) = _run_netlist_equivalence(
                rtl_files, top, target_fpga, nl, clock_ns, cycles, design.rtl_hash()
            )
        else:
            equivalence_status = "skipped"
            equivalence_detail = recon.why_unavailable(target_fpga)

    # ---- Pin constraints (board-level: right pins, clock source, rails) ----
    pin_status, pin_detail = "skipped", ""
    if pins is not None or require_pins:
        pin_status, pin_detail = _run_pin_check(
            pins, run, top, dev, board_file, clock_ns)

    # ---- Structural CDC analysis on the synthesized netlist ----
    cdc_report = None
    if run and run.workdir:
        netlist_file = Path(run.workdir) / "netlist.json"
        if netlist_file.exists():
            try:
                import json as _json

                from .cdc import analyze_cdc

                netlist = _json.loads(netlist_file.read_text())
                constraints = None
                if sdc is not None:
                    from .constraints import load_sdc

                    constraints = load_sdc(sdc)
                cdc_report = analyze_cdc(netlist, top=top, constraints=constraints)
            except Exception:  # noqa: BLE001 - never let CDC break the gate
                cdc_report = None

    # ---- Facts -> checks -> verdict ----
    rtl_text = "\n".join(
        Path(f).read_text() if Path(f).exists() else str(f) for f in rtl_files
    )
    checks = evaluate(
        synthesized=bool(run and (run.metrics.luts > 0 or run.success)),
        routed_ok=bool(metrics and metrics.routed_ok),
        fmax_mhz=float(metrics.fmax_mhz) if metrics else 0.0,
        target_mhz=design.target_freq_mhz,
        luts=int(metrics.luts) if metrics else 0,
        lut_capacity=DEVICE_LUT_CAPACITY.get(target_fpga, 0),
        io_count=io_count,
        io_capacity=DEVICE_IO_CAPACITY.get(target_fpga, 0),
        bringup_status=bringup_status,
        synth_log=run.log if run else "",
        rtl_text=rtl_text,
        bitstream_equivalence=equivalence_status,
        equivalence_detail=equivalence_detail,
        equivalence_tier_used=equivalence_tier_used,
        pin_status=pin_status,
        pin_detail=pin_detail,
        cdc_report=cdc_report,
    )
    score, verdict = score_and_verdict(checks)

    # Collect the actual tool errors/warnings so the user can fix them directly.
    from .diagnostics import extract

    diag_errors: list[str] = []
    diag_warnings: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for source in (run, bu):
        if source is None:
            continue
        for d in extract(source.log):
            # De-dup across runs by filename (temp workdir prefixes differ).
            loc = Path(d.location).name if d.location else ""
            key = (d.severity, d.message, loc)
            if key in seen:
                continue
            seen.add(key)
            (diag_errors if d.severity == "error" else diag_warnings).append(d.format())

    lut_capacity = DEVICE_LUT_CAPACITY.get(target_fpga, 0)
    io_capacity = DEVICE_IO_CAPACITY.get(target_fpga, 0)
    luts = int(metrics.luts) if metrics else 0

    # Structured critical path (the actionable "where to pipeline" for an agent).
    critical_path = None
    if metrics and metrics.routed_ok and run:
        from .timing import parse_critical_paths

        paths = parse_critical_paths(run.log)
        if paths:
            intra = [p for p in paths if "cross-domain" not in p.clock.lower()
                     and "async" not in p.clock.lower()]
            wp = max(intra or paths, key=lambda p: p.total_ns)
            src = next((s.detail for s in wp.stages if s.detail), "")
            snk = next((s.detail for s in reversed(wp.stages) if s.detail), "")
            critical_path = {
                "clock": wp.clock,
                "total_ns": round(wp.total_ns, 3),
                "logic_ns": round(wp.logic_ns, 3),
                "routing_ns": round(wp.routing_ns, 3),
                "logic_stages": wp.n_logic_stages,
                "from": src,
                "to": snk,
            }

    report_metrics = {
        "synthesized": bool(run and (run.metrics.luts > 0 or run.success)),
        "routed_ok": bool(metrics and metrics.routed_ok),
        "fmax_mhz": float(metrics.fmax_mhz) if metrics else 0.0,
        "target_mhz": design.target_freq_mhz,
        "luts": luts,
        "lut_capacity": lut_capacity,
        "lut_util": (luts / lut_capacity) if lut_capacity else 0.0,
        "ffs": int(metrics.ffs) if metrics else 0,
        "bram": int(metrics.bram) if metrics else 0,
        "dsp": int(metrics.dsp) if metrics else 0,
        "io_count": io_count,
        "io_capacity": io_capacity,
        "target_device": target_fpga,
        "bringup_status": bringup_status,
        "equivalence_status": equivalence_status,
        "equivalence_confidence": equivalence_confidence,
        "equivalence_detail": equivalence_detail,
        "critical_path": critical_path,
        "has_comb_loop": _has_comb_loop(run.log if run else ""),
        "has_latch": _has_latch(run.log if run else ""),
        "clock_domains": (cdc_report.n_domains if cdc_report is not None
                          else (count_clock_domains(rtl_text) if rtl_text else 0)),
        "cdc_worst": cdc_report.worst if cdc_report is not None else "unknown",
        "cdc_unsynchronized": (len(cdc_report.unsynchronized)
                               if cdc_report is not None else 0),
        "cdc_single_flop": (len(cdc_report.single_flop)
                            if cdc_report is not None else 0),
        "cdc_detail": (cdc_report.summary() if cdc_report is not None else ""),
        # Real, located tool messages so an agent can fix the exact line.
        "tool_errors": diag_errors,
        "tool_warnings": diag_warnings[:10],
    }

    return ReadinessReport(
        design_id=design.design_id(),
        verdict=verdict,
        score=score,
        checks=checks,
        simulated=simulated,
        fmax_mhz=float(metrics.fmax_mhz) if metrics else 0.0,
        target_mhz=design.target_freq_mhz,
        target_device=target_fpga,
        vendor=dev.vendor if dev else "",
        equivalence_tier=achievable_tier(dev),
        bitstream_path=run.bitstream_path if run else None,
        diagnostics=diag_errors,
        tool_warnings=diag_warnings[:10],
        metrics=report_metrics,
    )
