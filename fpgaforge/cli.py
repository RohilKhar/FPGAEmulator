"""Command-line interface: `fpgaforge run | optimize | train`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

from .backends.base import Design, FlowOptions
from .backends.ice40 import Ice40Backend
from .backends.mock import MockBackend
from .corpus import Corpus
from .emulator import BoardConfig
from .emulator import emulate as run_emulate
from .emulator import emulate_board as run_board
from .emulator import prove as run_prove
from .emulator import verify_bitstream as run_verify
from .model import FmaxPredictor
from .optimizer import default_backend, optimize
from .readiness import assess as run_assess
from .timing import signoff as run_signoff
from .virtual import BringUpConfig, VirtualFPGA
from .virtual.vfpga import bringup as run_bringup

app = typer.Typer(
    add_completion=False,
    help="An AI-native FPGA implementation engine (open-source iCE40 flow).",
)


def _pick_backend(mock: bool):
    if mock:
        return MockBackend()
    return default_backend()


@app.command()
def run(
    rtl: List[str] = typer.Argument(..., help="RTL file(s)."),
    top: str = typer.Option(..., "--top", "-t", help="Top module name."),
    target: str = typer.Option("ice40_up5k", "--target", help="Device target."),
    clock_ns: float = typer.Option(10.0, "--clock-ns", help="Target clock period (ns)."),
    abc9: bool = typer.Option(False, "--abc9", help="Use abc9 timing-driven mapping."),
    retime: bool = typer.Option(False, "--retime", help="Enable synth retiming."),
    pipeline: bool = typer.Option(False, "--pipeline", help="Request output pipelining."),
    seed: int = typer.Option(1, "--seed", help="nextpnr placement seed."),
    mock: bool = typer.Option(False, "--mock", help="Force the offline MockBackend."),
    corpus_path: Path = typer.Option(
        Path("data/corpus.jsonl"), "--corpus", help="Corpus JSONL path."
    ),
) -> None:
    """Run one flow and append the outcome to the corpus."""
    backend = _pick_backend(mock)
    design = Design(rtl_files=tuple(rtl), top=top, target=target, clock_ns=clock_ns)
    options = FlowOptions(
        abc9=abc9, retime=retime, pipeline_output=pipeline, seed=seed
    )
    workdir = Path(".runs") / "single" / options.key()
    typer.echo(f"[backend={backend.name}] running {design.design_id()} ...")
    result = backend.run(design, options, workdir)
    Corpus(corpus_path).append(result, extra={"target": target})

    m = result.metrics
    if result.success:
        typer.echo(
            f"OK  Fmax={m.fmax_mhz:.2f} MHz (target {m.target_freq_mhz:.1f}) "
            f"LUTs={m.luts} FF={m.ffs} DSP={m.dsp} BRAM={m.bram} "
            f"{'MEETS' if m.meets_timing else 'MISSES'} timing"
        )
        if result.bitstream_path:
            typer.echo(f"bitstream: {result.bitstream_path}")
    else:
        typer.echo(f"FAIL {result.error}")
        for d in result.errors():
            typer.echo(f"  {d.format()}")
        raise typer.Exit(code=1)


@app.command(name="optimize")
def optimize_cmd(
    rtl: List[str] = typer.Argument(..., help="RTL file(s)."),
    top: str = typer.Option(..., "--top", "-t", help="Top module name."),
    target: str = typer.Option("ice40_up5k", "--target", help="Device target."),
    objective: str = typer.Option(
        "maximize_fmax", "--objective", help="maximize_fmax | minimize_luts."
    ),
    iterations: int = typer.Option(8, "--iterations", "-n", help="Max backend runs."),
    clock_ns: float = typer.Option(10.0, "--clock-ns", help="Target clock period (ns)."),
    mock: bool = typer.Option(False, "--mock", help="Force the offline MockBackend."),
    corpus_path: Path = typer.Option(
        Path("data/corpus.jsonl"), "--corpus", help="Corpus JSONL path."
    ),
) -> None:
    """Optimize a design's implementation toward an objective."""
    backend = _pick_backend(mock)
    result = optimize(
        rtl=rtl,
        top=top,
        target_fpga=target,
        objective=objective,
        iterations=iterations,
        clock_ns=clock_ns,
        backend=backend,
        corpus=Corpus(corpus_path),
    )
    typer.echo(result.summary())


@app.command()
def train(
    corpus_path: Path = typer.Option(
        Path("data/corpus.jsonl"), "--corpus", help="Corpus JSONL path."
    ),
    out: Path = typer.Option(
        Path("data/model.joblib"), "--out", help="Where to save the model."
    ),
) -> None:
    """Train the Fmax predictor from the collected corpus."""
    corpus = Corpus(corpus_path)
    rows = corpus.load()
    predictor = FmaxPredictor().fit(rows)
    predictor.save(out)
    status = "trained regressor" if predictor.trained else "heuristic fallback"
    typer.echo(
        f"corpus rows: {len(rows)}, usable samples: {predictor.n_samples} -> {status}"
    )
    # Honest, cross-validated accuracy so we know if the ranking is trustworthy.
    report = FmaxPredictor.evaluate(rows)
    typer.echo(report.summary())
    if predictor.route_model is not None:
        typer.echo("  routability head: trained (predicts P(routes))")
    if predictor.lut_model is not None:
        typer.echo("  resource head: trained (predicts LUT usage)")
    typer.echo(f"saved model: {out}")


@app.command()
def bootstrap(
    rtl: List[str] = typer.Argument(..., help="RTL file(s) to sweep."),
    top: str = typer.Option(..., "--top", "-t", help="Top module name."),
    target: List[str] = typer.Option(
        ["ice40_up5k"], "--target", help="Device target(s); repeatable."
    ),
    clock_ns: List[float] = typer.Option(
        [10.0], "--clock-ns", help="Target clock period(s) (ns); repeatable."
    ),
    seed: List[int] = typer.Option([1, 2], "--seed", help="Placement seed(s)."),
    max_per_design: Optional[int] = typer.Option(
        None, "--max", help="Cap knob points per design (default: all)."
    ),
    mock: bool = typer.Option(False, "--mock", help="Force the offline MockBackend."),
    corpus_path: Path = typer.Option(
        Path("data/corpus.jsonl"), "--corpus", help="Corpus JSONL path."
    ),
    workdir: Path = typer.Option(
        Path(".runs/bootstrap"), "--workdir", help="Where to write artifacts."
    ),
) -> None:
    """Sweep a design across the knob space to grow the training corpus."""
    from .bootstrap import BootstrapSpec, bootstrap_corpus

    spec = BootstrapSpec(
        rtl_files=tuple(rtl),
        top=top,
        targets=tuple(target),
        clocks_ns=tuple(clock_ns),
    )
    backend = MockBackend() if mock else None
    report = bootstrap_corpus(
        [spec],
        seeds=tuple(seed),
        backend=backend,
        corpus=Corpus(corpus_path),
        run_dir=workdir,
        max_per_design=max_per_design,
        progress=lambda line: typer.echo(f"  {line}"),
    )
    typer.echo(report.summary())


@app.command()
def bringup(
    rtl: List[str] = typer.Argument(..., help="RTL file(s)."),
    top: str = typer.Option(..., "--top", "-t", help="Top module name."),
    target: str = typer.Option("ice40_up5k", "--target", help="Device family for mapping."),
    cycles: int = typer.Option(64, "--cycles", "-n", help="Clock cycles to run after reset."),
    clock: Optional[str] = typer.Option(None, "--clock", help="Clock port name override."),
    reset: Optional[str] = typer.Option(None, "--reset", help="Reset port name override."),
    active_low: bool = typer.Option(False, "--active-low", help="Reset is active-low."),
    testbench: Optional[Path] = typer.Option(
        None, "--testbench", help="Custom testbench with your own self-checks."
    ),
    timing: bool = typer.Option(
        False, "--timing", help="Also run timing sign-off (place & route + STA)."
    ),
    clock_ns: float = typer.Option(
        10.0, "--clock-ns", help="Target clock period for --timing."
    ),
    workdir: Path = typer.Option(
        Path(".runs/bringup"), "--workdir", help="Where to write artifacts."
    ),
) -> None:
    """Virtually bring up a design: synthesize to primitives and simulate it."""
    engine = VirtualFPGA()
    if not engine.is_available():
        typer.echo("ERROR: virtual bring-up needs yosys, iverilog, and vvp on PATH")
        raise typer.Exit(code=1)
    result = run_bringup(
        rtl=rtl,
        top=top,
        target_fpga=target,
        cycles=cycles,
        clock=clock,
        reset=reset,
        reset_active_high=(False if active_low else None),
        testbench=testbench,
        workdir=workdir,
        engine=engine,
        timing=timing,
        clock_ns=clock_ns,
    )
    typer.echo(result.summary())
    if not result.success:
        raise typer.Exit(code=1)


@app.command()
def timing(
    rtl: List[str] = typer.Argument(..., help="RTL file(s)."),
    top: str = typer.Option(..., "--top", "-t", help="Top module name."),
    target: str = typer.Option("ice40_up5k", "--target", help="Device target."),
    clock_ns: float = typer.Option(10.0, "--clock-ns", help="Target clock period (ns)."),
    seed: int = typer.Option(1, "--seed", help="nextpnr placement seed."),
    workdir: Path = typer.Option(
        Path(".runs/timing"), "--workdir", help="Where to write artifacts (incl. SDF)."
    ),
) -> None:
    """Timing sign-off: real critical-path delays and slack from STA."""
    report = run_signoff(
        rtl=rtl, top=top, target_fpga=target, clock_ns=clock_ns, seed=seed, workdir=workdir
    )
    typer.echo(report.summary())
    if not report.meets_timing:
        raise typer.Exit(code=1)


@app.command()
def emulate(
    bitstream: Path = typer.Argument(..., help="Bitstream to load (.bin or .asc)."),
    module: str = typer.Option("chip", "--module", help="Reconstructed module name."),
    show_luts: int = typer.Option(
        0, "--show-luts", help="Print this many decoded LUT truth tables."
    ),
    workdir: Path = typer.Option(
        Path(".runs/emulate"), "--workdir", help="Where to write artifacts."
    ),
) -> None:
    """Load a real bitstream, decode the configured fabric, and rebuild it.

    Unpacks the exact bytes that would be flashed, decodes each logic cell's
    LUT truth table / flop / carry usage straight from the bits, and
    reconstructs a simulatable netlist of the configured fabric.
    """
    result = run_emulate(bitstream, workdir=workdir, module=module)
    typer.echo(result.summary())
    if show_luts and result.fabric is not None:
        typer.echo("decoded LUTs:")
        logic_cells = [c for c in result.fabric.cells if c.lut_used]
        for c in logic_cells[:show_luts]:
            typer.echo(
                f"  tile({c.x},{c.y}) LC{c.index} init=0x{c.lut_init:04x} "
                f"dff={c.dff_enable} carry={c.carry_enable}"
            )
            typer.echo(f"    y = {c.lut_equation()}")
    if result.error:
        raise typer.Exit(code=1)


@app.command()
def verify(
    rtl: List[str] = typer.Argument(..., help="RTL file(s)."),
    top: str = typer.Option(..., "--top", "-t", help="Top module name."),
    target: str = typer.Option("ice40_up5k", "--target", help="Device target."),
    cycles: int = typer.Option(32, "--cycles", "-n", help="Cycles to compare after reset."),
    clock_mhz: float = typer.Option(50.0, "--clock-mhz", help="P&R target clock (MHz)."),
    clock: Optional[str] = typer.Option(None, "--clock", help="Clock port name override."),
    reset: Optional[str] = typer.Option(None, "--reset", help="Reset port name override."),
    active_low: bool = typer.Option(False, "--active-low", help="Reset is active-low."),
    stimulus: str = typer.Option(
        "counter", "--stimulus", help="Input stimulus: counter | random."
    ),
    seeds: Optional[List[int]] = typer.Option(
        None, "--seed", help="Random-campaign seed(s); repeat for multiple. "
        "Ignored for counter stimulus."
    ),
    adaptive: bool = typer.Option(
        False, "--adaptive", help="Keep adding random seeds until coverage "
        "saturates (no new behavior). Overrides fixed seeds."
    ),
    max_seeds: int = typer.Option(
        24, "--max-seeds", help="Cap for the adaptive campaign."
    ),
    workdir: Path = typer.Option(
        Path(".runs/verify"), "--workdir", help="Where to write artifacts."
    ),
) -> None:
    """First-shot check: does the flashed bitstream behave like the design?

    Runs RTL all the way to a real bitstream, unpacks that binary back into the
    configured fabric, and simulates both the bitstream and the design under
    identical stimulus. Random stimulus runs a multi-seed campaign and reports
    measured output-bit toggle coverage; uninitialized (don't-care) cycles are
    skipped rather than counted as mismatches.
    """
    result = run_verify(
        rtl=rtl,
        top=top,
        target_fpga=target,
        cycles=cycles,
        clock_mhz=clock_mhz,
        clock=clock,
        reset=reset,
        reset_active_high=(False if active_low else None),
        stimulus=stimulus,
        seeds=(list(seeds) if seeds else None),
        adaptive=adaptive,
        max_seeds=max_seeds,
        workdir=workdir,
    )
    typer.echo(result.summary())
    if not result.matches:
        raise typer.Exit(code=1)


@app.command()
def board(
    rtl: List[str] = typer.Argument(..., help="RTL file(s)."),
    top: str = typer.Option(..., "--top", "-t", help="Top module name."),
    target: str = typer.Option("ice40_up5k", "--target", help="Device target."),
    clock_mhz: float = typer.Option(12.0, "--clock-mhz", help="Board oscillator (MHz)."),
    baud: int = typer.Option(1_000_000, "--baud", help="UART bit rate to decode/inject."),
    duration_us: float = typer.Option(60.0, "--duration-us", help="How long to run (us)."),
    uart_send: Optional[str] = typer.Option(
        None, "--uart-send", help="ASCII string to transmit into the DUT's rx pin."
    ),
    workdir: Path = typer.Option(
        Path(".runs/board"), "--workdir", help="Where to write artifacts."
    ),
) -> None:
    """Run the flashed bitstream on a virtual board of peripherals.

    Builds the real bitstream, reconstructs the fabric, wires it to behavioral
    clock/reset/LED/UART/button/switch/GPIO models, runs it, and reports what
    the peripherals saw -- decoded UART text, LED blink counts, GPIO levels.
    """
    cfg = BoardConfig(
        clock_mhz=clock_mhz, baud=baud, duration_us=duration_us,
        uart_rx_bytes=[ord(c) for c in uart_send] if uart_send else [],
    )
    result = run_board(rtl=rtl, top=top, target_fpga=target, board=cfg, workdir=workdir)
    typer.echo(result.summary())
    if result.error:
        raise typer.Exit(code=1)


@app.command()
def prove(
    rtl: List[str] = typer.Argument(..., help="RTL file(s)."),
    top: str = typer.Option(..., "--top", "-t", help="Top module name."),
    target: str = typer.Option("ice40_up5k", "--target", help="Device target."),
    depth: int = typer.Option(20, "--depth", "-d", help="BMC / induction depth (cycles)."),
    bounded: bool = typer.Option(
        False, "--bounded", help="Bounded proof only (skip unbounded induction)."
    ),
    clock_mhz: float = typer.Option(50.0, "--clock-mhz", help="P&R target clock (MHz)."),
    clock: Optional[str] = typer.Option(None, "--clock", help="Clock port name override."),
    reset: Optional[str] = typer.Option(None, "--reset", help="Reset port name override."),
    strategy: str = typer.Option(
        "auto", "--strategy", help="Proof engine: auto | sat | smt. 'smt' keeps "
        "memories as arrays (scales to memory designs); 'auto' picks smt for "
        "memory designs when yosys-smtbmc + a solver are installed."
    ),
    netlist: Optional[Path] = typer.Option(
        None, "--netlist", help="Prove RTL == this gate-level netlist (e.g. Vivado "
        "'write_verilog -mode funcsim' or Quartus .vo) instead of a bitstream. "
        "Netlist-level tier for vendor-locked devices (AMD/Intel)."
    ),
    workdir: Path = typer.Option(
        Path(".runs/prove"), "--workdir", help="Where to write artifacts."
    ),
) -> None:
    """Formally prove the flashed bitstream equals the RTL for ALL inputs.

    Builds a real bitstream, reconstructs the configured fabric, and runs a
    miter proof. The SAT engine bit-blasts (fast for logic, explodes on memory);
    the SMT engine keeps memories as arrays so memory-heavy designs stay
    tractable. Stronger than `verify`, which only checks one stimulus sequence.

    With --netlist, proves RTL == the given post-implementation netlist instead
    (netlist-level equivalence), the strongest tier on vendor-locked silicon
    whose bitstream cannot be reconstructed.
    """
    result = run_prove(
        rtl=rtl,
        top=top,
        target_fpga=target,
        depth=depth,
        unbounded=not bounded,
        clock_mhz=clock_mhz,
        clock=clock,
        reset=reset,
        workdir=workdir,
        strategy=strategy,
        netlist=netlist,
    )
    typer.echo(result.summary())
    if result.equivalent is False:
        raise typer.Exit(code=1)


@app.command()
def physics(
    rtl: List[str] = typer.Argument(..., help="RTL file(s)."),
    top: str = typer.Option(..., "--top", "-t", help="Top module name."),
    target: str = typer.Option("ice40_up5k", "--target", help="Device target."),
    clock_ns: float = typer.Option(10.0, "--clock-ns", help="Target clock period (ns)."),
    fmax: Optional[float] = typer.Option(
        None, "--fmax", help="Skip P&R; use this STA Fmax (MHz) directly."
    ),
    sta_corner: str = typer.Option(
        "slow", "--sta-corner", help="Corner the STA Fmax represents: slow|typ|fast. "
        "nextpnr-ice40 is slow (already worst-case)."
    ),
    board_file: Optional[Path] = typer.Option(
        None, "--board-file", help="JSON describing nets/interfaces/package for "
        "signal-integrity and interface budgets."
    ),
    seed: int = typer.Option(1, "--seed", help="nextpnr placement seed."),
    vcd: Optional[Path] = typer.Option(
        None, "--vcd", help="Measure switching activity from this waveform (from a "
        "prior verify/bringup/board run) instead of the model default."
    ),
    measure_activity_flag: bool = typer.Option(
        False, "--measure-activity", help="Run a virtual bring-up to measure real "
        "switching activity for the power estimate."
    ),
    workdir: Path = typer.Option(
        Path(".runs/physics"), "--workdir", help="Where to write artifacts."
    ),
) -> None:
    """Physical sign-off: PVT-derated Fmax, signal integrity, interface budgets.

    Models the effects beyond the netlist -- process/voltage/temperature timing
    derating, I/O overshoot / ground bounce / settling, and external-interface
    setup/hold eye budgets. Supply board details via --board-file for the SI and
    interface analyses; PVT runs from the STA Fmax alone.
    """
    from .physics import physical_signoff
    from .physics.pvt import PVTConfig
    from .physics.signal_integrity import Net, PackageModel
    from .physics.interfaces import InterfaceBudget
    from .physics.crosstalk import CrosstalkPair
    from .physics.ibis import load_ibis, net_from_ibis, package_from_ibis
    from .physics.fieldsolver import (
        StackupGeometry, net_from_geometry, crosstalk_from_geometry,
    )
    from .physics.pdn import DecouplingCap, PDNConfig
    from .physics.power import PowerConfig

    if fmax is not None:
        fmax_sta, target_mhz = fmax, (1000.0 / clock_ns if clock_ns > 0 else 0.0)
    else:
        report = run_signoff(rtl=rtl, top=top, target_fpga=target,
                             clock_ns=clock_ns, seed=seed, workdir=workdir)
        typer.echo(report.summary())
        typer.echo("")
        fmax_sta, target_mhz = report.fmax_mhz, report.target_mhz
        if fmax_sta <= 0:
            typer.echo("no STA Fmax available; cannot run PVT derating")
            raise typer.Exit(code=1)

    nets: list = []
    interfaces: list = []
    crosstalk: list = []
    package = None
    pvt_cfg: PVTConfig | None = None
    pdn_cfg: PDNConfig | None = None
    resources: dict | None = None
    io_count_pwr = 0
    power_cfg: PowerConfig | None = None
    activity = None
    if board_file is not None:
        spec = json.loads(Path(board_file).read_text())
        base = Path(board_file).parent
        sta_corner = spec.get("sta_corner", sta_corner)
        if "package" in spec:
            package = _from_dict(PackageModel, spec["package"])
        if "pvt" in spec:
            pvt_cfg = _from_dict(PVTConfig, spec["pvt"])
        for n in spec.get("nets", []):
            # A net may reference an IBIS model, a stackup geometry, or raw values.
            if "geometry" in n:
                geom = _from_dict(StackupGeometry, n["geometry"])
                nets.append(net_from_geometry(
                    geom, name=n.get("name", "io"),
                    trace_len_mm=n.get("trace_len_mm", 50.0),
                    load_pf=n.get("load_pf", 10.0),
                    vdd=n.get("vdd", 3.3),
                    drive_impedance_ohm=n.get("drive_impedance_ohm", 40.0),
                    driver_rise_ns=n.get("driver_rise_ns", 1.0),
                    n_simultaneous=n.get("n_simultaneous", 1),
                ))
            elif "ibis" in n:
                ib = n["ibis"]
                path = ib["file"] if Path(ib["file"]).is_absolute() else base / ib["file"]
                model = load_ibis(path, ib.get("model"))
                net = net_from_ibis(
                    model, name=n.get("name", "io"),
                    trace_z0_ohm=n.get("trace_z0_ohm", 50.0),
                    trace_len_mm=n.get("trace_len_mm", 50.0),
                    load_pf=n.get("load_pf", 10.0),
                    n_simultaneous=n.get("n_simultaneous", 1),
                )
                nets.append(net)
                if package is None:
                    package = package_from_ibis(model)
            else:
                nets.append(_from_dict(Net, n))
        interfaces = [_from_dict(InterfaceBudget, i) for i in spec.get("interfaces", [])]
        for x in spec.get("crosstalk", []):
            # A crosstalk pair may derive its coupling from stackup geometry.
            if "geometry" in x:
                geom = _from_dict(StackupGeometry, x["geometry"])
                crosstalk.append(crosstalk_from_geometry(
                    geom, victim=x.get("victim", "victim"),
                    coupling_len_mm=x.get("coupling_len_mm", 25.0),
                    aggressor_rise_ns=x.get("aggressor_rise_ns", 1.0),
                    vdd=x.get("vdd", 3.3),
                    n_aggressors=x.get("n_aggressors", 1),
                    noise_margin_v=x.get("noise_margin_v"),
                ))
            else:
                crosstalk.append(_from_dict(CrosstalkPair, x))
        if "pdn" in spec:
            pd = spec["pdn"]
            caps = [_from_dict(DecouplingCap, c) for c in pd.get("caps", [])]
            pdn_cfg = _from_dict(PDNConfig, {k: v for k, v in pd.items() if k != "caps"})
            pdn_cfg.caps = caps
        if "power" in spec:
            pw = spec["power"]
            resources = pw.get("resources", {})
            io_count_pwr = pw.get("io_count", 0)
            activity = pw.get("activity")
            cfg_keys = {k: v for k, v in pw.items()
                        if k not in ("resources", "io_count", "activity")}
            power_cfg = _from_dict(PowerConfig, cfg_keys) if cfg_keys else None

    # Resolve a waveform to measure activity from (explicit --vcd, or run a
    # bring-up when --measure-activity is set). An explicit activity in the
    # board file still wins.
    vcd_path = str(vcd) if vcd else None
    if measure_activity_flag and vcd_path is None and activity is None:
        br = run_bringup(rtl=rtl, top=top, target_fpga=target, cycles=64,
                         clock=None, reset=None, workdir=workdir / "bringup")
        if br.vcd_path:
            vcd_path = br.vcd_path
            typer.echo(f"measured activity from bring-up waveform: {vcd_path}\n")
        else:
            typer.echo("could not produce a waveform; using model default activity\n")

    result = physical_signoff(
        fmax_sta_mhz=fmax_sta, target_mhz=target_mhz, design_id=top,
        pvt_cfg=pvt_cfg, sta_corner=sta_corner, nets=nets, package=package,
        interfaces=interfaces, crosstalk=crosstalk, pdn=pdn_cfg,
        resources=resources, io_count=io_count_pwr, power_cfg=power_cfg,
        activity=activity, vcd=vcd_path,
    )
    typer.echo(result.summary())
    if result.verdict == "BLOCKED":
        raise typer.Exit(code=1)


def _from_dict(cls, data: dict):
    """Build a dataclass from a dict, ignoring unknown keys."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in fields})


@app.command()
def assess(
    rtl: List[str] = typer.Argument(..., help="RTL file(s)."),
    top: str = typer.Option(..., "--top", "-t", help="Top module name."),
    target: str = typer.Option("ice40_up5k", "--target", help="Device target."),
    clock_ns: float = typer.Option(10.0, "--clock-ns", help="Target clock period (ns)."),
    cycles: int = typer.Option(64, "--cycles", help="Bring-up cycles to simulate."),
    iterations: int = typer.Option(6, "--iterations", "-n", help="Optimization attempts."),
    no_optimize: bool = typer.Option(False, "--no-optimize", help="Single run, no knob search."),
    no_prove: bool = typer.Option(
        False, "--no-prove", help="Skip the bitstream equivalence proof/verify step."
    ),
    testbench: Optional[Path] = typer.Option(
        None, "--testbench", help="Custom bring-up testbench with self-checks."
    ),
    pins: Optional[Path] = typer.Option(
        None, "--pins", help="Pin-constraint file (.pcf/.xdc/.qsf/.lpf). Validated "
        "against the design's ports, the package, and (with --board) the board "
        "spec; a .pcf also constrains the real place & route."
    ),
    require_pins: bool = typer.Option(
        False, "--require-pins", help="Fail the gate when no pin map is provided "
        "(auto-placed I/O is the classic first-shot killer)."
    ),
    board: Optional[Path] = typer.Option(
        None, "--board", help="Board spec JSON: cross-check clock sources and "
        "voltage rails against the pin map."
    ),
    mock: bool = typer.Option(False, "--mock", help="Force the offline MockBackend."),
) -> None:
    """First-pass readiness gate: can this design reach the FPGA on the first shot?"""
    report = run_assess(
        rtl=rtl,
        top=top,
        target_fpga=target,
        clock_ns=clock_ns,
        cycles=cycles,
        iterations=iterations,
        optimize=not no_optimize,
        testbench=testbench,
        prove_equivalence=not no_prove,
        pins=pins,
        require_pins=require_pins,
        board_file=board,
        backend=(MockBackend() if mock else None),
    )
    typer.echo(report.summary())
    if report.verdict == "BLOCKED":
        raise typer.Exit(code=1)


@app.command()
def selftest(
    rtl: List[str] = typer.Argument(..., help="RTL file(s)."),
    top: str = typer.Option(..., "--top", "-t", help="Top module name."),
    cycles: int = typer.Option(2048, "--cycles", help="Pseudo-random test cycles."),
    warmup: int = typer.Option(8, "--warmup", help="Reset cycles before capture."),
    target: str = typer.Option("ice40_up5k", "--target", help="Device target for --build."),
    build: bool = typer.Option(
        False, "--build", help="Also build the harness to a bitstream for --target."
    ),
    workdir: Path = typer.Option(
        Path(".runs/selftest"), "--workdir", help="Where to write artifacts."
    ),
) -> None:
    """Generate an on-FPGA self-test (BIST) harness with a golden signature.

    Wraps the design with an LFSR stimulus generator and a MISR signature
    register, computes the expected signature by cycle-accurate simulation, and
    bakes it into the harness. Flash it on real hardware: test_pass high means
    the physical silicon matched the simulation bit-for-bit -- catching
    defects, rail, clock, and thermal problems no simulator can see.
    """
    from .selftest import generate_selftest

    report = generate_selftest(rtl, top, cycles=cycles, warmup=warmup,
                               workdir=workdir)
    typer.echo(report.summary())
    if report.error:
        raise typer.Exit(code=1)
    if build:
        from .optimizer import backend_for_target
        from .backends.base import FlowOptions

        be = backend_for_target(target)
        design = Design(
            rtl_files=tuple(list(rtl) + [report.harness_path]),
            top=report.harness_top, target=target,
        )
        run = be.run(design, FlowOptions(), Path(workdir) / "build")
        if run.success and run.bitstream_path:
            typer.echo(f"harness bitstream: {run.bitstream_path}")
        else:
            typer.echo(f"harness build failed: {run.error}")
            raise typer.Exit(code=1)


@app.command(name="mutation-test")
def mutation_test_cmd(
    rtl: List[str] = typer.Argument(..., help="RTL file(s)."),
    top: str = typer.Option(..., "--top", "-t", help="Top module name."),
    target: str = typer.Option("ice40_up5k", "--target", help="Device target."),
    mutants: int = typer.Option(8, "--mutants", "-m", help="Number of bitstream mutants."),
    cycles: int = typer.Option(64, "--cycles", help="Compare cycles per mutant."),
    clock_mhz: float = typer.Option(24.0, "--clock-mhz", help="Emulation clock."),
    bits: int = typer.Option(1, "--bits", help="Bits to flip per mutant."),
    strategy: str = typer.Option(
        "netlist", "--strategy",
        help="'netlist' (functional LUT mutants) or 'bitstream' (raw bit flips).",
    ),
) -> None:
    """Validate the verifier itself: flip bitstream bits, confirm they're caught.

    Reports the kill rate -- the fraction of corrupted bitstreams the
    cycle-accurate comparison detects. A high kill rate means the verification
    has teeth; many survivors mean the stimulus is too weak.
    """
    from .emulator import mutation_test as run_mutation

    design = Design(rtl_files=tuple(rtl), top=top, target=target)
    result = run_mutation(design, n_mutants=mutants, cycles=cycles,
                          clock_mhz=clock_mhz, n_bits=bits, strategy=strategy)
    typer.echo(result.summary())
    if result.error:
        raise typer.Exit(code=1)


@app.command(name="timing-emulate")
def timing_emulate_cmd(
    rtl: List[str] = typer.Argument(..., help="RTL file(s)."),
    top: str = typer.Option(..., "--top", "-t", help="Top module name."),
    target: str = typer.Option("ice40_up5k", "--target", help="Device target."),
    clock_mhz: float = typer.Option(50.0, "--clock-mhz", help="Emulation clock (MHz)."),
    cycles: int = typer.Option(48, "--cycles", help="Functional compare cycles."),
    stimulus: str = typer.Option("counter", "--stimulus", help="'counter' or 'random'."),
) -> None:
    """Delay-aware emulation: functional match + real-delay setup at the clock.

    Fuses the zero-delay cycle-accurate verify with an independent longest-path
    timing analysis over the routed SDF, so you learn whether the flashed fabric
    both computes correctly *and* settles at the target clock -- and if not,
    which flop misses setup.
    """
    from .emulator import timing_emulate

    result = timing_emulate(rtl, top, target_fpga=target, clock_mhz=clock_mhz,
                            cycles=cycles, stimulus=stimulus)
    typer.echo(result.summary())
    if result.error or result.verdict != "TIMING-ACCURATE PASS":
        raise typer.Exit(code=1)


@app.command()
def reward(
    rtl: List[str] = typer.Argument(..., help="RTL file(s)."),
    top: str = typer.Option(..., "--top", "-t", help="Top module name."),
    target: str = typer.Option("ice40_up5k", "--target", help="Device target."),
    clock_ns: float = typer.Option(10.0, "--clock-ns", help="Target clock period (ns)."),
    cycles: int = typer.Option(64, "--cycles", help="Bring-up cycles to simulate."),
    quick: bool = typer.Option(
        False, "--quick", help="Skip bitstream equivalence for cheap rollouts."
    ),
    optimize_knobs: bool = typer.Option(
        False, "--optimize", help="Let the knob-search tune the design first "
        "(off by default so the reward scores the emitted RTL)."
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit the reward + issues as JSON (for an RL loop)."
    ),
    cache: bool = typer.Option(
        False, "--cache", help="Cache by design+flags; return instantly on repeats."
    ),
    no_physics: bool = typer.Option(
        False, "--no-physics", help="Skip modeled PVT/thermal issues."
    ),
    mock: bool = typer.Option(False, "--mock", help="Force the offline MockBackend."),
) -> None:
    """Score a design as an RL reward: shaped scalar + structured issues.

    A drop-in replacement for hooking a policy up to a real FPGA -- returns a
    dense reward in [0,1] that climbs as the design nears flashable, plus a
    machine-readable list of what to fix.
    """
    from .reward import score_design

    result = score_design(
        rtl=rtl, top=top, target_fpga=target, clock_ns=clock_ns, cycles=cycles,
        quick=quick, optimize=optimize_knobs, physics=not no_physics, cache=cache,
        backend=(MockBackend() if mock else None),
    )
    if as_json:
        typer.echo(json.dumps(result.to_dict(), indent=2))
    else:
        typer.echo(result.summary())


if __name__ == "__main__":
    app()
