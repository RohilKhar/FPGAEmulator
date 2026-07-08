"""GowinBackend: the open-source Gowin LittleBee flow (Project Apicula).

Pipeline: yosys (synth_gowin) -> nextpnr-himbaechel (Gowin arch; the older
nextpnr-gowin binary is accepted too) -> gowin_pack (Apicula) -> ``.fs``
bitstream. Because Apicula also ships ``gowin_unpack``, the flashed bitstream
can be decoded *back* to a netlist -- lifting Gowin parts to the same
bit-level equivalence tier as iCE40 (see ``emulator/reconstruct.py``).

Like the other backends, every step is captured and failures return a
RunResult with success=False rather than raising; ``is_available()`` gates on
the tools so the platform degrades to MockBackend when they're missing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .. import features as feat
from .. import reports
from .. import rtl_transform
from .base import Backend, Design, FlowOptions, RunMetrics, RunResult

from ..devices import by_backend

# target -> (apicula device name, default package). From the central registry.
_DEVICES: dict[str, tuple[str, str]] = {
    d.target: (d.part, d.package) for d in by_backend("gowin")
}


def _find_nextpnr() -> str | None:
    for exe in ("nextpnr-himbaechel", "nextpnr-gowin"):
        if shutil.which(exe):
            return exe
    return None


class GowinBackend(Backend):
    name = "gowin"

    def __init__(
        self,
        yosys: str = "yosys",
        nextpnr: str | None = None,
        gowin_pack: str = "gowin_pack",
        timeout_s: int = 900,
    ) -> None:
        self.yosys = yosys
        self.nextpnr = nextpnr or _find_nextpnr() or "nextpnr-himbaechel"
        self.gowin_pack = gowin_pack
        self.timeout_s = timeout_s

    def is_available(self) -> bool:
        return (shutil.which(self.yosys) is not None
                and shutil.which(self.nextpnr) is not None
                and shutil.which(self.gowin_pack) is not None)

    def run(self, design: Design, options: FlowOptions, workdir: Path) -> RunResult:
        workdir.mkdir(parents=True, exist_ok=True)
        workdir = workdir.resolve()
        result = RunResult(
            design_id=design.design_id(),
            options=options,
            metrics=RunMetrics(target_freq_mhz=design.target_freq_mhz),
            backend=self.name,
            workdir=str(workdir),
        )

        if design.target not in _DEVICES:
            result.error = f"unsupported gowin target: {design.target}"
            return result
        device, _package = _DEVICES[design.target]

        rtl_files = rtl_transform.prepare_rtl(design, options, workdir)
        netlist_path = workdir / "netlist.json"
        routed_path = workdir / "routed.json"
        fs_path = workdir / "out.fs"

        # ---- Synthesis (yosys synth_gowin) ----
        synth_log, netlist, ok = self._run_yosys(
            rtl_files, design.top, options, netlist_path, workdir
        )
        result.log += synth_log
        if not ok or netlist is None:
            result.error = "synthesis failed"
            return result

        result.features = feat.from_yosys_json(netlist, design.top)
        yosys_counts = reports.parse_yosys_stat_text(synth_log)

        # ---- Place & route + timing (nextpnr, himbaechel Gowin arch) ----
        pnr_log, _ = self._run_nextpnr(
            netlist_path, routed_path, device, design, options, workdir
        )
        result.log += "\n" + pnr_log
        result.metrics = reports.build_metrics(
            nextpnr_log=pnr_log,
            yosys_stat=yosys_counts,
            target_freq_mhz=design.target_freq_mhz,
        )
        if routed_path.exists():
            result.routed_netlist_path = str(routed_path)

        if not result.metrics.routed_ok and not routed_path.exists():
            result.error = "place-and-route failed"
            result.success = False
            return result

        # ---- Bitstream (gowin_pack, Apicula) ----
        if shutil.which(self.gowin_pack) and routed_path.exists():
            pack_log, pack_ok = self._run_cmd(
                [self.gowin_pack, "-d", device, "-o", str(fs_path),
                 str(routed_path)], workdir
            )
            result.log += "\n" + pack_log
            if pack_ok and fs_path.exists():
                result.bitstream_path = str(fs_path)

        result.success = result.metrics.routed_ok or routed_path.exists()
        return result

    # ------------------------------------------------------------------ #
    def _run_yosys(self, rtl_files, top, options, netlist_path, workdir):
        synth_flags = []
        if options.retime:
            synth_flags.append("-retime")
        synth_flags_str = " ".join(synth_flags)

        read = "\n".join(f"read_verilog {f}" for f in rtl_files)
        script = (
            f"{read}\n"
            f"synth_gowin -top {top} {synth_flags_str} -json {netlist_path}\n"
            f"stat\n"
        )
        script_path = workdir / "synth.ys"
        script_path.write_text(script)

        log, ok = self._run_cmd([self.yosys, "-q", "-s", str(script_path)], workdir)
        netlist = None
        if netlist_path.exists():
            try:
                netlist = json.loads(netlist_path.read_text())
            except json.JSONDecodeError:
                netlist = None
        return log, netlist, ok and netlist is not None

    def _run_nextpnr(self, netlist_path, routed_path, device, design, options,
                     workdir):
        cmd = [
            self.nextpnr,
            "--json", str(netlist_path),
            "--write", str(routed_path),
            "--freq", f"{design.target_freq_mhz:.3f}",
            "--seed", str(options.seed),
        ]
        if "himbaechel" in self.nextpnr:
            cmd += ["--device", device, "--vopt", f"family={device.split('-')[0]}"]
        else:  # legacy nextpnr-gowin
            cmd += ["--device", device]
        if design.pcf:
            # Gowin flows use .cst physical constraints; nextpnr-himbaechel
            # takes them via --vopt cst=
            cmd += ["--vopt", f"cst={Path(design.pcf).resolve()}"]
        return self._run_cmd(cmd, workdir)

    def _run_cmd(self, cmd: list[str], workdir: Path):
        try:
            proc = subprocess.run(
                cmd, cwd=str(workdir), capture_output=True, text=True,
                timeout=self.timeout_s,
            )
        except FileNotFoundError as exc:
            return f"$ {' '.join(cmd)}\n[not found] {exc}\n", False
        except subprocess.TimeoutExpired:
            return f"$ {' '.join(cmd)}\n[timeout after {self.timeout_s}s]\n", False
        log = (
            f"$ {' '.join(cmd)}\n"
            f"{proc.stdout}\n{proc.stderr}\n"
            f"[exit {proc.returncode}]\n"
        )
        return log, proc.returncode == 0
