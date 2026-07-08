"""Pluggable fabric reconstruction: decode a flashed bitstream back to a netlist.

Bit-level bring-up (``emulate``/``verify``/bitstream-``prove``) depends on being
able to decode the *actual* configured fabric out of the flashed bits. That is
only possible where an open bitstream database exists:

* :class:`IceStormReconstructor` -- Lattice iCE40 via Project IceStorm
  (``icebox_vlog``). This is the fully-supported path.
* :class:`XRayReconstructor` -- AMD/Xilinx 7-series via Project X-Ray
  (``bit2fasm`` + ``fasm2bels``). The bitstream format is documented by the
  community database; tool-gated on the prjxray/f4pga install.
* :class:`ApiculaReconstructor` -- Gowin LittleBee via Project Apicula
  (``gowin_unpack``), which decodes a ``.fs`` bitstream straight to Verilog.
* :class:`NoReconstruction` -- vendor-locked silicon (AMD UltraScale+, all
  Intel) whose bitstream is undocumented/encrypted. Bit-level reconstruction is
  *physically* impossible, so these devices degrade to the netlist-level
  equivalence tier (prove RTL == vendor post-implementation netlist).

Callers select the right reconstructor from the device registry via
:func:`reconstructor_for`, and check ``.available`` before attempting bring-up.
:func:`achievable_tier` reports the equivalence tier a device can reach *right
now* (device format capability AND installed tools), which is what the
readiness gate uses so it never overclaims.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from ..devices import EQ_BITSTREAM, EQ_NETLIST, EQ_NONE, DeviceInfo
from ..devices import get as _dev_get


class FabricReconstructor(ABC):
    """Turns a bitstream (``.asc``) into a simulatable fabric netlist."""

    kind: str = "none"
    available: bool = False

    @abstractmethod
    def reconstruct(self, asc_path: str | Path, out_v: str | Path, *,
                    pcf_path: str | Path | None = None,
                    module: str = "recon") -> str:
        """Write a Verilog netlist of the configured fabric to ``out_v``."""

    def why_unavailable(self, target: str) -> str:
        return f"bitstream reconstruction is not available for {target!r}"


class IceStormReconstructor(FabricReconstructor):
    """iCE40 reconstruction via Project IceStorm's ``icebox_vlog``."""

    kind = "icestorm"
    available = True

    def __init__(self, icebox_vlog: str = "icebox_vlog") -> None:
        self.icebox_vlog = icebox_vlog

    def reconstruct(self, asc_path, out_v, *, pcf_path=None, module="recon") -> str:
        from . import netlist as nl

        return nl.reconstruct(asc_path, out_v, pcf_path=pcf_path, module=module,
                              icebox_vlog=self.icebox_vlog)


class XRayReconstructor(FabricReconstructor):
    """AMD/Xilinx 7-series reconstruction via Project X-Ray.

    Pipeline: ``bit2fasm`` decodes the ``.bit`` against the community bitstream
    database into FASM (the documented list of every configured feature), then
    ``fasm2bels`` rebuilds a Verilog netlist of BEL primitives. Both come from
    the prjxray / f4pga projects; this class is gated on them being installed
    (plus ``PRJXRAY_DB_DIR`` pointing at the database checkout).
    """

    kind = "xray"

    def __init__(self, part: str = "") -> None:
        self.part = part
        self.db_dir = os.environ.get("PRJXRAY_DB_DIR", "")

    @property
    def available(self) -> bool:  # type: ignore[override]
        return bool(
            shutil.which("bit2fasm") and shutil.which("fasm2bels")
            and self.db_dir and Path(self.db_dir).exists()
        )

    def reconstruct(self, asc_path, out_v, *, pcf_path=None, module="recon") -> str:
        if not self.available:
            raise RuntimeError(self.why_unavailable(self.part))
        bit = Path(asc_path)
        out_v = Path(out_v)
        fasm = out_v.with_suffix(".fasm")
        subprocess.run(
            ["bit2fasm", "--db-root", self.db_dir, "--part", self.part,
             str(bit), "--fasm-file", str(fasm)],
            check=True, capture_output=True, text=True, timeout=600,
        )
        subprocess.run(
            ["fasm2bels", "--db_root", self.db_dir, "--part", self.part,
             "--fasm_in", str(fasm), "--verilog_out", str(out_v)],
            check=True, capture_output=True, text=True, timeout=600,
        )
        text = out_v.read_text()
        if module != "top":
            text = re.sub(r"\bmodule\s+top\b", f"module {module}", text, count=1)
            out_v.write_text(text)
        return text

    def why_unavailable(self, target: str) -> str:
        return (
            f"{target!r} (xc7) has an *open, documented* bitstream via Project "
            f"X-Ray, but the tools are not installed. Install prjxray "
            f"(bit2fasm) + f4pga (fasm2bels) and set PRJXRAY_DB_DIR to the "
            f"database checkout to enable bit-level bring-up; until then the "
            f"strongest guarantee is netlist-level equivalence."
        )


class ApiculaReconstructor(FabricReconstructor):
    """Gowin LittleBee reconstruction via Project Apicula (``gowin_unpack``).

    ``gowin_unpack -d <device> -o out.v bitstream.fs`` decodes the flashed
    ``.fs`` straight back to a Verilog netlist -- the same one-step decode that
    IceStorm gives us on iCE40.
    """

    kind = "apicula"

    def __init__(self, device: str = "GW1N-9C") -> None:
        self.device = device

    @property
    def available(self) -> bool:  # type: ignore[override]
        return shutil.which("gowin_unpack") is not None

    def reconstruct(self, asc_path, out_v, *, pcf_path=None, module="recon") -> str:
        if not self.available:
            raise RuntimeError(self.why_unavailable(self.device))
        out_v = Path(out_v)
        subprocess.run(
            ["gowin_unpack", "-d", self.device, "-o", str(out_v), str(asc_path)],
            check=True, capture_output=True, text=True, timeout=600,
        )
        text = out_v.read_text()
        if module != "top":
            text = re.sub(r"\bmodule\s+top\b", f"module {module}", text, count=1)
            out_v.write_text(text)
        return text

    def why_unavailable(self, target: str) -> str:
        return (
            f"{target!r} (Gowin) has an open bitstream via Project Apicula, "
            f"but gowin_unpack is not installed (pip install apycula)."
        )


class NoReconstruction(FabricReconstructor):
    """Placeholder for devices whose bitstream cannot be decoded."""

    kind = "none"
    available = False

    def reconstruct(self, asc_path, out_v, *, pcf_path=None, module="recon") -> str:
        raise RuntimeError(
            "this device has no open bitstream database; bit-level "
            "reconstruction is impossible (use the netlist equivalence tier)"
        )

    def why_unavailable(self, target: str) -> str:
        dev = _dev_get(target)
        vendor = dev.vendor if dev else "this vendor"
        return (
            f"{target!r} ({vendor}) has a proprietary/encrypted bitstream, so "
            f"bit-level bring-up is impossible. The strongest available "
            f"guarantee is netlist-level equivalence against the vendor "
            f"post-implementation netlist."
        )


def reconstructor_for(target: str | None,
                      icebox_vlog: str = "icebox_vlog") -> FabricReconstructor:
    """Return the reconstructor the device registry maps ``target`` to."""
    dev = _dev_get(target)
    if dev is None:
        return NoReconstruction()
    if dev.reconstructor == "icestorm":
        return IceStormReconstructor(icebox_vlog=icebox_vlog)
    if dev.reconstructor == "xray":
        return XRayReconstructor(part=dev.part)
    if dev.reconstructor == "apicula":
        return ApiculaReconstructor(device=dev.part or dev.chipdb_tag)
    return NoReconstruction()


def achievable_tier(dev: DeviceInfo | None) -> str:
    """The equivalence tier reachable *right now*: format capability AND tools.

    ``DeviceInfo.equivalence_tier`` states what the device's bitstream format
    permits; this narrows it to what the installed toolchain can actually do,
    so the readiness gate never overclaims (e.g. xc7 is bit-reconstructable via
    Project X-Ray, but degrades to the netlist tier until prjxray is installed).
    """
    if dev is None:
        return EQ_NONE
    tier = dev.equivalence_tier
    if tier == EQ_BITSTREAM and not reconstructor_for(dev.target).available:
        return EQ_NETLIST if dev.sim_lib else EQ_NONE
    return tier
