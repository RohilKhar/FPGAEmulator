"""Device / capability registry -- the single source of truth for targets.

Every part the platform knows about lives here as a :class:`DeviceInfo`. This
replaces the device tables that used to be scattered across the backends, the
emulator, and the readiness gate, and it is what lets the rest of the stack ask
capability questions ("can I reconstruct this device's bitstream?", "what is the
strongest equivalence tier I can establish here?") instead of hard-coding
vendor/family string checks.

The key idea is **tiered support**. Not every FPGA can be brought up to the same
level of trust:

* Open-bitstream families (Lattice iCE40 via Project IceStorm) can be
  *reconstructed* from the flashed bits, so we can prove the bitstream itself is
  equivalent to the RTL -- the strongest guarantee.
* Vendor-locked families (AMD/Xilinx UltraScale+, all Intel/Altera) have
  undocumented/encrypted bitstreams, so bit-level bring-up is *physically*
  impossible. There we degrade to the vendor post-implementation **netlist**
  (which Vivado/Quartus can emit) plus vendor timing/power sign-off.

So a device advertises the set of `tiers` it can reach, and downstream features
consult that instead of assuming iCE40.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---- capability tiers, weakest -> strongest ---------------------------- #
TIER_IMPLEMENT = "implement"            # RTL -> bitstream (a backend exists)
TIER_TIMING = "timing"                  # real static timing sign-off
TIER_POWER = "power"                    # power/thermal estimate from the tool
TIER_VERIFY = "verify"                  # cycle-accurate stimulus comparison
TIER_EMULATE = "emulate"                # decode the flashed bitstream -> fabric
TIER_PROVE_NETLIST = "prove_netlist"    # formal RTL == post-impl netlist
TIER_PROVE_BITSTREAM = "prove_bitstream"  # formal RTL == flashed bitstream

# Equivalence tiers, in descending strength, for the trust ladder.
EQ_BITSTREAM = "bitstream"   # bit-level: proven against the actual .bin
EQ_NETLIST = "netlist"       # gate-level: proven against post-impl netlist
EQ_NONE = "none"             # no formal equivalence path on this silicon


@dataclass(frozen=True)
class DeviceInfo:
    """Everything the platform needs to know about one target part."""

    target: str            # canonical id, e.g. "ice40_up5k", "xc7a35t"
    vendor: str            # "lattice" | "amd" | "intel" | "gowin"
    family: str            # "ice40" | "ecp5" | "xc7" | "cyclone10lp" ...
    backend: str           # which Backend implements it: ice40|ecp5|vivado|quartus
    package: str
    luts: int              # usable 4/6-LUTs (or logic elements)
    ffs: int
    bram_kbits: int
    dsp: int
    io: int
    # Reconstruction: name of the FabricReconstructor that can decode this
    # device's bitstream back into a netlist, or "" if none exists/implemented.
    # This is what makes bit-level bring-up (emulate/prove-bitstream) possible.
    reconstructor: str = ""
    # Backend-specific handles.
    pnr_flag: str = ""     # nextpnr device flag, e.g. "--up5k"
    chipdb_tag: str = ""   # IceStorm chipdb tag, e.g. "5k"
    part: str = ""         # vendor part number, e.g. "xc7a35tcpg236-1"
    has_dsp: bool = False   # hardened DSP the synth flow can map multipliers to
    # yosys behavioral cell library (space-separated `+/...` paths) used to
    # elaborate the vendor primitives in a post-implementation netlist for the
    # netlist-level equivalence proof. Empty when we don't (yet) have one.
    sim_lib: str = ""

    @property
    def reconstructable(self) -> bool:
        """True when the flashed bitstream can be decoded back to a fabric."""
        return bool(self.reconstructor)

    @property
    def is_open(self) -> bool:
        """True for the fully open-source (yosys+nextpnr) flow."""
        return self.backend in ("ice40", "ecp5")

    @property
    def equivalence_tier(self) -> str:
        """Strongest RTL-equivalence guarantee this device's *format* permits.

        This is the device capability; whether the decoding tools are installed
        is a separate question -- see ``emulator.reconstruct.achievable_tier``
        for the tier reachable right now.
        """
        if self.reconstructable:
            return EQ_BITSTREAM
        if self.backend in ("vivado", "quartus", "ecp5", "gowin"):
            # Vendor/open backends can emit a post-implementation netlist we can
            # formally compare against, even without bitstream reconstruction.
            return EQ_NETLIST
        return EQ_NONE

    @property
    def tiers(self) -> frozenset[str]:
        t = {TIER_IMPLEMENT}
        if self.backend != "mock":
            t |= {TIER_TIMING, TIER_POWER}
        if self.equivalence_tier == EQ_BITSTREAM:
            t |= {TIER_EMULATE, TIER_VERIFY, TIER_PROVE_NETLIST, TIER_PROVE_BITSTREAM}
        elif self.equivalence_tier == EQ_NETLIST:
            t |= {TIER_PROVE_NETLIST}
        return frozenset(t)

    def supports(self, tier: str) -> bool:
        return tier in self.tiers


# --------------------------------------------------------------------------- #
# The registry. Capacities are datasheet figures (approximate for the vendor
# parts, exact for the iCE40/ECP5 families we implement natively).
# --------------------------------------------------------------------------- #
_DEVICES: dict[str, DeviceInfo] = {}


def _reg(dev: DeviceInfo) -> None:
    _DEVICES[dev.target] = dev


# ---- Lattice iCE40 (Project IceStorm: fully open, bit-reconstructable) --- #
_reg(DeviceInfo("ice40_up5k", "lattice", "ice40", "ice40", "sg48",
                luts=5280, ffs=5280, bram_kbits=120, dsp=8, io=39,
                reconstructor="icestorm", pnr_flag="--up5k", chipdb_tag="5k",
                has_dsp=True))
_reg(DeviceInfo("ice40_hx8k", "lattice", "ice40", "ice40", "ct256",
                luts=7680, ffs=7680, bram_kbits=128, dsp=0, io=206,
                reconstructor="icestorm", pnr_flag="--hx8k", chipdb_tag="8k"))
_reg(DeviceInfo("ice40_hx1k", "lattice", "ice40", "ice40", "tq144",
                luts=1280, ffs=1280, bram_kbits=64, dsp=0, io=96,
                reconstructor="icestorm", pnr_flag="--hx1k", chipdb_tag="1k"))
_reg(DeviceInfo("ice40_lp8k", "lattice", "ice40", "ice40", "cm81",
                luts=7680, ffs=7680, bram_kbits=128, dsp=0, io=63,
                reconstructor="icestorm", pnr_flag="--lp8k", chipdb_tag="8k"))

# ---- Lattice ECP5 (Project Trellis: open synth/PnR; native reconstruction
#      not implemented here, so it uses the netlist equivalence tier) ------- #
_ECP5_SIM = "+/ecp5/cells_sim.v"
_reg(DeviceInfo("ecp5_12k", "lattice", "ecp5", "ecp5", "CABGA256",
                luts=12288, ffs=12288, bram_kbits=1008,
                dsp=28, io=197, pnr_flag="--12k", has_dsp=True, sim_lib=_ECP5_SIM))
_reg(DeviceInfo("ecp5_25k", "lattice", "ecp5", "ecp5", "CABGA256",
                luts=24288, ffs=24288, bram_kbits=1008, dsp=28, io=197,
                pnr_flag="--25k", has_dsp=True, sim_lib=_ECP5_SIM))
_reg(DeviceInfo("ecp5_45k", "lattice", "ecp5", "ecp5", "CABGA381",
                luts=44000, ffs=44000, bram_kbits=1944, dsp=72, io=285,
                pnr_flag="--45k", has_dsp=True, sim_lib=_ECP5_SIM))
_reg(DeviceInfo("ecp5_85k", "lattice", "ecp5", "ecp5", "CABGA381",
                luts=83640, ffs=83640, bram_kbits=3744, dsp=156, io=285,
                pnr_flag="--85k", has_dsp=True, sim_lib=_ECP5_SIM))

# ---- AMD/Xilinx 7-series (Vivado) ----------------------------------------- #
# Vivado `write_verilog -mode funcsim` emits UNISIM primitives (LUT*, FD*,
# CARRY4, RAMB*, DSP48E1) that yosys' bundled xilinx cells_sim elaborates.
# The 7-series bitstream format is *documented* by Project X-Ray, so these
# parts are bit-reconstructable (reconstructor="xray") when prjxray/f4pga are
# installed; they degrade to the netlist tier otherwise.
_XILINX_SIM = "+/xilinx/cells_sim.v"
_reg(DeviceInfo("xc7a35t", "amd", "xc7", "vivado", "cpg236",
                luts=20800, ffs=41600, bram_kbits=1800, dsp=90, io=106,
                part="xc7a35tcpg236-1", has_dsp=True, sim_lib=_XILINX_SIM,
                reconstructor="xray"))
_reg(DeviceInfo("xc7a100t", "amd", "xc7", "vivado", "csg324",
                luts=63400, ffs=126800, bram_kbits=4860, dsp=240, io=210,
                part="xc7a100tcsg324-1", has_dsp=True, sim_lib=_XILINX_SIM,
                reconstructor="xray"))
_reg(DeviceInfo("xc7z020", "amd", "zynq7", "vivado", "clg484",
                luts=53200, ffs=106400, bram_kbits=4480, dsp=220, io=200,
                part="xc7z020clg484-1", has_dsp=True, sim_lib=_XILINX_SIM,
                reconstructor="xray"))

# ---- Gowin LittleBee (Project Apicula: open bitstream, gowin_unpack) ------ #
_GOWIN_SIM = "+/gowin/cells_sim.v"
_reg(DeviceInfo("gowin_gw1n9", "gowin", "gw1n", "gowin", "QN88",
                luts=8640, ffs=6480, bram_kbits=468, dsp=20, io=71,
                part="GW1N-9C", has_dsp=True, sim_lib=_GOWIN_SIM,
                reconstructor="apicula"))
_reg(DeviceInfo("gowin_gw1n1", "gowin", "gw1n", "gowin", "QN48",
                luts=1152, ffs=864, bram_kbits=72, dsp=0, io=41,
                part="GW1N-1", sim_lib=_GOWIN_SIM,
                reconstructor="apicula"))

# ---- AMD/Xilinx UltraScale+ (Vivado; encrypted bitstream -> netlist tier) - #
_reg(DeviceInfo("xczu3eg", "amd", "zynqmp", "vivado", "sfvc784",
                luts=71000, ffs=141000, bram_kbits=7200, dsp=360, io=252,
                part="xczu3eg-sfvc784-1-e", has_dsp=True, sim_lib=_XILINX_SIM))

# ---- Intel/Altera (Quartus; encrypted bitstream -> netlist tier) --------- #
_INTEL_ALM_SIM = (
    "+/intel_alm/common/alm_sim.v +/intel_alm/common/dff_sim.v "
    "+/intel_alm/common/mem_sim.v +/intel_alm/common/dsp_sim.v "
    "+/intel_alm/common/misc_sim.v +/intel_alm/cyclonev/cells_sim.v"
)
_reg(DeviceInfo("cyclone10lp_10cl025", "intel", "cyclone10lp", "quartus", "u256",
                luts=24624, ffs=24624, bram_kbits=594, dsp=66, io=150,
                part="10CL025YU256C8G", has_dsp=True,
                sim_lib="+/intel/cyclone10lp/cells_sim.v"))
_reg(DeviceInfo("max10_10m50", "intel", "max10", "quartus", "f484",
                luts=49760, ffs=49760, bram_kbits=1638, dsp=144, io=360,
                part="10M50DAF484C7G", has_dsp=True,
                sim_lib="+/intel/max10/cells_sim.v"))
_reg(DeviceInfo("cyclonev_5csema5", "intel", "cyclonev", "quartus", "u23",
                luts=85000, ffs=85000, bram_kbits=3970, dsp=87, io=288,
                part="5CSEMA5F31C6", has_dsp=True, sim_lib=_INTEL_ALM_SIM))


# --------------------------------------------------------------------------- #
def get(target: str | None) -> DeviceInfo | None:
    """Look up a device by canonical target id."""
    if not target:
        return None
    return _DEVICES.get(target)

def require(target: str) -> DeviceInfo:
    dev = get(target)
    if dev is None:
        raise KeyError(
            f"unknown device target {target!r}; known: {', '.join(sorted(_DEVICES))}"
        )
    return dev

def all_devices() -> list[DeviceInfo]:
    return list(_DEVICES.values())

def targets() -> list[str]:
    return sorted(_DEVICES)

def by_backend(backend: str) -> list[DeviceInfo]:
    return [d for d in _DEVICES.values() if d.backend == backend]

def by_vendor(vendor: str) -> list[DeviceInfo]:
    return [d for d in _DEVICES.values() if d.vendor == vendor]
