"""Tests for the device/capability registry and backend routing."""

from fpgaforge import devices as d
from fpgaforge.backends.ice40 import _DEVICES as ICE_DEVICES, _DSP_TARGETS
from fpgaforge.backends.ecp5 import _DEVICES as ECP5_DEVICES
from fpgaforge.emulator.netlist import DEVICE_INFO
from fpgaforge.optimizer import backend_for_target


def test_every_device_has_a_backend_and_vendor():
    for dev in d.all_devices():
        assert dev.backend in {"ice40", "ecp5", "vivado", "quartus", "gowin"}
        assert dev.vendor in {"lattice", "amd", "intel", "gowin"}
        assert dev.luts > 0 and dev.io > 0


def test_ice40_is_bit_reconstructable():
    up5k = d.get("ice40_up5k")
    assert up5k.reconstructable is True
    assert up5k.reconstructor == "icestorm"
    assert up5k.equivalence_tier == d.EQ_BITSTREAM
    assert d.TIER_PROVE_BITSTREAM in up5k.tiers
    assert d.TIER_EMULATE in up5k.tiers


def test_locked_vendor_devices_are_netlist_tier_only():
    # UltraScale+ and Intel bitstreams are undocumented/encrypted: no bit tier.
    for target in ("xczu3eg", "cyclonev_5csema5", "max10_10m50"):
        dev = d.get(target)
        assert dev.reconstructable is False
        assert dev.equivalence_tier == d.EQ_NETLIST
        assert d.TIER_PROVE_BITSTREAM not in dev.tiers
        assert d.TIER_EMULATE not in dev.tiers
        assert d.TIER_PROVE_NETLIST in dev.tiers


def test_xc7_and_gowin_formats_are_bit_reconstructable():
    # 7-series is documented by Project X-Ray; Gowin by Project Apicula. The
    # *format* permits the bitstream tier; reaching it depends on tools (see
    # achievable_tier).
    assert d.get("xc7a35t").reconstructor == "xray"
    assert d.get("xc7a35t").equivalence_tier == d.EQ_BITSTREAM
    assert d.get("gowin_gw1n9").reconstructor == "apicula"
    assert d.get("gowin_gw1n9").equivalence_tier == d.EQ_BITSTREAM


def test_achievable_tier_is_tool_aware(monkeypatch):
    from fpgaforge.emulator import reconstruct as rc

    # xc7 degrades to the netlist tier until prjxray + its DB are installed.
    monkeypatch.delenv("PRJXRAY_DB_DIR", raising=False)
    assert rc.achievable_tier(d.get("xc7a35t")) == d.EQ_NETLIST
    # iCE40 is always bit-level (IceStorm is the primary supported flow).
    assert rc.achievable_tier(d.get("ice40_up5k")) == d.EQ_BITSTREAM
    assert rc.achievable_tier(None) == d.EQ_NONE


def test_reconstructor_selection():
    from fpgaforge.emulator.reconstruct import (
        ApiculaReconstructor, IceStormReconstructor, NoReconstruction,
        XRayReconstructor, reconstructor_for,
    )

    assert isinstance(reconstructor_for("ice40_up5k"), IceStormReconstructor)
    assert isinstance(reconstructor_for("xc7a35t"), XRayReconstructor)
    assert isinstance(reconstructor_for("gowin_gw1n9"), ApiculaReconstructor)
    assert isinstance(reconstructor_for("xczu3eg"), NoReconstruction)
    # The X-Ray path tells the user exactly what to install.
    why = reconstructor_for("xc7a35t").why_unavailable("xc7a35t")
    assert "X-Ray" in why and "PRJXRAY_DB_DIR" in why


def test_ecp5_is_netlist_tier_here():
    # We synth/PnR ECP5, but do not implement bitstream reconstruction, so its
    # strongest tier in this codebase is netlist-level.
    ecp5 = d.get("ecp5_85k")
    assert ecp5.reconstructable is False
    assert ecp5.equivalence_tier == d.EQ_NETLIST


def test_derived_tables_match_registry():
    assert ICE_DEVICES == {
        dev.target: (dev.pnr_flag, dev.package) for dev in d.by_backend("ice40")
    }
    assert ECP5_DEVICES == {
        dev.target: (dev.pnr_flag, dev.package) for dev in d.by_backend("ecp5")
    }
    assert _DSP_TARGETS == frozenset(
        dev.target for dev in d.by_backend("ice40") if dev.has_dsp
    )
    # Only IceStorm-reconstructable devices are in the emulator's DEVICE_INFO.
    assert set(DEVICE_INFO) == {
        dev.target for dev in d.by_backend("ice40") if dev.reconstructor == "icestorm"
    }


def test_backend_routing_by_registry():
    # Falls back to mock when vendor tools are absent (CI has no Vivado/Quartus).
    assert backend_for_target("xc7a35t").name in {"vivado", "mock"}
    assert backend_for_target("cyclonev_5csema5").name in {"quartus", "mock"}
    # iCE40 tools are present in the test env.
    assert backend_for_target("ice40_up5k").name in {"ice40", "mock"}


def test_unknown_target_is_none():
    assert d.get("nonsuch_fpga") is None
    assert d.get(None) is None
