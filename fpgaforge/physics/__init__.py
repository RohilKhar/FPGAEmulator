"""Physics-based sign-off models for effects that live beyond the netlist.

These model, to first/second order, the physical phenomena that a purely
logical/timing flow cannot see:

* :mod:`~fpgaforge.physics.pvt` -- process/voltage/temperature derating of the
  STA Fmax into a *guaranteed* worst-case number.
* :mod:`~fpgaforge.physics.signal_integrity` -- I/O rise time, transmission-line
  reflections/overshoot, simultaneous-switching noise, and settling. Optionally
  runs a real ``ngspice`` circuit simulation when the tool is present.
* :mod:`~fpgaforge.physics.interfaces` -- source-synchronous / DDR timing
  budgets (setup/hold vs flight time, skew, and jitter).
* :mod:`~fpgaforge.physics.fieldsolver` -- 2-D quasi-static field solver that
  derives trace Z0/velocity and coupled-line k_l/k_c from PCB stackup geometry.
* :mod:`~fpgaforge.physics.pdn` -- power-distribution-network impedance vs the
  target impedance, including decoupling self-resonance / anti-resonance peaks.

Honesty: these are physically-grounded *models*, not a substitute for lab
measurement or full 3D-EM/SPICE extraction. Their accuracy is bounded by the
board/package parameters you feed them; they shrink risk and quantify margins,
they do not make hardware bring-up unnecessary.
"""

from .pvt import PVTConfig, PVTCorner, PVTResult, ICE40_UP5K_PVT, derate_fmax
from .signal_integrity import (
    Net,
    PackageModel,
    SIResult,
    analyze_net,
    spice_deck,
    ICE40_SG48,
)
from .interfaces import InterfaceBudget, InterfaceResult, analyze_interface
from .crosstalk import CrosstalkPair, CrosstalkResult, analyze_crosstalk
from .ibis import (
    IbisModel,
    parse_ibis,
    load_ibis,
    net_from_ibis,
    package_from_ibis,
)
from .fieldsolver import (
    StackupGeometry,
    LineSolution,
    microstrip_line,
    stripline_line,
    coupled_microstrip,
    solve,
    net_from_geometry,
    crosstalk_from_geometry,
)
from .pdn import DecouplingCap, PDNConfig, PDNResult, analyze_pdn
from .power import PowerConfig, PowerResult, ICE40_UP5K_POWER, estimate_power
from .signoff import PhysicalReport, physical_signoff

__all__ = [
    "PVTConfig",
    "PVTCorner",
    "PVTResult",
    "ICE40_UP5K_PVT",
    "derate_fmax",
    "Net",
    "PackageModel",
    "SIResult",
    "analyze_net",
    "spice_deck",
    "ICE40_SG48",
    "InterfaceBudget",
    "InterfaceResult",
    "analyze_interface",
    "CrosstalkPair",
    "CrosstalkResult",
    "analyze_crosstalk",
    "IbisModel",
    "parse_ibis",
    "load_ibis",
    "net_from_ibis",
    "package_from_ibis",
    "StackupGeometry",
    "LineSolution",
    "microstrip_line",
    "stripline_line",
    "coupled_microstrip",
    "solve",
    "net_from_geometry",
    "crosstalk_from_geometry",
    "DecouplingCap",
    "PDNConfig",
    "PDNResult",
    "analyze_pdn",
    "PowerConfig",
    "PowerResult",
    "ICE40_UP5K_POWER",
    "estimate_power",
    "PhysicalReport",
    "physical_signoff",
]
