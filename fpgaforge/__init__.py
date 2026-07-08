"""fpgaforge: an AI-native FPGA implementation engine.

Public API:
    optimize(...)            -> run the model-guided optimization loop
    OptimizationResult        -> structured result of optimize()
    Design, FlowOptions       -> describe what to build and how
    RunMetrics, RunResult     -> outcome of a single flow run
    Ice40Backend, MockBackend -> real / offline backends
"""

from .backends.base import (
    Backend,
    Design,
    FlowOptions,
    RunMetrics,
    RunResult,
)
from .backends.ice40 import Ice40Backend
from .backends.ecp5 import Ecp5Backend
from .backends.vivado import VivadoBackend
from .backends.quartus import QuartusBackend
from .backends.gowin import GowinBackend
from .pins import PinConstraints, PinReport, check_pins, load_pins
from .selftest import SelfTestReport, generate_selftest
from .devices import DeviceInfo, get as get_device, all_devices, targets as device_targets
from .backends.mock import MockBackend
from .emulator import (
    BoardConfig,
    BoardResult,
    Emulator,
    EmulationResult,
    FabricConfig,
    LogicCell,
    ProofResult,
    UartCapture,
    VerificationResult,
    decode_fabric,
    emulate,
    emulate_board,
    prove,
    verify_bitstream,
)
from .emulator import MutationResult, mutation_test
from .emulator import TimingEmulationResult, timing_emulate
from .sdf_timing import (
    CrossArc,
    DomainTiming,
    SDFTimingResult,
    analyze_sdf,
    parse_sdf,
)
from .optimizer import OptimizationResult, optimize, backend_for_target, default_backend
from .physics import (
    CrosstalkPair,
    CrosstalkResult,
    DecouplingCap,
    IbisModel,
    InterfaceBudget,
    LineSolution,
    Net,
    PackageModel,
    PDNConfig,
    PDNResult,
    PhysicalReport,
    PowerConfig,
    PowerResult,
    PVTConfig,
    PVTResult,
    SIResult,
    StackupGeometry,
    analyze_crosstalk,
    analyze_interface,
    analyze_net,
    analyze_pdn,
    estimate_power,
    coupled_microstrip,
    crosstalk_from_geometry,
    derate_fmax,
    load_ibis,
    microstrip_line,
    net_from_geometry,
    net_from_ibis,
    physical_signoff,
    stripline_line,
)
from .readiness import ReadinessReport, assess
from .reward import DesignReward, Issue, score_design, score_report, score_batch
from .model import FmaxPredictor, ModelReport, make_vector
from .bootstrap import BootstrapReport, BootstrapSpec, bootstrap_corpus
from .vcd import ActivityReport, measure_activity, parse_vcd_activity
from .cache import RewardCache, reward_key
from .constraints import Constraints, parse_sdc, load_sdc
from .cdc import CDCReport, Crossing, analyze_cdc
from .timing import CriticalPath, TimingReport, signoff
from .virtual import BringUpConfig, BringUpResult, VirtualFPGA, bringup

__all__ = [
    "Backend",
    "Design",
    "FlowOptions",
    "RunMetrics",
    "RunResult",
    "Ice40Backend",
    "Ecp5Backend",
    "VivadoBackend",
    "QuartusBackend",
    "GowinBackend",
    "PinConstraints",
    "PinReport",
    "check_pins",
    "load_pins",
    "SelfTestReport",
    "generate_selftest",
    "MockBackend",
    "DeviceInfo",
    "get_device",
    "all_devices",
    "device_targets",
    "OptimizationResult",
    "optimize",
    "backend_for_target",
    "default_backend",
    "VirtualFPGA",
    "BringUpConfig",
    "BringUpResult",
    "bringup",
    "ReadinessReport",
    "assess",
    "DesignReward",
    "Issue",
    "score_design",
    "score_report",
    "score_batch",
    "RewardCache",
    "reward_key",
    "Constraints",
    "parse_sdc",
    "load_sdc",
    "CDCReport",
    "Crossing",
    "analyze_cdc",
    "TimingReport",
    "CriticalPath",
    "signoff",
    "Emulator",
    "EmulationResult",
    "VerificationResult",
    "ProofResult",
    "FabricConfig",
    "LogicCell",
    "BoardConfig",
    "BoardResult",
    "UartCapture",
    "decode_fabric",
    "emulate",
    "emulate_board",
    "verify_bitstream",
    "prove",
    "MutationResult",
    "mutation_test",
    "TimingEmulationResult",
    "timing_emulate",
    "SDFTimingResult",
    "DomainTiming",
    "CrossArc",
    "analyze_sdf",
    "parse_sdf",
    "PhysicalReport",
    "physical_signoff",
    "PVTConfig",
    "PVTResult",
    "derate_fmax",
    "Net",
    "PackageModel",
    "SIResult",
    "analyze_net",
    "InterfaceBudget",
    "analyze_interface",
    "CrosstalkPair",
    "CrosstalkResult",
    "analyze_crosstalk",
    "IbisModel",
    "load_ibis",
    "net_from_ibis",
    "StackupGeometry",
    "LineSolution",
    "microstrip_line",
    "stripline_line",
    "coupled_microstrip",
    "net_from_geometry",
    "crosstalk_from_geometry",
    "DecouplingCap",
    "PDNConfig",
    "PDNResult",
    "analyze_pdn",
    "PowerConfig",
    "PowerResult",
    "estimate_power",
    "FmaxPredictor",
    "ModelReport",
    "make_vector",
    "BootstrapReport",
    "BootstrapSpec",
    "bootstrap_corpus",
    "ActivityReport",
    "measure_activity",
    "parse_vcd_activity",
]

__version__ = "0.1.0"
