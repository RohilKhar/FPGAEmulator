"""A 2-D quasi-static field solver for PCB transmission lines.

Rather than hand-entering a trace impedance and coupling ratios, derive them
from the *stackup geometry* -- trace width, dielectric height, copper
thickness, permittivity, and edge spacing:

* single-line **characteristic impedance** and **propagation velocity** for
  microstrip (Hammerstad-Jensen) and symmetric stripline (Cohn / IPC-2141);
* coupled-line **even/odd mode impedances** (Garg-Bahl static capacitances) and
  from them the inductive/capacitive coupling ratios ``k_l`` and ``k_c`` that
  the crosstalk model consumes.

These are the accepted closed-form quasi-static approximations used across SI
practice. They are accurate to a few percent for typical geometries and are the
right input to the higher-level models -- but a full 3-D field solver on the
real layout is still the gold standard for tight designs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .signal_integrity import Net
from .crosstalk import CrosstalkPair

_C0_MM_NS = 299.792458        # speed of light, mm/ns
_C0 = 2.99792458e8            # m/s
_EPS0 = 8.8541878128e-12      # F/m


@dataclass
class StackupGeometry:
    """PCB trace geometry (all lengths in millimetres)."""

    kind: str = "microstrip"       # "microstrip" | "stripline"
    trace_w_mm: float = 0.2
    height_mm: float = 0.1         # dielectric height to reference plane
    thickness_mm: float = 0.035    # copper thickness (~1 oz)
    er: float = 4.3                # FR-4-ish
    spacing_mm: float = 0.2        # edge-to-edge gap to the neighbour (coupling)
    plane_sep_mm: float = 0.2      # stripline: separation between the two planes


@dataclass
class LineSolution:
    z0_ohm: float
    er_eff: float
    velocity_mm_ns: float
    # Coupled-mode results (populated when a neighbour spacing is given).
    z0_even: float = 0.0
    z0_odd: float = 0.0
    k_backward: float = 0.0        # NEXT coefficient (Z0e-Z0o)/(Z0e+Z0o)
    k_l: float = 0.0               # inductive coupling ratio Lm/L0
    k_c: float = 0.0               # capacitive coupling ratio Cm/C0

    def summary(self) -> str:
        lines = [
            "field solver",
            f"  Z0        : {self.z0_ohm:.1f} ohm",
            f"  er_eff    : {self.er_eff:.3f}",
            f"  velocity  : {self.velocity_mm_ns:.1f} mm/ns "
            f"({self.velocity_mm_ns/_C0_MM_NS:.2f} c)",
        ]
        if self.z0_even:
            lines += [
                f"  Z0 even/odd: {self.z0_even:.1f} / {self.z0_odd:.1f} ohm",
                f"  coupling  : k_backward={self.k_backward:.3f} "
                f"k_l={self.k_l:.3f} k_c={self.k_c:.3f}",
            ]
        return "\n".join(lines)


# ---------------------- single-line microstrip ---------------------- #
def _microstrip_ereff_z0(u: float, er: float) -> tuple[float, float]:
    """Hammerstad-Jensen effective permittivity and air impedance for w/h=u."""
    a = (1 + (1.0 / 49) * math.log((u ** 4 + (u / 52) ** 2) / (u ** 4 + 0.432))
         + (1.0 / 18.7) * math.log(1 + (u / 18.1) ** 3))
    b = 0.564 * ((er - 0.9) / (er + 3)) ** 0.053
    er_eff = (er + 1) / 2 + (er - 1) / 2 * (1 + 10.0 / u) ** (-a * b)
    fu = 6 + (2 * math.pi - 6) * math.exp(-((30.666 / u) ** 0.7528))
    z0_air = 60.0 * math.log(fu / u + math.sqrt(1 + (2.0 / u) ** 2))
    return er_eff, z0_air


def microstrip_line(g: StackupGeometry) -> LineSolution:
    u = g.trace_w_mm / g.height_mm
    er_eff, z0_air = _microstrip_ereff_z0(u, g.er)
    z0 = z0_air / math.sqrt(er_eff)
    v = _C0_MM_NS / math.sqrt(er_eff)
    return LineSolution(z0_ohm=z0, er_eff=er_eff, velocity_mm_ns=v)


def stripline_line(g: StackupGeometry) -> LineSolution:
    b = g.plane_sep_mm
    w, t = g.trace_w_mm, g.thickness_mm
    z0 = (60.0 / math.sqrt(g.er)) * math.log(4 * b / (0.67 * math.pi * (0.8 * w + t)))
    v = _C0_MM_NS / math.sqrt(g.er)
    return LineSolution(z0_ohm=z0, er_eff=g.er, velocity_mm_ns=v)


# ---------------------- coupled microstrip (Garg) ------------------- #
def _kpk_over_kk(k: float) -> float:
    """Ratio K(k')/K(k) via the standard conformal-mapping approximation."""
    k = min(max(k, 1e-6), 1 - 1e-9)
    kp = math.sqrt(1 - k * k)
    if k <= 1 / math.sqrt(2):
        # K(k)/K(k') = pi / ln(2 (1+sqrt(k'))/(1-sqrt(k')))  -> invert
        kk = math.pi / math.log(2 * (1 + math.sqrt(kp)) / (1 - math.sqrt(kp)))
    else:
        kk = math.log(2 * (1 + math.sqrt(k)) / (1 - math.sqrt(k))) / math.pi
    return 1.0 / kk


def _coupled_caps(u: float, g_ratio: float, er: float) -> tuple[float, float]:
    """Garg-Bahl even/odd per-length capacitances (F/m) for w/h=u, s/h=g_ratio."""
    er_eff, z0_air = _microstrip_ereff_z0(u, er)
    z0 = z0_air / math.sqrt(er_eff)
    cp = _EPS0 * er * u
    c_single = math.sqrt(er_eff) / (_C0 * z0)         # total single-line C (F/m)
    cf = 0.5 * (c_single - cp)
    A = math.exp(-0.1 * math.exp(2.33 - 2.53 * u))
    cf_p = cf / (1 + A * (1.0 / g_ratio) * math.tanh(8 * g_ratio))
    # odd-mode gap capacitances
    k = g_ratio / (g_ratio + 2 * u)
    cga = _EPS0 * _kpk_over_kk(k)
    cgd = ((_EPS0 * er / math.pi) * math.log(1.0 / math.tanh(math.pi * g_ratio / 4))
           + 0.65 * cf * (0.02 * (1.0 / g_ratio) * math.sqrt(er) + 1 - er ** -2))
    c_even = cp + cf + cf_p
    c_odd = cp + cf + cga + cgd
    return c_even, c_odd


def coupled_microstrip(g: StackupGeometry) -> LineSolution:
    """Even/odd impedances and coupling ratios for an edge-coupled pair."""
    sol = microstrip_line(g)
    u = g.trace_w_mm / g.height_mm
    gr = g.spacing_mm / g.height_mm

    ce, co = _coupled_caps(u, gr, g.er)
    ce_a, co_a = _coupled_caps(u, gr, 1.0)             # air (for L and Z0)

    z0e = 1.0 / (_C0 * math.sqrt(ce * ce_a))
    z0o = 1.0 / (_C0 * math.sqrt(co * co_a))
    # Inductance is dielectric-independent -> use air impedances.
    z0e_air = 1.0 / (_C0 * ce_a)
    z0o_air = 1.0 / (_C0 * co_a)

    sol.z0_even, sol.z0_odd = z0e, z0o
    sol.k_backward = (z0e - z0o) / (z0e + z0o)
    sol.k_l = (z0e_air - z0o_air) / (z0e_air + z0o_air)
    sol.k_c = (co - ce) / (co + ce)
    return sol


# --------------------------- entry points --------------------------- #
def solve(g: StackupGeometry) -> LineSolution:
    """Solve a line (with coupling if the geometry is microstrip)."""
    if g.kind == "stripline":
        return stripline_line(g)
    return coupled_microstrip(g)


def net_from_geometry(
    g: StackupGeometry,
    name: str = "io",
    trace_len_mm: float = 50.0,
    load_pf: float = 10.0,
    vdd: float = 3.3,
    drive_impedance_ohm: float = 40.0,
    driver_rise_ns: float = 1.0,
    n_simultaneous: int = 1,
) -> Net:
    """Build an SI :class:`Net` whose Z0/velocity come from the stackup."""
    sol = solve(g)
    return Net(
        name=name, vdd=vdd, drive_impedance_ohm=drive_impedance_ohm,
        driver_rise_ns=driver_rise_ns, load_pf=load_pf,
        trace_z0_ohm=sol.z0_ohm, trace_len_mm=trace_len_mm,
        velocity_mm_ns=sol.velocity_mm_ns, n_simultaneous=n_simultaneous,
    )


def crosstalk_from_geometry(
    g: StackupGeometry,
    victim: str = "victim",
    coupling_len_mm: float = 25.0,
    aggressor_rise_ns: float = 1.0,
    vdd: float = 3.3,
    n_aggressors: int = 1,
    noise_margin_v: float | None = None,
) -> CrosstalkPair:
    """Build a :class:`CrosstalkPair` with coupling ratios from the stackup."""
    sol = coupled_microstrip(g)
    return CrosstalkPair(
        victim=victim, vdd=vdd, aggressor_rise_ns=aggressor_rise_ns,
        coupling_len_mm=coupling_len_mm, velocity_mm_ns=sol.velocity_mm_ns,
        k_c=abs(sol.k_c), k_l=abs(sol.k_l), n_aggressors=n_aggressors,
        noise_margin_v=noise_margin_v,
    )
