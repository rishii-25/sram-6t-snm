"""
6T SRAM cell characterization with NGSPICE (PTM 45nm HP, BSIM4).

Metrics
  * Hold SNM  : butterfly curves with word line OFF, largest-square method.
  * Read SNM  : butterfly curves with word line ON and both bit lines held at VDD.
  * Write margin (WM): BL is ramped from VDD to 0 with WL = VDD; WM is the BL
    voltage at which the cell flips (higher = easier to write).

Cell (one half shown; the other half is the mirror image):
    Q  side: PU1 (pmos, Q <- VDD, gate QB), PD1 (nmos, Q -> GND, gate QB), AX1 (BL <-> Q, gate WL)
    QB side: PU2 (pmos, gate Q),            PD2 (nmos, gate Q),            AX2 (BLB <-> QB, gate WL)

Process corners are emulated with a threshold-voltage shift (delvto) and a
mobility multiplier (mulu0) on each device, because PTM ships one nominal deck.
Random mismatch for Monte Carlo uses Pelgrom's law: sigma_Vth = A_VT / sqrt(W*L).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "ptm_45nm_hp.pm"

L = 45e-9  # drawn channel length


# --------------------------------------------------------------------------- #
# Cell description
# --------------------------------------------------------------------------- #
@dataclass
class Sizing:
    """Transistor widths in metres. Cell ratio CR = W_PD / W_AX, pull-up ratio PR = W_PU / W_AX."""
    w_pd: float = 180e-9
    w_ax: float = 90e-9
    w_pu: float = 90e-9

    @property
    def cell_ratio(self) -> float:
        return self.w_pd / self.w_ax

    @property
    def pullup_ratio(self) -> float:
        return self.w_pu / self.w_ax


# (n_dVth, n_mobility, p_dVth, p_mobility). For pmos a positive delvto makes |Vth| smaller (faster).
CORNERS = {
    "TT": (0.00, 1.00, 0.00, 1.00),
    "FF": (-0.05, 1.10, +0.05, 1.10),
    "SS": (+0.05, 0.90, -0.05, 0.90),
    "FS": (-0.05, 1.10, -0.05, 0.90),  # fast nmos, slow pmos
    "SF": (+0.05, 0.90, +0.05, 1.10),  # slow nmos, fast pmos
}

DEVICES = ("PU1", "PD1", "AX1", "PU2", "PD2", "AX2")


@dataclass
class Condition:
    vdd: float = 1.0
    corner: str = "TT"
    temp: float = 27.0
    sizing: Sizing = field(default_factory=Sizing)
    # extra per-device Vth shift (V), used for Monte Carlo mismatch
    mismatch: dict = field(default_factory=lambda: {d: 0.0 for d in DEVICES})


def _dev(cond: Condition, name: str):
    """Return (W, delvto, mulu0) for one device."""
    n_dv, n_mu, p_dv, p_mu = CORNERS[cond.corner]
    s = cond.sizing
    w = {"PU": s.w_pu, "PD": s.w_pd, "AX": s.w_ax}[name[:2]]
    if name.startswith("PU"):
        dv, mu = p_dv, p_mu
        dv += -cond.mismatch[name]  # pmos: +mismatch in |Vth| => more negative Vth
    else:
        dv, mu = n_dv, n_mu
        dv += cond.mismatch[name]
    return w, dv, mu


def _mos(cond, name, d, g, s, b, model):
    w, dv, mu = _dev(cond, name)
    return f"M{name} {d} {g} {s} {b} {model} W={w:.4e} L={L:.4e} delvto={dv:.5f} mulu0={mu:.4f}"


# --------------------------------------------------------------------------- #
# NGSPICE runner
# --------------------------------------------------------------------------- #
def _run(netlist: str, outfile_name: str = "out.txt") -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        cir = Path(td) / "c.cir"
        cir.write_text(netlist.replace("__OUT__", str(Path(td) / outfile_name)))
        # ngspice's OpenMP threads busy-wait when several runs share cores; one thread each is much faster
        env = dict(os.environ, OMP_NUM_THREADS="1")
        r = subprocess.run(["ngspice", "-b", str(cir)], capture_output=True, text=True, timeout=120, env=env)
        out = Path(td) / outfile_name
        if not out.exists():
            raise RuntimeError(f"ngspice failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        return np.loadtxt(out, skiprows=1)


def _header(cond: Condition) -> str:
    return (f"* 6T SRAM\n.include {MODEL}\n.option temp={cond.temp}\n"
            f"Vdd vdd 0 {cond.vdd}\n")


# --------------------------------------------------------------------------- #
# Butterfly curves (two half-cell VTCs with the feedback loop broken)
# --------------------------------------------------------------------------- #
def half_cell_netlist(cond: Condition, side: int, wl_on: bool, step: float = 0.002) -> str:
    """VTC of one half-cell: input swept, output loaded by its access transistor.
    side=1 -> input Q, output QB (devices PU2/PD2/AX2 on BLB)
    side=2 -> input QB, output Q (devices PU1/PD1/AX1 on BL)
    """
    pu, pd, ax = ("PU2", "PD2", "AX2") if side == 1 else ("PU1", "PD1", "AX1")
    vwl = cond.vdd if wl_on else 0.0
    return (_header(cond)
            + f"Vin in 0 0\nVwl wl 0 {vwl}\nVbl bl 0 {cond.vdd}\n"
            + _mos(cond, pu, "out", "in", "vdd", "vdd", "pmos") + "\n"
            + _mos(cond, pd, "out", "in", "0", "0", "nmos") + "\n"
            + _mos(cond, ax, "bl", "wl", "out", "0", "nmos") + "\n"
            + ".control\nset wr_singlescale\nset wr_vecnames\n"
            + f"dc Vin 0 {cond.vdd} {step}\nwrdata __OUT__ v(out)\nquit\n.endc\n.end\n")


def butterfly(cond: Condition, wl_on: bool):
    """Return (x1, y1), (x2, y2) in (Q, QB) coordinates.
    Curve A: QB = f1(Q).  Curve B: Q = f2(QB)  -> plotted as (f2(QB), QB)."""
    a = _run(half_cell_netlist(cond, 1, wl_on))
    b = _run(half_cell_netlist(cond, 2, wl_on))
    return (a[:, 0], a[:, 1]), (b[:, 1], b[:, 0])


def snm_largest_square(curve_a, curve_b, n: int = 2000):
    """Seevinck largest-square SNM.
    Rotate by 45 deg: u = (x - y)/sqrt2, v = (x + y)/sqrt2. For a fixed u the
    vertical gap between the curves in the rotated frame is the diagonal of the
    square that fits between them; side = gap / sqrt2. SNM = smaller of the two lobes.
    Returns (snm, (lobe_left, lobe_right), squares) where squares holds the
    (x0, y0, side) of the largest square in each lobe for plotting."""
    s2 = np.sqrt(2.0)
    xa, ya = curve_a
    xb, yb = curve_b
    ua, va = (xa - ya) / s2, (xa + ya) / s2
    ub, vb = (xb - yb) / s2, (xb + yb) / s2
    ia, ib = np.argsort(ua), np.argsort(ub)
    ua, va, ub, vb = ua[ia], va[ia], ub[ib], vb[ib]
    lo, hi = max(ua.min(), ub.min()), min(ua.max(), ub.max())
    u = np.linspace(lo, hi, n)
    d = np.interp(u, ua, va) - np.interp(u, ub, vb)  # curve A minus curve B along v

    # Lobes are separated by the zero crossings of d (the metastable point and
    # the two stable points). Measure the largest |d| on each side of the
    # metastable crossing nearest u = 0.
    zc = np.where(np.diff(np.sign(d)) != 0)[0]
    if len(zc) == 0:
        return 0.0, (0.0, 0.0), []
    mid = zc[np.argmin(np.abs(u[zc]))]
    left, right = slice(0, mid + 1), slice(mid + 1, None)
    lobes, squares = [], []
    for sl in (left, right):
        dd = d[sl]
        if dd.size == 0:
            lobes.append(0.0)
            continue
        k = np.argmax(np.abs(dd))
        diag = abs(dd[k])
        side = diag / s2
        # recover the lower-left corner of the square for plotting
        uk = u[sl][k]
        v_low = min(np.interp(uk, ua, va), np.interp(uk, ub, vb))
        x0 = (uk + v_low) / s2
        y0 = (v_low - uk) / s2
        lobes.append(side)
        squares.append((x0, y0, side))
    # a lobe that collapsed (cell monostable) means SNM = 0
    if len(zc) < 3:
        return 0.0, tuple(lobes), squares
    return float(min(lobes)), tuple(lobes), squares


def hold_snm(cond: Condition):
    a, b = butterfly(cond, wl_on=False)
    return snm_largest_square(a, b)[0]


def read_snm(cond: Condition):
    a, b = butterfly(cond, wl_on=True)
    return snm_largest_square(a, b)[0]


# --------------------------------------------------------------------------- #
# Write margin: ramp BL down with WL on; cell starts at Q = 1
# --------------------------------------------------------------------------- #
def write_margin(cond: Condition, t_ramp: float = 200e-9) -> float:
    v = cond.vdd
    net = (_header(cond)
           + f"Vwl wl 0 {v}\nVblb blb 0 {v}\n"
           + f"Vbl bl 0 PWL(0 {v} 1n {v} {1e-9 + t_ramp:.3e} 0)\n"
           + _mos(cond, "PU1", "q", "qb", "vdd", "vdd", "pmos") + "\n"
           + _mos(cond, "PD1", "q", "qb", "0", "0", "nmos") + "\n"
           + _mos(cond, "AX1", "bl", "wl", "q", "0", "nmos") + "\n"
           + _mos(cond, "PU2", "qb", "q", "vdd", "vdd", "pmos") + "\n"
           + _mos(cond, "PD2", "qb", "q", "0", "0", "nmos") + "\n"
           + _mos(cond, "AX2", "blb", "wl", "qb", "0", "nmos") + "\n"
           + f".ic v(q)={v} v(qb)=0\n"
           + ".control\nset wr_singlescale\nset wr_vecnames\n"
           + f"tran 0.2n {1e-9 + t_ramp:.3e} uic\nwrdata __OUT__ v(q) v(qb) v(bl)\nquit\n.endc\n.end\n")
    d = _run(net)
    q, qb, bl = d[:, 1], d[:, 2], d[:, 3]
    flip = np.where(q < qb)[0]
    if len(flip) == 0:
        return 0.0  # never flipped: not writable
    return float(bl[flip[0]])


# --------------------------------------------------------------------------- #
# Monte Carlo mismatch
# --------------------------------------------------------------------------- #
A_VT = 2.0e-3 * 1e-6  # Pelgrom coefficient: 2.0 mV*um (typical for 45nm-class bulk)


def sigma_vth(width: float) -> float:
    return A_VT / np.sqrt(width * L)


def random_mismatch(sizing: Sizing, rng: np.random.Generator) -> dict:
    w = {"PU": sizing.w_pu, "PD": sizing.w_pd, "AX": sizing.w_ax}
    return {d: float(rng.normal(0.0, sigma_vth(w[d[:2]]))) for d in DEVICES}
