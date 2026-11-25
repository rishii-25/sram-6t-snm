"""Run every sweep, write results/*.csv, results/summary.json and results/figures/*.png.

Usage:  python src/run_all.py [--mc 1000] [--jobs 1]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sram import (CORNERS, Condition, Sizing, butterfly, hold_snm, random_mismatch,  # noqa: E402
                  read_snm, sigma_vth, snm_largest_square, write_margin)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
CORNER_COLORS = {"TT": "#222222", "FF": "#d62728", "SS": "#1f77b4", "FS": "#2ca02c", "SF": "#ff7f0e"}


def all_metrics(cond: Condition) -> dict:
    return {"hold_snm": hold_snm(cond), "read_snm": read_snm(cond), "write_margin": write_margin(cond)}


def _row(args):
    vdd, corner, temp, w_pd = args
    c = Condition(vdd=vdd, corner=corner, temp=temp, sizing=Sizing(w_pd=w_pd))
    m = all_metrics(c)
    return {"vdd": vdd, "corner": corner, "temp_C": temp, "cell_ratio": round(w_pd / 90e-9, 3), **m}


def _mc_sample(args):
    vdd, seed = args
    rng = np.random.default_rng(seed)
    c = Condition(vdd=vdd)
    c.mismatch = random_mismatch(c.sizing, rng)
    return {"vdd": vdd, "seed": seed, "read_snm": read_snm(c), "hold_snm": hold_snm(c),
            "write_margin": write_margin(c), **{f"dvth_{k}": v for k, v in c.mismatch.items()}}


# --------------------------------------------------------------------------- #
def fig_butterfly():
    c = Condition()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    out = {}
    for ax, wl_on, title in ((axes[0], False, "Hold (WL = 0)"), (axes[1], True, "Read (WL = VDD, BL = BLB = VDD)")):
        a, b = butterfly(c, wl_on)
        snm, lobes, squares = snm_largest_square(a, b)
        ax.plot(*a, color="#1f77b4", lw=1.8, label="Inverter 1: QB = f(Q)")
        ax.plot(*b, color="#d62728", lw=1.8, label="Inverter 2: Q = f(QB)")
        for x0, y0, s in squares:
            ax.add_patch(plt.Rectangle((x0, y0), s, s, fill=False, ls="--", lw=1.2, color="#444"))
        ax.set_title(f"{title}\nSNM = {snm * 1e3:.0f} mV")
        ax.set_xlabel("V(Q) [V]"); ax.set_ylabel("V(QB) [V]")
        ax.set_xlim(0, c.vdd); ax.set_ylim(0, c.vdd); ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="upper right")
        out["hold" if not wl_on else "read"] = snm
    fig.suptitle("6T SRAM butterfly curves, PTM 45nm HP, TT, VDD = 1.0 V, 27 °C, CR = 2")
    fig.tight_layout(); fig.savefig(FIG / "butterfly_hold_read.png"); plt.close(fig)
    return out


def fig_vdd_corners(df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, col, title in zip(axes, ["hold_snm", "read_snm", "write_margin"],
                              ["Hold SNM", "Read SNM", "Write margin"]):
        for corner in CORNERS:
            d = df[df.corner == corner].sort_values("vdd")
            ax.plot(d.vdd, d[col] * 1e3, "o-", ms=4, color=CORNER_COLORS[corner], label=corner)
        ax.set_title(title); ax.set_xlabel("VDD [V]"); ax.set_ylabel("mV")
    axes[0].legend(fontsize=8)
    fig.suptitle("Stability vs supply voltage across process corners (27 °C, CR = 2)")
    fig.tight_layout(); fig.savefig(FIG / "metrics_vs_vdd_corners.png"); plt.close(fig)


def fig_cell_ratio(df):
    d = df.sort_values("cell_ratio")
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    ax.plot(d.cell_ratio, d.read_snm * 1e3, "o-", label="Read SNM")
    ax.plot(d.cell_ratio, d.write_margin * 1e3, "s-", label="Write margin")
    ax.plot(d.cell_ratio, d.hold_snm * 1e3, "^--", alpha=0.6, label="Hold SNM")
    ax.set_xlabel("Cell ratio  W_PD / W_AX"); ax.set_ylabel("mV")
    ax.set_title("Read stability vs writability trade-off\n(TT, VDD = 1.0 V, 27 °C)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "cell_ratio_tradeoff.png"); plt.close(fig)


def fig_mc(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, col, title in ((axes[0], "read_snm", "Read SNM"), (axes[1], "write_margin", "Write margin")):
        for vdd, color in zip(sorted(df.vdd.unique(), reverse=True), ["#1f77b4", "#d62728"]):
            x = df[df.vdd == vdd][col] * 1e3
            ax.hist(x, bins=40, alpha=0.55, color=color,
                    label=f"VDD {vdd:.1f} V: μ={x.mean():.0f}, σ={x.std():.1f} mV")
        ax.set_title(f"{title} under random Vth mismatch"); ax.set_xlabel("mV"); ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.suptitle(f"Monte Carlo ({len(df) // df.vdd.nunique()} samples per VDD, Pelgrom A_VT = 2 mV·µm, TT, 27 °C)")
    fig.tight_layout(); fig.savefig(FIG / "monte_carlo_hist.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=1000, help="Monte Carlo samples per VDD")
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers (1 = sequential)")
    args = ap.parse_args()
    t0 = time.time()
    summary = {}

    print("[1/5] butterfly curves")
    summary["nominal_TT_1V0"] = fig_butterfly()

    ex = ProcessPoolExecutor(args.jobs) if args.jobs > 1 else None
    pmap = (lambda f, it, **kw: list(ex.map(f, it, **kw))) if ex else (lambda f, it, **kw: [f(x) for x in it])
    print("[2/5] VDD x corner sweep")
    vdds = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    rows = pmap(_row, [(v, c, 27.0, 180e-9) for c in CORNERS for v in vdds])
    df_vc = pd.DataFrame(rows); df_vc.to_csv(RES / "vdd_corner_sweep.csv", index=False)
    fig_vdd_corners(df_vc)

    print("[3/5] temperature sweep")
    rows = pmap(_row, [(1.0, "TT", t, 180e-9) for t in (-40.0, 0.0, 27.0, 85.0, 125.0)])
    df_t = pd.DataFrame(rows); df_t.to_csv(RES / "temperature_sweep.csv", index=False)

    print("[4/5] cell-ratio sweep")
    rows = pmap(_row, [(1.0, "TT", 27.0, cr * 90e-9) for cr in np.arange(1.0, 3.01, 0.25)])
    df_cr = pd.DataFrame(rows); df_cr.to_csv(RES / "cell_ratio_sweep.csv", index=False)
    fig_cell_ratio(df_cr)

    print(f"[5/5] Monte Carlo ({args.mc} samples x 2 VDD)")
    jobs = [(v, 1000 * i + int(v * 10)) for v in (1.0, 0.7) for i in range(args.mc)]
    df_mc = pd.DataFrame(pmap(_mc_sample, jobs, chunksize=16))
    df_mc.to_csv(RES / "monte_carlo.csv", index=False)
    fig_mc(df_mc)
    if ex:
        ex.shutdown()

    # ---- summary ---------------------------------------------------------- #
    nom = df_vc[(df_vc.corner == "TT") & (np.isclose(df_vc.vdd, 1.0))].iloc[0]
    summary["TT_1V0"] = {k: round(float(nom[k]) * 1e3, 1) for k in ("hold_snm", "read_snm", "write_margin")}
    worst = {}
    for col in ("hold_snm", "read_snm", "write_margin"):
        d = df_vc[np.isclose(df_vc.vdd, 1.0)]
        r = d.loc[d[col].idxmin()]
        worst[col] = {"corner": r.corner, "mV": round(float(r[col]) * 1e3, 1)}
    summary["worst_corner_at_1V0"] = worst
    summary["temperature_TT_1V0_mV"] = {
        f"{int(r.temp_C)}C": {k: round(float(r[k]) * 1e3, 1) for k in ("hold_snm", "read_snm", "write_margin")}
        for _, r in df_t.iterrows()}
    summary["sigma_vth_mV"] = {k: round(sigma_vth(w) * 1e3, 2) for k, w in
                               (("PD", 180e-9), ("AX", 90e-9), ("PU", 90e-9))}
    mc = {}
    for vdd, d in df_mc.groupby("vdd"):
        entry = {}
        for col in ("read_snm", "write_margin", "hold_snm"):
            mu, sd = float(d[col].mean()), float(d[col].std())
            entry[col] = {"mean_mV": round(mu * 1e3, 1), "std_mV": round(sd * 1e3, 2),
                          "mu_over_sigma": round(mu / sd, 2) if sd > 0 else None,
                          "min_mV": round(float(d[col].min()) * 1e3, 1),
                          "fails": int((d[col] <= 0).sum())}
        mc[f"{vdd:.1f}V"] = entry
    summary["monte_carlo"] = mc
    summary["mc_samples_per_vdd"] = args.mc
    summary["runtime_s"] = round(time.time() - t0, 1)
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
