#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Publication figures for the UQ-FEM results report.

Deterministic: no random number generation anywhere. Re-running reproduces
byte-identical numbers (PDF bytes may differ only in the embedded timestamp,
which is pinned below).

Outputs vector PDFs into  UFEM_2.0/report/figures/ :
    fig_ld_family.pdf
    fig_damage_family.pdf
    fig_design_censoring.pdf
    fig_failure_rate_ctop.pdf
    fig_peak_stats.pdf
    fig_peak_vs_inputs.pdf
    fig_E_collinearity.pdf
    fig_stiffness_vs_ctop.pdf

Design follows the `dataviz` skill: validated colorblind-safe categorical
palette (slots 1 and 2 of the reference palette, which validate under the
all-pairs gate), recessive hairline chrome, thin marks, no in-figure titles
(captions live in LaTeX), units on every axis label.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]                      # .../ufem_env
UFEM = ROOT / "UFEM_2.0"
AUDIT = UFEM / "data_audit"
OUT = UFEM / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INPUTS_CSV = ROOT / "uq_lhs_samples_training.csv"
VALIDITY_CSV = AUDIT / "sample_validity.csv"
RF2_NPY = AUDIT / "RF2_on_common_U2_grid.npy"
U2_NPY = AUDIT / "common_U2_grid.npy"
IDS_CSV = AUDIT / "common_grid_sample_ids.csv"
DAMAGE_CSV = (ROOT / "Scripts_2_0" / "03_postprocess" / "01_extracted_data"
              / "damage_evolution_full.csv")

# --------------------------------------------------------------------------
# Style  (single consistent style for every figure)
# --------------------------------------------------------------------------
# Reference categorical palette, light surface. Slots 1-3 clear the all-pairs
# colourblind gate (worst CVD dE 9.2, normal-vision dE 24.0, OKLab x100);
# only slots 1 and 2 are used for identity anywhere in this report.
C_SERIES_1 = "#2a78d6"   # blue   - primary / completed runs
C_SERIES_2 = "#eb6834"   # orange - secondary / failed (censored) runs
C_BAND = "#9ec5f4"       # blue step 200 - pointwise envelope fill
C_INK = "#0b0b0b"        # primary ink
C_INK_2 = "#52514e"      # secondary ink
C_MUTED = "#898781"      # axis / tick labels, individual realisations
C_GRID = "#e1e0d9"       # hairline gridline
C_AXIS = "#c3c2b7"       # baseline / spine

FIG_W = 6.3              # inches ~= \linewidth of a one-column A4 body

plt.rcParams.update({
    "pdf.fonttype": 42,             # embed TrueType, keep text selectable
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "pdf.compression": 6,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    # Sized so that even if LaTeX scales a 6.3 in figure down to a narrower
    # \linewidth (worst realistic case ~0.90), every glyph stays >= 8 pt.
    "font.size": 9.5,
    "axes.labelsize": 10.0,
    "axes.titlesize": 10.0,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 9.0,
    "figure.titlesize": 10.0,
    "axes.labelcolor": C_INK,
    "text.color": C_INK,
    "xtick.color": C_MUTED,
    "ytick.color": C_MUTED,
    "xtick.labelcolor": C_INK_2,
    "ytick.labelcolor": C_INK_2,
    "axes.edgecolor": C_AXIS,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",          # solid hairline grid, never dashed
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.6,
    "legend.borderaxespad": 0.4,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
# Pin the PDF creation date so repeated runs are byte-reproducible.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")

ANNOT = dict(fontsize=9.0, color=C_INK_2, ha="left", va="top",
             bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                       edgecolor=C_GRID, linewidth=0.5, alpha=0.92))

RESULTS: dict = {}


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, format="pdf")
    # Optional raster preview for visual QA only (set UFEM_FIG_PNG=<dir>).
    png_dir = os.environ.get("UFEM_FIG_PNG")
    if png_dir:
        Path(png_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(png_dir) / name.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print(f"  wrote {path}  ({path.stat().st_size/1024:.1f} kB)")


def pearson(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return pearson(rx, ry)


# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
print("Loading data ...")
inputs = pd.read_csv(INPUTS_CSV)
valid = pd.read_csv(VALIDITY_CSV)

design = inputs.merge(
    valid[["sample_id", "status", "peak_RF2", "U2_at_peak", "k0_N_per_mm"]],
    on="sample_id", how="left", suffixes=("", "_v"))
design["completed"] = design["status"].eq("valid")

ok = design[design["completed"]].copy()
bad = design[~design["completed"]].copy()

U2_grid = np.load(U2_NPY)                       # (201,) mm
RF2_grid = np.load(RF2_NPY) / 1000.0            # (198, 201) N -> kN
grid_ids = pd.read_csv(IDS_CSV)["sample_id"].to_numpy()
assert RF2_grid.shape == (grid_ids.size, U2_grid.size)

RESULTS["n_design"] = int(len(design))
RESULTS["n_completed"] = int(len(ok))
RESULTS["n_failed"] = int(len(bad))
RESULTS["overall_failure_rate"] = float(len(bad) / len(design))

# Damage curves: de-duplicate repeated times per job, resample onto U2 grid.
dmg_raw = pd.read_csv(DAMAGE_CSV)
dmg_raw = dmg_raw.drop_duplicates(subset=["job", "time"], keep="first")
dmg_raw = dmg_raw.sort_values(["job", "time"], kind="mergesort")

dmg_jobs = sorted(dmg_raw["job"].unique())
DMG = np.full((len(dmg_jobs), U2_grid.size), np.nan)
for i, job in enumerate(dmg_jobs):
    sub = dmg_raw.loc[dmg_raw["job"] == job, ["U2", "DAMAGEC_max"]]
    u = np.clip(sub["U2"].to_numpy(float), 0.0, None)
    d = sub["DAMAGEC_max"].to_numpy(float)
    # enforce a strictly increasing abscissa for interpolation
    order = np.argsort(u, kind="mergesort")
    u, d = u[order], d[order]
    keep = np.concatenate(([True], np.diff(u) > 0))
    u, d = u[keep], d[keep]
    DMG[i] = np.interp(U2_grid, u, d, left=d[0], right=d[-1])

RESULTS["n_damage_jobs"] = int(len(dmg_jobs))
print(f"  {len(design)} design points | {len(ok)} completed | {len(bad)} failed"
      f" | {RF2_grid.shape[0]} L-D curves | {len(dmg_jobs)} damage curves")


def band(mat: np.ndarray):
    """Pointwise 5th / 50th / 95th percentile across realisations."""
    p05, p50, p95 = np.nanpercentile(mat, [5, 50, 95], axis=0)
    return p05, p50, p95


def family_panel(ax, x, mat, lo, med, hi, lab_lo, lab_hi):
    for row in mat:
        ax.plot(x, row, color=C_MUTED, lw=0.35, alpha=0.16,
                solid_capstyle="round", zorder=1)
    ax.fill_between(x, lo, hi, color=C_BAND, alpha=0.55, lw=0, zorder=2)
    ax.plot(x, med, color=C_SERIES_1, lw=1.8, solid_capstyle="round", zorder=3)
    return [
        Line2D([], [], color=C_MUTED, lw=0.9, alpha=0.55,
               label=f"individual runs (n = {mat.shape[0]})"),
        Patch(facecolor=C_BAND, alpha=0.55, edgecolor="none",
              label=f"pointwise {lab_lo} to {lab_hi} envelope"),
        Line2D([], [], color=C_SERIES_1, lw=1.8, label="pointwise median"),
    ]


# ==========================================================================
# 1. Load-displacement family
# ==========================================================================
print("fig_ld_family.pdf")
lo, med, hi = band(RF2_grid)
fig, ax = plt.subplots(figsize=(FIG_W, 3.5))
handles = family_panel(ax, U2_grid, RF2_grid, lo, med, hi, "5%", "95%")
ax.set_xlabel("Mid-span displacement $U_2$ [mm]")
ax.set_ylabel("Reaction force $RF_2$ [kN]")
ax.set_xlim(0, U2_grid.max())
ax.set_ylim(0, None)
ax.legend(handles=handles, loc="lower right")
save(fig, "fig_ld_family.pdf")

imed = int(np.argmax(med))
RESULTS["ld_median_peak_kN"] = float(med.max())
RESULTS["ld_median_peak_U2_mm"] = float(U2_grid[imed])
RESULTS["ld_envelope_at_median_peak_kN"] = [float(lo[imed]), float(hi[imed])]
RESULTS["ld_median_at_U2_20mm_kN"] = float(med[-1])
RESULTS["ld_envelope_at_U2_20mm_kN"] = [float(lo[-1]), float(hi[-1])]
RESULTS["ld_envelope_width_max_kN"] = float(np.max(hi - lo))
RESULTS["ld_envelope_width_max_at_U2_mm"] = float(U2_grid[int(np.argmax(hi - lo))])

# ==========================================================================
# 2. Damage family
# ==========================================================================
print("fig_damage_family.pdf")
SAT = 0.947
dlo, dmed, dhi = band(DMG)
fig, ax = plt.subplots(figsize=(FIG_W, 3.5))
handles = family_panel(ax, U2_grid, DMG, dlo, dmed, dhi, "5%", "95%")
ax.axhline(SAT, color=C_INK_2, lw=0.9, ls=(0, (4, 2.5)), zorder=4)
ax.annotate(f"saturation $d_c = {SAT:.3f}$",
            xy=(0.985, SAT), xycoords=("axes fraction", "data"),
            xytext=(0, -4), textcoords="offset points",
            ha="right", va="top", fontsize=9.0, color=C_INK_2)
ax.set_xlabel("Mid-span displacement $U_2$ [mm]")
ax.set_ylabel("Max. compressive damage $d_{c,\\max}$ [-]")
ax.set_xlim(0, U2_grid.max())
ax.set_ylim(0, 1.02)
ax.legend(handles=handles, loc="lower right")
save(fig, "fig_damage_family.pdf")

RESULTS["damage_saturation_level"] = SAT
RESULTS["damage_median_at_U2_10mm"] = float(np.interp(10.0, U2_grid, dmed))
RESULTS["damage_envelope_at_U2_10mm"] = [float(np.interp(10.0, U2_grid, dlo)),
                                         float(np.interp(10.0, U2_grid, dhi))]
u_half = [float(U2_grid[np.argmax(c >= 0.5 * SAT)]) for c in DMG]
RESULTS["damage_U2_at_half_sat_median_mm"] = float(np.median(u_half))
RESULTS["damage_U2_at_half_sat_p05_p95_mm"] = [float(np.percentile(u_half, 5)),
                                               float(np.percentile(u_half, 95))]
RESULTS["damage_frac_saturated_at_20mm"] = float(np.mean(DMG[:, -1] >= SAT - 1e-6))

# ==========================================================================
# 3. Design-space censoring: scatter matrix
# ==========================================================================
print("fig_design_censoring.pdf")
VARS = [("Fcm_MPa", "$f_{cm}$ [MPa]"),
        ("c_nom_bottom_mm", "$c_{nom,bot}$ [mm]"),
        ("c_nom_top_mm", "$c_{nom,top}$ [mm]")]
n = len(VARS)
fig, axes = plt.subplots(n, n, figsize=(FIG_W, FIG_W * 0.92),
                         sharex="col")
for i in range(n):
    for j in range(n):
        ax = axes[i, j]
        xk, xl = VARS[j]
        yk, yl = VARS[i]
        if i == j:
            bins = np.histogram_bin_edges(design[xk], bins=18)
            ax.hist(ok[xk], bins=bins, color=C_SERIES_1, alpha=0.75,
                    lw=0, zorder=2)
            ax.hist(bad[xk], bins=bins, histtype="step", color=C_SERIES_2,
                    lw=1.1, zorder=3)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.grid(False)
        elif i > j:
            ax.scatter(bad[xk], bad[yk], s=7, facecolor="none",
                       edgecolor=C_SERIES_2, linewidth=0.55, alpha=0.85,
                       zorder=2)
            ax.scatter(ok[xk], ok[yk], s=7, color=C_SERIES_1, lw=0,
                       alpha=0.85, zorder=3)
        else:
            ax.axis("off")
            continue
        if i == n - 1:
            ax.set_xlabel(xl)
        if j == 0 and i > 0:
            ax.set_ylabel(yl)
axes[0, 0].set_ylabel("count")
handles = [
    Line2D([], [], marker="o", ls="none", ms=4.5, color=C_SERIES_1,
           label=f"completed  (n = {len(ok)})"),
    Line2D([], [], marker="o", ls="none", ms=4.5, markerfacecolor="none",
           markeredgecolor=C_SERIES_2, markeredgewidth=0.9,
           label=f"failed / missing  (n = {len(bad)})"),
]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.98),
           ncol=1)
fig.tight_layout(pad=0.4)
save(fig, "fig_design_censoring.pdf")

RESULTS["censoring_means"] = {
    k: {"completed": float(ok[k].mean()), "failed": float(bad[k].mean())}
    for k, _ in VARS}

# ==========================================================================
# 4. Failure rate per input quartile
# ==========================================================================
print("fig_failure_rate_ctop.pdf")


def quartile_failure(col: str):
    q = pd.qcut(design[col], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    grp = design.groupby(q, observed=True)
    rate = grp["completed"].apply(lambda s: 1.0 - s.mean())
    cnt = grp.size()
    edges = np.quantile(design[col], [0, .25, .5, .75, 1.0])
    return rate, cnt, edges


fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.9), sharey=True)
panel_cfg = [("c_nom_top_mm", "$c_{nom,top}$ quartile [mm]"),
             ("Fcm_MPa", "$f_{cm}$ quartile [MPa]")]
for ax, (col, xlab) in zip(axes, panel_cfg):
    rate, cnt, edges = quartile_failure(col)
    xs = np.arange(4)
    ax.bar(xs, rate.to_numpy(), width=0.62, color=C_SERIES_1, lw=0, zorder=2)
    for x, r in zip(xs, rate.to_numpy()):
        ax.annotate(f"{r:.2f}", xy=(x, r), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9.0, color=C_INK_2, zorder=5,
                    bbox=dict(boxstyle="square,pad=0.10", facecolor="white",
                              edgecolor="none"))
    labs = [f"Q{k+1}\n{edges[k]:.0f} to {edges[k+1]:.0f}" for k in range(4)]
    ax.set_xticks(xs)
    ax.set_xticklabels(labs)
    ax.set_xlabel(xlab)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="x", visible=False)
    RESULTS[f"failure_rate_by_{col}"] = {
        f"Q{k+1}": {"rate": float(rate.iloc[k]), "n": int(cnt.iloc[k]),
                    "range": [float(edges[k]), float(edges[k+1])]}
        for k in range(4)}
axes[0].set_ylabel("failure rate [-]")
for ax in axes:
    ax.axhline(RESULTS["overall_failure_rate"], color=C_INK_2, lw=0.8,
               ls=(0, (4, 2.5)), zorder=3)
axes[0].legend(handles=[Line2D([], [], color=C_INK_2, lw=0.8,
                               ls=(0, (4, 2.5)),
                               label="overall rate "
                                     f"{RESULTS['overall_failure_rate']:.2f}"
                                     f"  ({RESULTS['n_failed']}/"
                                     f"{RESULTS['n_design']})")],
               loc="upper right")
fig.tight_layout(pad=0.4)
save(fig, "fig_failure_rate_ctop.pdf")

# ==========================================================================
# 5. Peak-response statistics
# ==========================================================================
print("fig_peak_stats.pdf")
peak_kN = ok["peak_RF2"].to_numpy(float) / 1000.0
u_peak = ok["U2_at_peak"].to_numpy(float)

fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.9))
for ax, data, xlab in [(axes[0], peak_kN, "Peak load $RF_{2,\\max}$ [kN]"),
                       (axes[1], u_peak,
                        "Displacement at peak $U_{2,peak}$ [mm]")]:
    counts, _, _ = ax.hist(data, bins=22, color=C_SERIES_1, alpha=0.85, lw=0,
                           zorder=2)
    m, s = float(np.mean(data)), float(np.std(data, ddof=1))
    ax.axvline(m, color=C_SERIES_2, lw=1.4, zorder=3)
    ax.annotate(f"n = {len(data)}\nmean = {m:.2f}\nstd = {s:.2f}\n"
                f"CoV = {s/m*100:.1f}%",
                xy=(0.03, 0.97), xycoords="axes fraction", **ANNOT)
    ax.set_xlabel(xlab)
    ax.set_ylim(0, counts.max() * 1.42)     # headroom for the annotation box
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("count")
axes[1].set_ylabel("count")
axes[1].legend(handles=[Line2D([], [], color=C_SERIES_2, lw=1.4,
                               label="mean")], loc="upper right")
fig.tight_layout(pad=0.4)
save(fig, "fig_peak_stats.pdf")

RESULTS["peak_load_kN"] = {
    "mean": float(np.mean(peak_kN)), "std": float(np.std(peak_kN, ddof=1)),
    "cov_pct": float(np.std(peak_kN, ddof=1) / np.mean(peak_kN) * 100),
    "min": float(peak_kN.min()), "max": float(peak_kN.max()),
    "median": float(np.median(peak_kN)),
    "p05_p95": [float(np.percentile(peak_kN, 5)),
                float(np.percentile(peak_kN, 95))]}
RESULTS["U2_at_peak_mm"] = {
    "mean": float(np.mean(u_peak)), "std": float(np.std(u_peak, ddof=1)),
    "cov_pct": float(np.std(u_peak, ddof=1) / np.mean(u_peak) * 100),
    "min": float(u_peak.min()), "max": float(u_peak.max()),
    "median": float(np.median(u_peak)),
    "p05_p95": [float(np.percentile(u_peak, 5)),
                float(np.percentile(u_peak, 95))]}

# ==========================================================================
# 6. Peak load vs inputs
# ==========================================================================
print("fig_peak_vs_inputs.pdf")
fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 2.5), sharey=True)
cfg = [("Fcm_MPa", "$f_{cm}$ [MPa]"),
       ("c_nom_top_mm", "$c_{nom,top}$ [mm]"),
       ("c_nom_bottom_mm", "$c_{nom,bot}$ [mm]")]
RESULTS["pearson_peak_vs_inputs"] = {}
for ax, (col, xlab) in zip(axes, cfg):
    x = ok[col].to_numpy(float)
    r = pearson(x, peak_kN)
    RESULTS["pearson_peak_vs_inputs"][col] = float(r)
    ax.scatter(x, peak_kN, s=9, color=C_SERIES_1, lw=0, alpha=0.7, zorder=2)
    ax.annotate(f"$r$ = {r:+.3f}", xy=(0.04, 0.96), xycoords="axes fraction",
                **ANNOT)
    ax.set_xlabel(xlab)
axes[0].set_ylabel("Peak load [kN]")
fig.tight_layout(pad=0.4)
save(fig, "fig_peak_vs_inputs.pdf")

# ==========================================================================
# 7. E - fcm collinearity
# ==========================================================================
print("fig_E_collinearity.pdf")
fcm = design["Fcm_MPa"].to_numpy(float)
E = design["E_MPa"].to_numpy(float)
xs = np.linspace(fcm.min(), fcm.max(), 400)
ec2 = 22000.0 * (xs / 10.0) ** 0.3
resid = E - 22000.0 * (fcm / 10.0) ** 0.3

fig, ax = plt.subplots(figsize=(FIG_W, 3.2))
ax.scatter(fcm, E, s=11, color=C_SERIES_1, lw=0, alpha=0.6, zorder=2,
           label=f"LHS design points (n = {len(design)})")
ax.plot(xs, ec2, color=C_SERIES_2, lw=1.6, zorder=3,
        label=r"EC2: $E = 22000\,(f_{cm}/10)^{0.3}$")
ax.set_xlabel("$f_{cm}$ [MPa]")
ax.set_ylabel("$E$ [MPa]")
ax.annotate("Spearman $\\rho$ = 1.0\n(deterministic dependence,\n"
            f"max residual = {np.max(np.abs(resid)):.1e} MPa)",
            xy=(0.04, 0.96), xycoords="axes fraction", **ANNOT)
ax.legend(loc="lower right")
fig.tight_layout(pad=0.4)
save(fig, "fig_E_collinearity.pdf")

RESULTS["E_fcm"] = {
    "spearman_rho": float(spearman(fcm, E)),
    "pearson_r": float(pearson(fcm, E)),
    "max_abs_residual_MPa": float(np.max(np.abs(resid))),
    "E_range_MPa": [float(E.min()), float(E.max())]}

# ==========================================================================
# 8. Initial stiffness vs c_nom_top
# ==========================================================================
print("fig_stiffness_vs_ctop.pdf")
k0 = ok["k0_N_per_mm"].to_numpy(float) / 1000.0     # kN/mm
ctop = ok["c_nom_top_mm"].to_numpy(float)
r_k0 = pearson(ctop, k0)

fig, ax = plt.subplots(figsize=(FIG_W, 3.2))
ax.scatter(ctop, k0, s=11, color=C_SERIES_1, lw=0, alpha=0.7, zorder=2)
ax.set_xlabel("$c_{nom,top}$ [mm]")
ax.set_ylabel("Initial stiffness $k_0$ [kN/mm]")
ax.annotate(f"$r$ = {r_k0:+.3f}   (n = {len(ok)})",
            xy=(0.04, 0.96), xycoords="axes fraction", **ANNOT)
fig.tight_layout(pad=0.4)
save(fig, "fig_stiffness_vs_ctop.pdf")

RESULTS["k0_kN_per_mm"] = {
    "pearson_r_vs_c_nom_top": float(r_k0),
    "pearson_r_vs_Fcm": float(pearson(ok["Fcm_MPa"], k0)),
    "pearson_r_vs_c_nom_bottom": float(pearson(ok["c_nom_bottom_mm"], k0)),
    "mean": float(np.mean(k0)), "std": float(np.std(k0, ddof=1)),
    "min": float(k0.min()), "max": float(k0.max())}

# --------------------------------------------------------------------------
print("\n=== computed numbers ===")
print(json.dumps(RESULTS, indent=2))
(HERE.parent / "figure_stats.json").write_text(
    json.dumps(RESULTS, indent=2), encoding="utf-8")
print("\nAll figures written to", OUT)
