# -*- coding: utf-8 -*-
"""
Audit of Abaqus FEM runs for UQ surrogate modelling.

Inputs:
  - uq_lhs_samples_training.csv           (LHS design, 4 uncertain inputs)
  - load_displacement_full.csv            (job, time, U2, RF2)
  - damage_evolution_full.csv             (job, time, U2, DAMAGEC_max)

Outputs (into UFEM_2.0/data_audit/):
  - sample_validity.csv
  - audit_summary.json
  - audit_script.py  (copy of this file)
"""

import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
ROOT = r"c:\Users\jidro\Documents\Elijah\RUB\Third Semester\Uncertainty FEM\Project\ufem_env"
INPUTS_CSV = os.path.join(ROOT, "uq_lhs_samples_training.csv")
EXTRACTED = os.path.join(ROOT, "Scripts_2_0", "03_postprocess", "01_extracted_data")
LD_CSV = os.path.join(EXTRACTED, "load_displacement_full.csv")
DMG_CSV = os.path.join(EXTRACTED, "damage_evolution_full.csv")
OUTDIR = os.path.join(ROOT, "UFEM_2.0", "data_audit")

TARGET_TIME = 1.0          # full step time
TIME_FRAC_OK = 0.95        # >= 95 % of target time => "reached full step"
MIN_POINTS = 20            # minimum number of increments to be usable
POSTPEAK_DROP = 0.10       # >= 10 % load drop after peak => softening captured
POSTPEAK_MIN_PTS = 5       # minimum points after peak
STIFF_FRAC = 0.10          # first 10 % of (displacement at peak) for initial stiffness

os.makedirs(OUTDIR, exist_ok=True)


def jobid_to_int(j):
    try:
        return int(str(j).split("_")[-1])
    except Exception:
        return -1


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
inp = pd.read_csv(INPUTS_CSV)
ld = pd.read_csv(LD_CSV)
dmg = pd.read_csv(DMG_CSV)

for df in (ld, dmg):
    df["sample_id"] = df["job"].map(jobid_to_int)

ld_jobs = sorted(ld["sample_id"].unique().tolist())
dmg_jobs = sorted(dmg["sample_id"].unique().tolist())
all_ids = sorted(inp["sample_id"].unique().tolist())

inventory = {
    "n_input_samples": int(len(inp)),
    "n_unique_input_sample_ids": int(len(all_ids)),
    "input_sample_id_min": int(min(all_ids)),
    "input_sample_id_max": int(max(all_ids)),
    "n_jobs_load_displacement": int(len(ld_jobs)),
    "n_jobs_damage": int(len(dmg_jobs)),
    "n_rows_load_displacement": int(len(ld)),
    "n_rows_damage": int(len(dmg)),
    "missing_from_load_displacement": [int(i) for i in all_ids if i not in set(ld_jobs)],
    "missing_from_damage": [int(i) for i in all_ids if i not in set(dmg_jobs)],
    "missing_from_both": [int(i) for i in all_ids
                          if i not in set(ld_jobs) and i not in set(dmg_jobs)],
    "in_ld_not_in_damage": [int(i) for i in ld_jobs if i not in set(dmg_jobs)],
    "in_damage_not_in_ld": [int(i) for i in dmg_jobs if i not in set(ld_jobs)],
    "jobs_not_in_input_design": [int(i) for i in ld_jobs if i not in set(all_ids)],
}

# --------------------------------------------------------------------------
# 2. Per-job diagnostics
# --------------------------------------------------------------------------
ld_g = {sid: g for sid, g in ld.groupby("sample_id")}
dmg_g = {sid: g for sid, g in dmg.groupby("sample_id")}

rows = []
curves = {}  # sid -> (U2 monotone-increasing arr, RF2 arr) for interpolation study

for sid in all_ids:
    r = {"sample_id": sid, "job": "sample_%03d" % sid}
    g = ld_g.get(sid)
    if g is None:
        r.update({"present_ld": False, "present_damage": sid in dmg_g,
                  "status": "missing", "reason": "no rows in load_displacement_full.csv"})
        rows.append(r)
        continue

    # monotonicity is checked on the ORIGINAL file row order (as written by Abaqus)
    t_raw = g["time"].to_numpy(float)
    time_monotone = bool(np.all(np.diff(t_raw) >= 0))

    g = g.sort_values("time", kind="mergesort")
    t = g["time"].to_numpy(float)
    u = g["U2"].to_numpy(float)
    f = g["RF2"].to_numpy(float)

    n_nan = int(np.isnan(t).sum() + np.isnan(u).sum() + np.isnan(f).sum())
    n_dup_time = int(len(t) - len(np.unique(t)))
    n_dup_rows = int(g.duplicated(subset=["time", "U2", "RF2"]).sum())
    u_monotone = bool(np.all(np.diff(u) >= -1e-9))

    fin = np.isfinite(t) & np.isfinite(u) & np.isfinite(f)
    tc, uc, fc = t[fin], u[fin], f[fin]

    if len(tc) == 0:
        r.update({"present_ld": True, "present_damage": sid in dmg_g, "n_points": 0,
                  "n_nan": n_nan, "status": "unusable", "reason": "all rows non-finite"})
        rows.append(r)
        continue

    au = np.abs(uc)
    ipk = int(np.argmax(fc))
    peak_rf = float(fc[ipk])
    u_at_peak = float(uc[ipk])
    n_post = int(len(fc) - 1 - ipk)
    rf_final = float(fc[-1])
    postpeak_drop = float((peak_rf - fc[ipk:].min()) / peak_rf) if peak_rf > 0 else np.nan
    postpeak_drop_final = float((peak_rf - rf_final) / peak_rf) if peak_rf > 0 else np.nan

    # initial stiffness: least squares through origin on U2 <= STIFF_FRAC * u_at_peak
    k0 = np.nan
    n_k0 = 0
    if u_at_peak > 0:
        m = (uc > 0) & (uc <= STIFF_FRAC * u_at_peak)
        n_k0 = int(m.sum())
        if n_k0 >= 2:
            k0 = float(np.dot(uc[m], fc[m]) / np.dot(uc[m], uc[m]))
        elif len(uc) >= 3:  # fall back to first 3 points
            mm = slice(0, 3)
            uu, ff = uc[mm], fc[mm]
            if np.dot(uu, uu) > 0:
                k0 = float(np.dot(uu, ff) / np.dot(uu, uu))
                n_k0 = 3

    dg = dmg_g.get(sid)
    dmg_u_half = dmg_at_u10 = np.nan
    dmg_monotone = None
    if dg is not None:
        dg = dg.sort_values("time", kind="mergesort")
        dvals = dg["DAMAGEC_max"].to_numpy(float)
        dt = dg["time"].to_numpy(float)
        du = dg["U2"].to_numpy(float)
        dmg_final = float(dvals[-1]) if len(dvals) else np.nan
        dmg_max = float(np.nanmax(dvals)) if len(dvals) else np.nan
        dmg_final_time = float(dt[-1]) if len(dt) else np.nan
        dmg_n = int(len(dvals))
        dmg_nan = int(np.isnan(dvals).sum())
        dmg_monotone = bool(np.all(np.diff(dvals) >= -1e-12))
        if dmg_n > 2 and np.isfinite(dmg_max) and dmg_max > 0:
            # U2 at which damage first reaches 50 % of its own maximum (informative scalar,
            # unlike dmg_final which saturates at the CDP table cap for every run)
            i = int(np.argmax(dvals >= 0.5 * dmg_max))
            dmg_u_half = float(du[i])
            o = np.argsort(du, kind="mergesort")
            dmg_at_u10 = float(np.interp(10.0, du[o], dvals[o]))
    else:
        dmg_final = dmg_max = dmg_final_time = np.nan
        dmg_n = 0
        dmg_nan = 0

    r.update({
        "present_ld": True,
        "present_damage": dg is not None,
        "n_points": int(len(t)),
        "n_points_finite": int(len(tc)),
        "t_final": float(tc[-1]),
        "t_frac_of_target": float(tc[-1] / TARGET_TIME),
        "reached_full_step": bool(tc[-1] >= TIME_FRAC_OK * TARGET_TIME),
        "U2_max_abs": float(au.max()),
        "U2_final": float(uc[-1]),
        "peak_RF2": peak_rf,
        "U2_at_peak": u_at_peak,
        "idx_peak": ipk,
        "n_points_after_peak": n_post,
        "RF2_final": rf_final,
        "postpeak_drop_frac_max": postpeak_drop,
        "postpeak_drop_frac_final": postpeak_drop_final,
        "softening_captured": bool(n_post >= POSTPEAK_MIN_PTS and postpeak_drop >= POSTPEAK_DROP),
        "k0_N_per_mm": k0,
        "n_pts_for_k0": n_k0,
        "damage_final": dmg_final,
        "damage_max": dmg_max,
        "damage_U2_at_half_max": dmg_u_half,
        "damage_at_U2_10mm": dmg_at_u10,
        "damage_monotone": dmg_monotone,
        "damage_n_points": dmg_n,
        "damage_t_final": dmg_final_time,
        "n_nan": n_nan,
        "n_dup_time": n_dup_time,
        "n_dup_rows_exact": n_dup_rows,
        "time_monotone": time_monotone,
        "U2_monotone": u_monotone,
        "damage_n_nan": dmg_nan,
    })
    rows.append(r)

    if len(uc) >= MIN_POINTS:
        # keep strictly increasing U2 for interpolation feasibility check
        order = np.argsort(uc, kind="mergesort")
        uu, ff = uc[order], fc[order]
        keep = np.concatenate(([True], np.diff(uu) > 0))
        curves[sid] = (uu[keep], ff[keep])

diag = pd.DataFrame(rows)

# --------------------------------------------------------------------------
# 3. Pass / fail classification
# --------------------------------------------------------------------------
def classify(r):
    if r.get("status") == "missing":
        return "missing", "no rows in load_displacement_full.csv"
    if not r.get("present_ld", False):
        return "missing", "no rows in load_displacement_full.csv"
    reasons = []
    if r.get("n_nan", 0) > 0:
        reasons.append("NaN values present (%d)" % r["n_nan"])
    if r.get("n_points_finite", 0) < MIN_POINTS:
        reasons.append("too few points (%d < %d)" % (r.get("n_points_finite", 0), MIN_POINTS))
    if not r.get("time_monotone", True):
        reasons.append("non-monotone time")
    if reasons:
        return "unusable", "; ".join(reasons)
    if r.get("reached_full_step", False):
        return "valid", "reached t=%.3f (>=%.0f%% of target); %s" % (
            r["t_final"], TIME_FRAC_OK * 100,
            "softening captured" if r["softening_captured"] else "no clear post-peak softening")
    if r.get("softening_captured", False):
        return "partial", ("early termination at t=%.3f (%.1f%% of step) but peak captured "
                           "with %.1f%% post-peak load drop over %d points"
                           % (r["t_final"], 100 * r["t_frac_of_target"],
                              100 * r["postpeak_drop_frac_max"], r["n_points_after_peak"]))
    return "unusable", ("early termination at t=%.3f (%.1f%% of step), no post-peak softening "
                        "(%d pts after peak, %.1f%% drop)"
                        % (r["t_final"], 100 * r["t_frac_of_target"],
                           r["n_points_after_peak"], 100 * (r["postpeak_drop_frac_max"] or 0)))


cls = diag.apply(lambda r: pd.Series(classify(r), index=["status", "reason"]), axis=1)
diag["status"] = cls["status"]
diag["reason"] = cls["reason"]

status_counts = diag["status"].value_counts().to_dict()
status_counts = {k: int(v) for k, v in status_counts.items()}

# merge inputs
merged = inp.merge(diag, on="sample_id", how="outer").sort_values("sample_id")

failed = merged[merged["status"].isin(["missing", "unusable"])]
partial = merged[merged["status"] == "partial"]
good = merged[merged["status"] == "valid"]
usable = merged[merged["status"].isin(["valid", "partial"])]

INPUT_COLS = ["Fcm_MPa", "c_nom_bottom_mm", "c_nom_top_mm", "E_MPa"]


def describe_block(df, cols):
    out = {}
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy()
        if len(v) == 0:
            out[c] = None
            continue
        out[c] = {"n": int(len(v)), "mean": float(v.mean()),
                  "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  "cov": float(v.std(ddof=1) / v.mean()) if len(v) > 1 and v.mean() != 0 else None,
                  "min": float(v.min()), "p05": float(np.percentile(v, 5)),
                  "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                  "max": float(v.max())}
    return out


failure_cluster = {
    "failed_input_stats": describe_block(failed, INPUT_COLS),
    "valid_input_stats": describe_block(good, INPUT_COLS),
    "all_input_stats": describe_block(inp, INPUT_COLS),
}

# quartile-of-input failure rates (does failure cluster in a corner?)
q_fail = {}
for c in INPUT_COLS:
    tmp = merged.dropna(subset=[c]).copy()
    tmp["bad"] = tmp["status"].isin(["missing", "unusable"]).astype(int)
    try:
        tmp["q"] = pd.qcut(tmp[c], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
        q_fail[c] = {str(k): {"n": int(v["bad"].size), "n_failed": int(v["bad"].sum()),
                              "fail_rate": float(v["bad"].mean())}
                     for k, v in tmp.groupby("q", observed=True)}
    except Exception as e:
        q_fail[c] = {"error": str(e)}

# point-biserial correlation input vs failure flag
merged["_bad"] = merged["status"].isin(["missing", "unusable"]).astype(float)
try:
    from scipy import stats as _st
except Exception:
    _st = None
fail_corr = {}
for c in INPUT_COLS:
    sub = merged.dropna(subset=[c])
    e = {"pearson_vs_failure_flag": float(np.corrcoef(sub[c].astype(float), sub["_bad"])[0, 1])}
    if _st is not None:
        a = sub.loc[sub["_bad"] == 1, c].astype(float)
        b = sub.loc[sub["_bad"] == 0, c].astype(float)
        e["mean_failed"] = float(a.mean())
        e["mean_valid"] = float(b.mean())
        e["welch_t_p"] = float(_st.ttest_ind(a, b, equal_var=False).pvalue)
        e["mannwhitneyu_p"] = float(_st.mannwhitneyu(a, b).pvalue)
        ct = pd.crosstab(pd.qcut(sub[c], 4, labels=False), sub["_bad"])
        e["chi2_quartile_p"] = float(_st.chi2_contingency(ct).pvalue)
    fail_corr[c] = e

# --------------------------------------------------------------------------
# 4. Statistics over valid samples + input/output correlations
# --------------------------------------------------------------------------
METRIC_COLS = ["peak_RF2", "U2_at_peak", "k0_N_per_mm", "damage_final",
               "damage_U2_at_half_max", "damage_at_U2_10mm",
               "U2_max_abs", "t_final", "n_points"]

# ---- input design: collinearity / effective dimensionality -----------------
inp_corr_p = inp[INPUT_COLS].corr(method="pearson")
inp_corr_s = inp[INPUT_COLS].corr(method="spearman")
_lf, _le = np.log(inp["Fcm_MPa"].to_numpy(float)), np.log(inp["E_MPa"].to_numpy(float))
_b, _a = np.polyfit(_lf, _le, 1)
_pred = np.exp(_a) * inp["Fcm_MPa"].to_numpy(float) ** _b
input_design = {
    "pearson_matrix": {c: {d: float(inp_corr_p.loc[c, d]) for d in INPUT_COLS} for c in INPUT_COLS},
    "spearman_matrix": {c: {d: float(inp_corr_s.loc[c, d]) for d in INPUT_COLS} for c in INPUT_COLS},
    "E_vs_Fcm_power_law": {"coef": float(np.exp(_a)), "exponent": float(_b),
                           "max_rel_error": float(np.max(np.abs(_pred - inp["E_MPa"]) / inp["E_MPa"]))},
    "note": ("E_MPa is a deterministic power law of Fcm_MPa (Spearman = 1.0, max rel. error ~1e-15); "
             "the design has 3 independent inputs, not 4. Do NOT use Fcm and E as separate "
             "surrogate features - they are perfectly rank-collinear."),
    "unique_seeds": [int(x) for x in sorted(inp["seed"].unique())],
}

stats_valid = describe_block(good, METRIC_COLS)
stats_usable = describe_block(usable, METRIC_COLS)


def corr_table(df, xs, ys):
    out = {}
    for y in ys:
        out[y] = {}
        for x in xs:
            sub = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 3:
                out[y][x] = None
                continue
            out[y][x] = {
                "n": int(len(sub)),
                "pearson": float(sub[x].corr(sub[y], method="pearson")),
                "spearman": float(sub[x].corr(sub[y], method="spearman")),
            }
    return out


corr_valid = corr_table(good, INPUT_COLS,
                        ["peak_RF2", "U2_at_peak", "k0_N_per_mm", "damage_final",
                         "damage_U2_at_half_max", "damage_at_U2_10mm"])
corr_usable = corr_table(usable, INPUT_COLS, ["peak_RF2"])

# --------------------------------------------------------------------------
# 5. Curve alignment
# --------------------------------------------------------------------------
valid_ids = good["sample_id"].astype(int).tolist()
usable_ids = usable["sample_id"].astype(int).tolist()

npts = {sid: len(ld_g[sid]) for sid in ld_jobs}
npts_valid = [npts[s] for s in valid_ids if s in npts]

# identical time grids?
ref = None
same_time_grid = True
same_len = True
for s in valid_ids:
    t = np.sort(ld_g[s]["time"].to_numpy(float))
    if ref is None:
        ref = t
        continue
    if len(t) != len(ref):
        same_len = False
        same_time_grid = False
        break
    if not np.allclose(t, ref, atol=1e-9):
        same_time_grid = False
        break

u_max_valid = np.array([float(np.abs(ld_g[s]["U2"].to_numpy(float)).max()) for s in valid_ids])
u_max_usable = np.array([float(np.abs(ld_g[s]["U2"].to_numpy(float)).max()) for s in usable_ids])

grid_coverage = {}
for thr in [1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 12.0, 15.0, 18.0, 19.0, 19.5, 20.0]:
    grid_coverage["U2_max_%.1f" % thr] = {
        "n_valid_covering": int((u_max_valid >= thr - 1e-9).sum()),
        "n_usable_covering": int((u_max_usable >= thr - 1e-9).sum()),
    }

# how many curves survive interpolation onto a common U2 grid at some candidate maxima
def survivors(u_arr, thr):
    return int((u_arr >= thr - 1e-9).sum())


rec_thr_valid = float(np.min(u_max_valid)) if len(u_max_valid) else np.nan
rec_thr_p05 = float(np.percentile(u_max_usable, 5)) if len(u_max_usable) else np.nan

alignment = {
    "n_points_per_job_min": int(min(npts.values())) if npts else None,
    "n_points_per_job_max": int(max(npts.values())) if npts else None,
    "n_points_per_job_median": float(np.median(list(npts.values()))) if npts else None,
    "n_unique_point_counts_all_jobs": int(len(set(npts.values()))),
    "n_unique_point_counts_valid_jobs": int(len(set(npts_valid))),
    "identical_time_grid_across_valid": bool(same_time_grid),
    "identical_length_across_valid": bool(same_len),
    "U2_is_20x_time": None,   # filled below
    "U2_max_valid_min": float(u_max_valid.min()) if len(u_max_valid) else None,
    "U2_max_valid_max": float(u_max_valid.max()) if len(u_max_valid) else None,
    "U2_max_usable_min": float(u_max_usable.min()) if len(u_max_usable) else None,
    "U2_max_usable_p05": rec_thr_p05,
    "grid_coverage": grid_coverage,
    "recommended_common_U2_grid": None,
}

# check U2 = 20*time relation (displacement-controlled)
rel = []
for s in valid_ids[:50]:
    g = ld_g[s]
    t = g["time"].to_numpy(float)
    u = np.abs(g["U2"].to_numpy(float))
    m = t > 0
    if m.sum():
        rel.append(float(np.nanmax(np.abs(u[m] / t[m] - 20.0))))
alignment["U2_is_20x_time"] = bool(len(rel) > 0 and max(rel) < 1e-3)
alignment["U2_over_time_max_dev_from_20"] = float(max(rel)) if rel else None

# recommended grid: 0 .. min U2_max over valid, 201 points -- and actually build it
if len(u_max_valid):
    umax_rec = float(np.floor(u_max_valid.min() * 100) / 100)
    Ugrid = np.linspace(0.0, umax_rec, 201)
    mat, mat_ids = [], []
    for s in usable_ids:
        uu, ff = curves[s]
        if uu[0] <= 1e-9 and uu[-1] >= umax_rec - 1e-9:
            mat.append(np.interp(Ugrid, uu, ff))
            mat_ids.append(s)
    mat = np.asarray(mat)
    alignment["recommended_common_U2_grid"] = {
        "u_min": 0.0, "u_max": umax_rec, "n_points": 201,
        "n_valid_curves_surviving": survivors(u_max_valid, umax_rec),
        "n_usable_curves_surviving": survivors(u_max_usable, umax_rec),
        "interpolation_executed_shape": list(mat.shape),
        "interpolation_n_nan": int(np.isnan(mat).sum()),
        "RF2_at_u_max": {"mean": float(mat[:, -1].mean()),
                         "std": float(mat[:, -1].std(ddof=1)),
                         "min": float(mat[:, -1].min()),
                         "max": float(mat[:, -1].max())},
    }
    np.save(os.path.join(OUTDIR, "RF2_on_common_U2_grid.npy"), mat)
    np.save(os.path.join(OUTDIR, "common_U2_grid.npy"), Ugrid)
    pd.Series(mat_ids, name="sample_id").to_csv(
        os.path.join(OUTDIR, "common_grid_sample_ids.csv"), index=False)

# --------------------------------------------------------------------------
# 6. Write outputs
# --------------------------------------------------------------------------
OUT_COLS = (["sample_id", "job", "status", "reason"] + INPUT_COLS + ["seed"] +
            ["present_ld", "present_damage", "n_points", "n_points_finite", "t_final",
             "t_frac_of_target", "reached_full_step", "U2_max_abs", "U2_final",
             "peak_RF2", "U2_at_peak", "n_points_after_peak", "RF2_final",
             "postpeak_drop_frac_max", "postpeak_drop_frac_final", "softening_captured",
             "k0_N_per_mm", "n_pts_for_k0", "damage_final", "damage_max",
             "damage_U2_at_half_max", "damage_at_U2_10mm", "damage_monotone",
             "damage_n_points", "damage_t_final", "n_nan", "n_dup_time",
             "n_dup_rows_exact", "time_monotone", "U2_monotone", "damage_n_nan"])
OUT_COLS = [c for c in OUT_COLS if c in merged.columns]
merged[OUT_COLS].to_csv(os.path.join(OUTDIR, "sample_validity.csv"), index=False)

failed_list = []
for _, r in failed.iterrows():
    failed_list.append({
        "sample_id": int(r["sample_id"]),
        "status": r["status"],
        "reason": r["reason"],
        "Fcm_MPa": None if pd.isna(r["Fcm_MPa"]) else float(r["Fcm_MPa"]),
        "c_nom_bottom_mm": None if pd.isna(r["c_nom_bottom_mm"]) else float(r["c_nom_bottom_mm"]),
        "c_nom_top_mm": None if pd.isna(r["c_nom_top_mm"]) else float(r["c_nom_top_mm"]),
        "E_MPa": None if pd.isna(r["E_MPa"]) else float(r["E_MPa"]),
        "t_final": None if ("t_final" not in r or pd.isna(r["t_final"])) else float(r["t_final"]),
        "peak_RF2": None if ("peak_RF2" not in r or pd.isna(r["peak_RF2"])) else float(r["peak_RF2"]),
    })

partial_list = []
for _, r in partial.iterrows():
    partial_list.append({
        "sample_id": int(r["sample_id"]), "t_final": float(r["t_final"]),
        "peak_RF2": float(r["peak_RF2"]), "U2_at_peak": float(r["U2_at_peak"]),
        "postpeak_drop_frac_max": float(r["postpeak_drop_frac_max"]),
        "n_points_after_peak": int(r["n_points_after_peak"]),
        "Fcm_MPa": float(r["Fcm_MPa"]), "E_MPa": float(r["E_MPa"]),
    })

summary = {
    "generated_by": "audit_script.py",
    "criteria": {
        "target_step_time": TARGET_TIME,
        "time_fraction_for_full_step": TIME_FRAC_OK,
        "min_points": MIN_POINTS,
        "postpeak_drop_fraction": POSTPEAK_DROP,
        "postpeak_min_points": POSTPEAK_MIN_PTS,
        "initial_stiffness_window_frac_of_U2_at_peak": STIFF_FRAC,
        "definition_valid": "present, no NaN, >=MIN_POINTS finite pts, monotone time, t_final >= 0.95*1.0",
        "definition_partial": "present, clean, but t_final < 0.95 AND peak captured with >=10% post-peak drop over >=5 pts",
        "definition_unusable": "present but NaN / too few pts / non-monotone time / early stop without softening",
        "definition_missing": "sample_id absent from load_displacement_full.csv",
    },
    "inventory": inventory,
    "input_design": input_design,
    "status_counts": status_counts,
    "data_quality_flags": {
        "n_jobs_with_duplicate_time_stamps": int((diag.get("n_dup_time", pd.Series(dtype=float)) > 0).sum()),
        "n_duplicate_time_rows_total": int(diag.get("n_dup_time", pd.Series(dtype=float)).fillna(0).sum()),
        "n_exact_duplicate_rows_total": int(diag.get("n_dup_rows_exact", pd.Series(dtype=float)).fillna(0).sum()),
        "n_jobs_nonmonotone_time_in_file_order": int((diag["time_monotone"] == False).sum()),
        "n_jobs_nonmonotone_U2": int((diag["U2_monotone"] == False).sum()),
        "n_jobs_with_NaN": int((diag.get("n_nan", pd.Series(dtype=float)).fillna(0) > 0).sum()),
        "damage_final_is_constant": bool(good["damage_final"].nunique() == 1),
        "damage_final_constant_value": float(good["damage_final"].iloc[0]) if len(good) else None,
        "damage_final_note": ("DAMAGEC_max saturates at the CDP compression-damage table cap "
                              "(0.947) in every completed run -> zero variance, useless as a "
                              "surrogate target. Use damage_U2_at_half_max or damage_at_U2_10mm."),
    },
    "failed_jobs": failed_list,
    "partial_jobs": partial_list,
    "failure_clustering": {
        "input_stats_by_group": failure_cluster,
        "failure_rate_by_input_quartile": q_fail,
        "point_biserial_input_vs_failure": fail_corr,
    },
    "statistics_valid": stats_valid,
    "statistics_valid_plus_partial": stats_usable,
    "correlations_valid": corr_valid,
    "correlations_valid_plus_partial_peak_only": corr_usable,
    "curve_alignment": alignment,
}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float):
        return None if (np.isnan(o) or np.isinf(o)) else o
    return o


with open(os.path.join(OUTDIR, "audit_summary.json"), "w") as fh:
    json.dump(_clean(summary), fh, indent=2)

shutil.copyfile(os.path.abspath(__file__), os.path.join(OUTDIR, "audit_script.py"))

print(json.dumps(_clean(summary), indent=2))
