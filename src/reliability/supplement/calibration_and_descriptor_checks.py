"""Checks of the calibration and of the descriptor-error association used in the Discussion, from the caches and the
leave-one-case-out memos of a run (nothing is refitted). Outside the package hash (subfolder).

Per cohort (foot, thorax test cases):
- per-case calibration slope (OLS of e on the prediction, as in r33/r36) with and without the vertices whose error kept
  the EXACT_MAX_MM bound (Section 2.4), and the number of cases with slope above one in each case;
- between-case compression: per-case mean predicted and mean measured error (ranges, OLS slope across cases);
- Spearman correlation of each vertex-level descriptor with the measured error on super-resolved and trilinear
  surfaces (median across cases, paired comparison).

Output: results/rS_calibration_checks.json
Usage: python src/reliability/supplement/calibration_and_descriptor_checks.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from reliability import a4_config as C  # noqa: E402
from reliability import a4_statistics as E  # noqa: E402

DESCRIPTORS = ("nz_abs", "d_f", "angular_defect", "roughness", "nz_abs_x_d_f", "d_shape")  # columns 0-5 of F


def memo(label: str) -> dict:
    sel = json.load(open(os.path.join(C.A4_RESULTS, "rf_selection_%s_%s.json" % (label, C.RF_SEARCH_SAMPLER)), encoding="utf-8"))
    z = np.load(os.path.join(C.A4_RESULTS, "_lofo_%s_%s.npz" % (label, sel["key"])), allow_pickle=False)
    return {k[5:]: z[k].astype(float) for k in z.files if k.startswith("pred_")}


def slope(p, e):
    return float(np.polyfit(p, e, 1)[0])


def cohort(cache_dir: str, label_sr: str, label_tri: str) -> dict:
    pred, pred_t = memo(label_sr), memo(label_tri)
    sl, sl_near, n_far, mean_p, mean_e = {}, {}, {}, {}, {}
    rho = {d: {} for d in DESCRIPTORS}
    rho_t = {d: {} for d in DESCRIPTORS}
    for f in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        h = os.path.basename(f)[:-4]
        z = np.load(f, allow_pickle=False)
        e, et = z["e"].astype(float), z["e_tri"].astype(float)
        F, Ft = z["F"].astype(float), z["F_tri"].astype(float)
        p = pred[h]
        near = e <= C.EXACT_MAX_MM
        sl[h], sl_near[h], n_far[h] = slope(p, e), slope(p[near], e[near]), int((~near).sum())
        mean_p[h], mean_e[h] = float(p.mean()), float(e.mean())
        for j, d in enumerate(DESCRIPTORS):
            rho[d][h] = float(spearmanr(F[:, j], e)[0])
            rho_t[d][h] = float(spearmanr(Ft[:, j], et)[0])
        _ = pred_t[h]  # the trilinear memo must cover the same cases
    ids = sorted(sl)
    mp = np.array([mean_p[h] for h in ids])
    me = np.array([mean_e[h] for h in ids])
    return {
        "n": len(ids),
        "slope_all_vertices": E.summary(list(sl.values())),
        "slope_without_vertices_beyond_exact_max": E.summary(list(sl_near.values())),
        "cases_with_slope_above_one": {"all_vertices": int(sum(v > 1 for v in sl.values())), "without_beyond_exact_max": int(sum(v > 1 for v in sl_near.values()))},
        "vertices_beyond_exact_max": {"total": int(sum(n_far.values())), "cases": int(sum(v > 0 for v in n_far.values()))},
        "between_case_compression": {
            "mean_predicted_mm_range": [float(mp.min()), float(mp.max())],
            "mean_measured_mm_range": [float(me.min()), float(me.max())],
            "ols_slope_measured_on_predicted_across_cases": slope(mp, me),
            "spearman_across_cases": float(spearmanr(mp, me)[0]),
        },
        "descriptor_error_spearman": {
            d: {"sr": E.summary(list(rho[d].values())), "trilinear": E.summary(list(rho_t[d].values())), "sr_minus_trilinear": E.paired_comparison(rho[d], rho_t[d])}
            for d in DESCRIPTORS
        },
    }


def main():
    out = {
        "description": "calibration slope with/without the vertices kept at the EXACT_MAX_MM bound; between-case compression of the prediction; descriptor-error association per surface",
        "exact_max_mm": C.EXACT_MAX_MM,
        "foot": cohort(C.A4_CACHE_FOOT, "rf9_sr", "rf9_tri"),
        "thorax_test": cohort(C.A4_CACHE_THORAX, "thorax_sr", "thorax_tri"),
        "environment": C.environment_record(with_torch=False),
    }
    p = os.path.join(C.A4_RESULTS, "rS_calibration_checks.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for g in ("foot", "thorax_test"):
        d = out[g]
        print(g, "slope all %.3f (%.3f-%.3f) | without >%g mm %.3f (%.3f-%.3f) | cases >1: %d -> %d | far vertices %d in %d cases" % (
            d["slope_all_vertices"]["median"], *d["slope_all_vertices"]["iqr"], C.EXACT_MAX_MM,
            d["slope_without_vertices_beyond_exact_max"]["median"], *d["slope_without_vertices_beyond_exact_max"]["iqr"],
            d["cases_with_slope_above_one"]["all_vertices"], d["cases_with_slope_above_one"]["without_beyond_exact_max"],
            d["vertices_beyond_exact_max"]["total"], d["vertices_beyond_exact_max"]["cases"]))
        b = d["between_case_compression"]
        print("   mean predicted %.3f-%.3f vs measured %.3f-%.3f mm; across-case OLS %.2f" % (*b["mean_predicted_mm_range"], *b["mean_measured_mm_range"], b["ols_slope_measured_on_predicted_across_cases"]))
        for k, v in d["descriptor_error_spearman"].items():
            print("   rho(%s, e): SR %.3f tri %.3f (tri higher in %d, p %.2g)" % (k, v["sr"]["median"], v["trilinear"]["median"], v["sr_minus_trilinear"]["n"] - v["sr_minus_trilinear"]["difference"]["positives"], v["sr_minus_trilinear"]["p_wilcoxon"]))
    print("->", p)


if __name__ == "__main__":
    main()
