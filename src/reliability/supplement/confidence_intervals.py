"""Effect sizes with confidence intervals for the paired comparisons of the paper and bootstrap intervals for the
headline medians. The Wilcoxon p-value of a comparison in which every case has the same sign is bounded by 2/2^n
(7.1e-15 for n = 48), so it measures consistency of sign only; this script adds the Hodges-Lehmann estimate of the
paired difference with its distribution-free 95 % interval and a percentile bootstrap of the median. Reads the
result files of the run (per-case values). Outside the package hash (subfolder).

Output: results/rS_confidence_intervals.json
Usage: python src/reliability/supplement/confidence_intervals.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import norm, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from reliability import a4_config as C  # noqa: E402

N_BOOT, SEED, ALPHA = 20000, 0, 0.05


def J(name):
    return json.load(open(os.path.join(C.A4_RESULTS, name), encoding="utf-8"))


def hodges_lehmann(d):
    """HL estimate of the paired difference and its distribution-free 1 - ALPHA interval (Walsh averages)."""
    d = np.asarray(d, float)
    n = len(d)
    i, j = np.triu_indices(n)
    w = np.sort((d[i] + d[j]) / 2.0)
    m = n * (n + 1) / 2.0
    k = int(np.floor(m / 2.0 - norm.ppf(1 - ALPHA / 2) * np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)))
    return float(np.median(w)), [float(w[k - 1]), float(w[int(m - k)])]


def boot_median(v, rng):
    v = np.asarray(v, float)
    b = np.median(v[rng.integers(0, len(v), (N_BOOT, len(v)))], axis=1)
    return [float(np.percentile(b, 100 * ALPHA / 2)), float(np.percentile(b, 100 * (1 - ALPHA / 2)))]


def paired(a: dict, b: dict, rng) -> dict:
    ids = sorted(set(a) & set(b))
    d = np.array([a[h] - b[h] for h in ids], float)
    hl, ci = hodges_lehmann(d)
    return {"n": len(d), "median_difference": float(np.median(d)), "bootstrap_ci_median": boot_median(d, rng),
            "hodges_lehmann": hl, "hodges_lehmann_ci": ci, "positives": int((d > 0).sum()), "p_wilcoxon": float(wilcoxon(d).pvalue)}


def main():
    rng = np.random.default_rng(SEED)
    r31, r32, r33 = J("r31_reconstruction_error.json"), J("r32_association.json"), J("r33_localization_calibration.json")
    r341, r342, r36, r37 = J("r341_intensity_uncertainty.json"), J("r342_surface_uncertainty.json"), J("r36_thorax.json"), J("r37_registration.json")
    rs = J("rS_median_baseline.json")
    pc33, pc32, pc32t, pc36, pc37 = r33["per_case_sr"], r32["per_case_sr"], r32["per_case_trilinear"], r36["per_case_sr"], r37["per_case"]
    field_v = {h: v["rho_vertex"] for h, v in pc32.items()}
    field_r = {h: v["rho_region"] for h, v in pc32.items()}
    comp = {
        "foot_mae_constant_mean_minus_field_mm": paired({h: v["constant_mae_mm"] for h, v in pc33.items()}, {h: v["mae_mm"] for h, v in pc33.items()}, rng),
        "foot_mae_constant_median_minus_field_mm": paired({h: v["mae_constant_median"] for h, v in rs["per_case"].items()}, {h: v["mae_model"] for h, v in rs["per_case"].items()}, rng),
        "foot_rho_vertex_sr_minus_trilinear": paired(field_v, {h: v["rho_vertex"] for h, v in pc32t.items()}, rng),
        "foot_rho_region_sr_minus_trilinear": paired(field_r, {h: v["rho_region"] for h, v in pc32t.items()}, rng),
        "foot_error_median_sr_minus_trilinear_mm": paired({h: v["sr"]["median_mm"] for h, v in r31["per_case"].items()}, {h: v["trilinear"]["median_mm"] for h, v in r31["per_case"].items()}, rng),
        "thorax_mae_constant_minus_field_mm": paired({h: v["constant_mae_mm"] for h, v in pc36.items()}, {h: v["mae_mm"] for h, v in pc36.items()}, rng),
    }
    for name, src in (("u_ens", r341), ("u_mc", r341), ("u_dropens", r341), ("u_ds", r341), ("u_geo_std", r342), ("u_geo_abs", r342), ("u_geo_mc", r342)):
        if name in src.get("per_case", {}):
            pcu = src["per_case"][name]
            comp["foot_rho_vertex_field_minus_%s" % name] = paired(field_v, {h: v["rho_vertex"] for h, v in pcu.items()}, rng)
            comp["foot_rho_region_field_minus_%s" % name] = paired(field_r, {h: v["rho_region"] for h, v in pcu.items()}, rng)
    for s in ("random", "d_shape", "oracle", "global"):
        comp["registration_field_minus_%s_mm" % s] = paired({h: v["field"]["median_mm"] for h, v in pc37.items()}, {h: v[s]["median_mm"] for h, v in pc37.items()}, rng)
    # Benjamini-Hochberg within the families of the paper (Section 2.12): intensity-space (m = 4), surface-space
    # (m = 3), registration (m = 4), at vertex and region level separately
    from reliability import a4_statistics as E
    fam = {
        "vertex_intensity": ["foot_rho_vertex_field_minus_%s" % n for n in ("u_ens", "u_mc", "u_dropens", "u_ds")],
        "region_intensity": ["foot_rho_region_field_minus_%s" % n for n in ("u_ens", "u_mc", "u_dropens", "u_ds")],
        "vertex_surface": ["foot_rho_vertex_field_minus_%s" % n for n in ("u_geo_std", "u_geo_abs", "u_geo_mc")],
        "region_surface": ["foot_rho_region_field_minus_%s" % n for n in ("u_geo_std", "u_geo_abs", "u_geo_mc")],
        "registration": ["registration_field_minus_%s_mm" % s for s in ("random", "d_shape", "oracle", "global")],
    }
    for name, keys in fam.items():
        keys = [k for k in keys if k in comp]
        bh = E.benjamini_hochberg({k: comp[k]["p_wilcoxon"] for k in keys})
        for k in keys:
            comp[k]["p_bh"] = bh[k]
            comp[k]["bh_family"] = "%s (m = %d)" % (name, len(keys))
    ratio = np.array([v["field"]["median_mm"] / v["oracle"]["median_mm"] for v in pc37.values()])
    oracle_ratio = {"n": len(ratio), "median": float(np.median(ratio)), "iqr": [float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))],
                    "feet_within_5_percent": int((ratio <= 1.05).sum()), "feet_more_than_50_percent_worse": int((ratio > 1.5).sum()), "max": float(ratio.max())}
    medians = {
        "foot_rho_vertex_field": list(field_v.values()), "foot_rho_region_field": list(field_r.values()),
        "foot_auroc_decile": [v["auroc_decile"] for v in pc33.values()], "foot_slope": [v["slope"] for v in pc33.values()],
        "foot_mae_field_mm": [v["mae_mm"] for v in pc33.values()],
        "thorax_rho_vertex_field": [v["rho_vertex"] for v in pc36.values()], "thorax_rho_region_field": [v["rho_region"] for v in pc36.values()],
        "thorax_auroc_decile": [v["auroc_decile"] for v in pc36.values()], "thorax_slope": [v["slope"] for v in pc36.values()],
        "registration_field_mm": [v["field"]["median_mm"] for v in pc37.values()], "registration_random_mm": [v["random"]["median_mm"] for v in pc37.values()],
        "registration_oracle_mm": [v["oracle"]["median_mm"] for v in pc37.values()], "registration_global_mm": [v["global"]["median_mm"] for v in pc37.values()],
    }
    out = {
        "description": "paired differences: median, percentile-bootstrap 95 %% CI of the median (%d resamples, seed %d), Hodges-Lehmann estimate with distribution-free 95 %% CI, positives, two-sided Wilcoxon; headline medians with bootstrap 95 %% CI" % (N_BOOT, SEED),
        "wilcoxon_floor_n48": 2.0 / 2**48,
        "paired": comp,
        "registration_field_over_oracle_ratio": oracle_ratio,
        "medians": {k: {"n": len(v), "median": float(np.median(v)), "bootstrap_ci": boot_median(v, rng)} for k, v in medians.items()},
        "environment": C.environment_record(with_torch=False),
    }
    p = os.path.join(C.A4_RESULTS, "rS_confidence_intervals.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for k, v in comp.items():
        print("%s: median %.4f [%.4f, %.4f] HL %.4f [%.4f, %.4f] pos %d/%d p %.2g" % (k, v["median_difference"], *v["bootstrap_ci_median"], v["hodges_lehmann"], *v["hodges_lehmann_ci"], v["positives"], v["n"], v["p_wilcoxon"]))
    for k, v in out["medians"].items():
        print("%s: %.3f [%.3f, %.3f]" % (k, v["median"], *v["bootstrap_ci"]))
    print("->", p)


if __name__ == "__main__":
    main()
