"""PHASE 3 — one JSON per Results subsection (§3.1–§3.7), from the caches of
`foot_cache/` and `thorax_cache/`. Common rules (§2.7.3, §2.7.4, §2.9, §2.10, §2.12): rf9; training with 4000
vertices per training case (seed from the identifier); evaluation on ALL vertices of the held-out case;
metrics per case; summary by median and IQR; paired two-sided Wilcoxon; Benjamini–Hochberg per family.
Usage: python src/reliability/a4_analyses.py [--only 31,32,...]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reliability import a4_config as C
from reliability import a4_statistics as E
from reliability import a4_registration as REG


def load(folder):
    D = {}
    for a in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        z = np.load(a, allow_pickle=False)
        d = {k: z[k] for k in z.files if k != "meta"}
        d["meta"] = json.loads(str(z["meta"]))
        D[os.path.basename(a)[:-4]] = d
    return D


def write(name, obj, caches):
    obj["_traceability"] = {
        "caches": caches,
        "environment": C.environment_record(with_torch=False),
        "protocol": "rf9; 4000 vertices per training case; evaluation on the full surface; median and IQR; two-sided Wilcoxon; BH per family",
    }
    p = os.path.join(C.A4_RESULTS, name)
    json.dump(obj, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("saved", p, flush=True)


_LABS = {}


def labels(h, V, key_V):
    """k-means (Eq. 36–37) a single time per (case, surface)."""
    if (h, key_V) not in _LABS:
        _LABS[(h, key_V)] = E.regions(V)
    return _LABS[(h, key_V)]


_SELECTION = {}  # label of the full model -> (params per fold, memo key) (the ablation reuses per fold)


def _data_digest(F, e, ids):
    import hashlib

    dig = hashlib.sha1()
    for h in ids:
        dig.update(np.ascontiguousarray(F[h], dtype=np.float32).tobytes())
        dig.update(np.ascontiguousarray(e[h], dtype=np.float32).tobytes())
    return dig.hexdigest()


def _rf_design(label, reuse_from, digest_fixed_params):
    """Description of the forest procedure that enters the memo key."""
    if C.RF_SELECTION != "nested":
        return ["fixed", C.RF_PARAMS]
    from reliability import a4_optimize_rf as O

    return [
        "nested",
        C.RF_SEARCH,
        C.RF_SEARCH_FOLDS,
        C.RF_SEARCH_N_VERTICES,
        None if reuse_from else O.n_trials_for(label),
        C.RF_SEARCH_STARTUP,
        C.RF_SEARCH_SAMPLER,
        reuse_from,
        digest_fixed_params,
    ]


def memo_key(F, e, columns, label, reuse_from=None, digest_fixed_params=None):
    """LOFO memo key: cases, columns, forest design, n_training, constant, code hash and digest of the
 CONTENT of F and e."""
    import hashlib

    ids = sorted(F)
    return hashlib.sha1(
        json.dumps(
            [
                ids,
                columns,
                _rf_design(label, reuse_from, digest_fixed_params),
                C.N_TRAIN_PER_CASE,
                C.CONSTANT_OVER,
                C.code_hash(),
                _data_digest(F, e, ids),
            ],
            default=str,
        ).encode()
    ).hexdigest()[:12]


def _selected_params(base_label, expected_key):
    """Per-fold configurations of the full model (process memory or rf_selection_<label>_<sampler>.json). The JSON
 is accepted ONLY if the memo key recorded in it is the expected key of the full model in the current design."""
    if base_label in _SELECTION and _SELECTION[base_label][1] == expected_key:
        return _SELECTION[base_label][0]
    p = os.path.join(
        C.A4_RESULTS, "rf_selection_%s_%s.json" % (base_label, C.RF_SEARCH_SAMPLER)
    )
    if not os.path.exists(p):
        raise RuntimeError(
            "nested selection of the full model %r not found (%s): run r32 before r35"
            % (base_label, p)
        )
    j = json.load(open(p, encoding="utf-8"))
    if j.get("key") != expected_key:
        raise RuntimeError(
            "rf_selection of %r has key %s, expected %s: design/data/code changed; redo r32"
            % (base_label, j.get("key"), expected_key)
        )
    _SELECTION[base_label] = (j["params_per_case"], expected_key)
    return j["params_per_case"]


def lofo_memo(F, e, columns=None, label="rf9", reuse_from=None):
    """LOFO with on-disk memoization (key: see memo_key). With
 C.RF_SELECTION == "nested", the per-fold selection is done by a4_optimize_rf and written to
 results/rf_selection_<label>_<sampler>.json (memo key, design, per-fold configuration, all attempts,
 per-fold details: seed, winning attempt, leaf as a count, leaves per tree, outer rho); with `reuse_from`
 (ablation, §2.10), each fold uses the configuration selected for the full model `reuse_from`, validated by the
 key."""
    import hashlib

    nested = C.RF_SELECTION == "nested"
    fixed_params, dig_pf = None, None
    if nested and reuse_from:
        base_key = memo_key(F, e, None, reuse_from)
        fixed_params = _selected_params(reuse_from, base_key)
        dig_pf = hashlib.sha1(
            json.dumps(fixed_params, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
    key = memo_key(F, e, columns, label, reuse_from, dig_pf)
    ids = sorted(F)
    p = os.path.join(C.A4_RESULTS, "_lofo_%s_%s.npz" % (label, key))
    if os.path.exists(p):
        z = np.load(p, allow_pickle=False)
        return {h: (z["pred_" + h], float(z["const_" + h])) for h in ids}
    if not nested:
        out = E.lofo(F, e, columns=columns)
    else:
        from reliability import a4_optimize_rf as O

        n_trials = O.n_trials_for(label)
        print(
            "nested selection of the forest: %s (%d attempts/fold%s)"
            % (label, n_trials, ", reuse %s" % reuse_from if reuse_from else ""),
            flush=True,
        )
        out, params, attempts, det = O.nested_lofo(
            F, e, columns=columns, n_trials=n_trials, fixed_params=fixed_params
        )
        if not reuse_from:
            _SELECTION[label] = (params, key)
            json.dump(
                {
                    "description": O.__doc__,
                    "label": label,
                    "key": key,
                    "columns": columns,
                    "design": _rf_design(label, None, None),
                    "params_per_case": params,
                    "details_per_case": det,
                    "attempts_per_case": attempts,
                    "environment": C.environment_record(with_torch=False),
                },
                open(
                    os.path.join(
                        C.A4_RESULTS,
                        "rf_selection_%s_%s.json" % (label, C.RF_SEARCH_SAMPLER),
                    ),
                    "w",
                    encoding="utf-8",
                ),
                indent=1,
                ensure_ascii=False,
                default=str,
            )
    np.savez_compressed(
        p,
        **{"pred_" + h: out[h][0] for h in ids},
        **{"const_" + h: np.array(out[h][1]) for h in ids},
    )
    return out


def per_case_full(pred, D, key_e="e", key_V="V"):
    """Metrics of §2.9 for each case: {id: dict}."""
    out = {}
    for h, (p, const) in pred.items():
        e = D[h][key_e].astype(float)
        V = D[h][key_V].astype(float)
        lab = labels(h, V, key_V)
        m = {
            "rho_vertex": E.spearman(p, e),
            "rho_region": E.spearman(
                E.regional_means(p, lab), E.regional_means(e, lab)
            ),
            "_reg_pred": E.regional_means(p, lab).tolist(),
            "_reg_obs": E.regional_means(e, lab).tolist(),
            "n_vertices": len(e),
        }
        m.update(E.localization(p, e))
        m.update(E.calibration(p, e, const))
        out[h] = m
    return out


def aggregate(per_case, keys):
    r = {k: E.summary([v[k] for v in per_case.values()]) for k in keys}
    pp = np.concatenate([v["_reg_pred"] for v in per_case.values()])
    po = np.concatenate([v["_reg_obs"] for v in per_case.values()])
    r["pooled_region_rho"] = E.spearman(pp, po)
    r["n_pooled_regions"] = len(pp)
    return r


CH = [
    "rho_vertex",
    "rho_region",
    "auroc_decile",
    "precision_10",
    "flagged_error_mm",
    "overall_error_mm",
    "residual_error_mm",
    "mae_mm",
    "constant_mae_mm",
    "slope",
    "intercept_mm",
]


def aggregated_deciles(per_case):
    dec = []
    for d in range(C.N_CALIBRATION_DECILES):
        dec.append(
            {
                "decile": d + 1,
                "predicted_mm": E.summary(
                    [v["deciles"][d]["predicted_mm"] for v in per_case.values()]
                ),
                "observed_mm": E.summary(
                    [v["deciles"][d]["observed_mm"] for v in per_case.values()]
                ),
            }
        )
    return dec


def clean(per_case):
    return {
        h: {k: v for k, v in m.items() if not k.startswith("_")}
        for h, m in per_case.items()
    }


# ----------------------------------------------------------------------------- 3.1
def r31(D):
    sr = {h: D[h]["e"].astype(float) for h in D}
    tri = {h: D[h]["e_tri"].astype(float) for h in D}

    def stats(e):  # §2.12: median and IQR
        return {
            "median_mm": float(np.median(e)),
            "iqr_mm": [float(np.percentile(e, 25)), float(np.percentile(e, 75))],
            "n_vertices": len(e),
        }

    pc = {h: {"sr": stats(sr[h]), "trilinear": stats(tri[h])} for h in D}
    med_sr = {h: pc[h]["sr"]["median_mm"] for h in D}
    med_tri = {h: pc[h]["trilinear"]["median_mm"] for h in D}
    return {
        "section": "3.1 Geometric Reconstruction Error (Eq. 5)",
        "n_cases": len(D),
        "sr": {"median_mm": E.summary([pc[h]["sr"]["median_mm"] for h in D])},
        "trilinear": {
            "median_mm": E.summary([pc[h]["trilinear"]["median_mm"] for h in D])
        },
        "sr_vs_trilinear_median_per_case": E.paired_comparison(med_sr, med_tri),
        "per_case": pc,
    }


# ----------------------------------------------------------------------------- 3.2 / 3.3
def r32_33(D):
    F = {h: D[h]["F"] for h in D}
    e = {h: D[h]["e"] for h in D}
    pred_sr = lofo_memo(F, e, label="rf9_sr")  # Eq. 20
    pc_sr = per_case_full(pred_sr, D)
    Ft = {h: D[h]["F_tri"] for h in D}
    et = {h: D[h]["e_tri"] for h in D}
    pred_tri = lofo_memo(Ft, et, label="rf9_tri")
    pc_tri = per_case_full(pred_tri, D, "e_tri", "V_tri")
    comp = E.paired_comparison(
        {h: pc_sr[h]["rho_vertex"] for h in D},
        {h: pc_tri[h]["rho_vertex"] for h in D},
    )
    r32 = {
        "section": "3.2 Vertex- and Region-Level Reliability Prediction (Eq. 35–39)",
        "n_cases": len(D),
        "sr": {k: v for k, v in aggregate(pc_sr, ["rho_vertex", "rho_region"]).items()},
        "trilinear": {
            k: v for k, v in aggregate(pc_tri, ["rho_vertex", "rho_region"]).items()
        },
        "sr_vs_trilinear_rho_vertex": comp,
        "per_case_sr": {
            h: {k: pc_sr[h][k] for k in ("rho_vertex", "rho_region", "n_vertices")}
            for h in D
        },
        "per_case_trilinear": {
            h: {k: pc_tri[h][k] for k in ("rho_vertex", "rho_region", "n_vertices")}
            for h in D
        },
    }
    mae_c = E.paired_comparison(
        {h: pc_sr[h]["constant_mae_mm"] for h in D}, {h: pc_sr[h]["mae_mm"] for h in D}
    )
    # besides the per-case summary, the line and deciles POOLED over all evaluated vertices (descriptive, §2.12)
    p_all = np.concatenate([pred_sr[h][0] for h in sorted(D)])
    e_all = np.concatenate([D[h]["e"].astype(float) for h in sorted(D)])
    const_all = float(np.mean([pred_sr[h][1] for h in sorted(D)]))
    pooled_cal = E.calibration(p_all, e_all, const_all)
    pooled_cal = {k: v for k, v in pooled_cal.items() if k != "constant_mae_mm"}
    r33 = {
        "section": "3.3 High-Error Localization and Physical Calibration (Eq. 40–46)",
        "n_cases": len(D),
        "sr_pooled_calibration_descriptive": pooled_cal,
        "sr": aggregate(
            pc_sr,
            [
                "auroc_decile",
                "precision_10",
                "flagged_error_mm",
                "overall_error_mm",
                "residual_error_mm",
                "mae_mm",
                "constant_mae_mm",
                "slope",
                "intercept_mm",
            ],
        ),
        "sr_deciles": aggregated_deciles(pc_sr),
        "constant_mae_minus_model_mae": mae_c,
        "trilinear": aggregate(
            pc_tri,
            [
                "auroc_decile",
                "precision_10",
                "mae_mm",
                "constant_mae_mm",
                "slope",
                "intercept_mm",
            ],
        ),
        "per_case_sr": clean(pc_sr),
    }
    return r32, r33, pred_sr, pc_sr


# ----------------------------------------------------------------------------- 3.4
def r34(D, pred_sr, pc_sr, keys, section, family_name):
    res = {}
    res["field_rf9"] = aggregate(
        pc_sr,
        [
            "rho_vertex",
            "rho_region",
            "auroc_decile",
            "precision_10",
            "flagged_error_mm",
        ],
    )
    pc_u = {}
    for u in keys:
        pred_u = {h: (D[h][u].astype(float), pred_sr[h][1]) for h in D}
        pc_u[u] = per_case_full(pred_u, D)
        res[u] = aggregate(
            pc_u[u],
            [
                "rho_vertex",
                "rho_region",
                "auroc_decile",
                "precision_10",
                "flagged_error_mm",
            ],
        )
    comps = {}
    for metric in ("rho_vertex", "auroc_decile"):
        fam = {}
        for u in keys:
            fam[u] = E.paired_comparison(
                {h: pc_sr[h][metric] for h in D}, {h: pc_u[u][h][metric] for h in D}
            )
        bh = E.benjamini_hochberg({u: fam[u]["p_wilcoxon"] for u in keys})
        for u in keys:
            fam[u]["p_bh"] = bh[u]
            fam[u]["significant_bh_0_05"] = bool(bh[u] < C.ALPHA)
        comps["field_minus_" + metric] = {
            "family": family_name + " (" + metric + ", m=%d)" % len(keys),
            "comparisons": fam,
        }
    return {
        "section": section,
        "n_cases": len(D),
        "results": res,
        "paired_comparisons": comps,
        "per_case": {
            u: {
                h: {
                    k: pc_u[u][h][k]
                    for k in ("rho_vertex", "rho_region", "auroc_decile")
                }
                for h in D
            }
            for u in keys
        },
    }


# ----------------------------------------------------------------------------- 3.5
def r35(D, pc_sr):
    F = {h: D[h]["F"] for h in D}
    e = {h: D[h]["e"] for h in D}
    rho_full = {h: pc_sr[h]["rho_vertex"] for h in D}
    out = {
        "section": "3.5 Feature Contribution and Ablation (Eq. 47–50)",
        "n_cases": len(D),
        "full": aggregate(
            pc_sr, ["rho_vertex", "auroc_decile", "precision_10", "mae_mm"]
        ),
        "isolated": {},
        "without_one": {},
    }
    reuse = "rf9_sr" if C.RF_ABLATION_REUSE_SELECTION else None  # §2.10
    for j, name in enumerate(C.FEATURE_NAMES):
        pc = per_case_full(
            lofo_memo(F, e, columns=[j], label="iso%d" % j, reuse_from=reuse), D
        )  # Eq. 47
        out["isolated"][name] = aggregate(
            pc, ["rho_vertex", "auroc_decile", "precision_10", "mae_mm"]
        )
    p_fam = {}
    for j, name in enumerate(C.FEATURE_NAMES):
        cols = [c for c in range(C.N_FEATURES) if c != j]
        pc = per_case_full(
            lofo_memo(F, e, columns=cols, label="seed%d" % j, reuse_from=reuse), D
        )  # Eq. 48–49
        delta = {h: rho_full[h] - pc[h]["rho_vertex"] for h in D}  # Eq. 50
        comp = E.paired_comparison(rho_full, {h: pc[h]["rho_vertex"] for h in D})
        out["without_one"][name] = {
            "summary": aggregate(
                pc, ["rho_vertex", "auroc_decile", "precision_10", "mae_mm"]
            ),
            "delta_rho": E.summary(list(delta.values())),
            "comparison_full_vs_without": comp,
        }
        p_fam[name] = comp["p_wilcoxon"]
    bh = E.benjamini_hochberg(p_fam)
    for name in C.FEATURE_NAMES:
        out["without_one"][name]["comparison_full_vs_without"]["p_bh"] = bh[name]
    out["bh_family"] = "leave-one-feature-out, m=9 (Δ_ij, Eq. 50)"
    return out


# ----------------------------------------------------------------------------- 3.6
def thorax_for_r36(T):
    """The RPLHR-CT validation was used in the SR selection; section 3.6 uses only the 100 test cases."""
    T = {h: d for h, d in T.items() if d["meta"].get("partition") == C.R36_PARTITION}
    if len(T) != C.N_THORAX_TEST:
        raise RuntimeError(
            "r36: %d cases of partition %s; expected %d"
            % (len(T), C.R36_PARTITION, C.N_THORAX_TEST)
        )
    return T


def r36(T):
    F = {h: T[h]["F"] for h in T}
    e = {h: T[h]["e"] for h in T}
    pc = per_case_full(lofo_memo(F, e, label="thorax_sr"), T)
    Ft = {h: T[h]["F_tri"] for h in T}
    et = {h: T[h]["e_tri"] for h in T}
    pct = per_case_full(lofo_memo(Ft, et, label="thorax_tri"), T, "e_tri", "V_tri")
    cons = os.path.join(C.A4_RESULTS, "rplhr_pairs_consistency.json")
    return {
        "section": "3.6 Evaluation on Real Paired Thoracic CT (§2.2.2, LOO on the 100 test cases)",
        "n_cases": len(T),
        "partition": C.R36_PARTITION,
        "sr": aggregate(pc, CH),
        "sr_deciles": aggregated_deciles(pc),
        "trilinear": aggregate(pct, CH),
        "sr_vs_trilinear_rho_vertex": E.paired_comparison(
            {h: pc[h]["rho_vertex"] for h in T}, {h: pct[h]["rho_vertex"] for h in T}
        ),
        "pair_consistency": (
            json.load(open(cons, encoding="utf-8"))["consistency_2_2_2"]
            | {
                "per_case": None,
                "note": "check of the pairs on the 150 cases (§2.2.2), independent of the partition used here",
            }
        )
        if os.path.exists(cons)
        else None,
        "per_case_sr": clean(pc),
    }


# ----------------------------------------------------------------------------- 3.7
def r37(D, pred_sr):
    pc = {}
    t0 = time.time()
    for h in sorted(D):
        V = D[h]["V"].astype(float)
        q = D[h]["q"].astype(float)
        F = D[h]["F"]
        e = D[h]["e"].astype(float)
        res = REG.record_case(
            V,
            q,
            pred_sr[h][0],
            F[:, 5],
            e,
            seed=C.REG_SEED,
            labels=labels(h, V, "V"),
        )
        pc[h] = REG.case_summary(res)
    med = {s: {h: pc[h][s]["median_mm"] for h in pc} for s in C.REG_STRATEGIES}
    fam = {
        s: E.paired_comparison(med["field"], med[s])
        for s in C.REG_STRATEGIES
        if s != "field"
    }
    bh = E.benjamini_hochberg({s: fam[s]["p_wilcoxon"] for s in fam})
    for s in fam:
        fam[s]["p_bh"] = bh[s]
    red = {h: 100.0 * (1 - med["field"][h] / med["random"][h]) for h in pc}
    med_c, med_a = (
        float(np.median(list(med["field"].values()))),
        float(np.median(list(med["random"].values()))),
    )
    return {
        "section": "3.7 Reliability-Guided Registration (§2.11)",
        "n_cases": len(D),
        "K": C.REG_K,
        "trials": C.REG_TRIALS,
        "strategies": {s: E.summary(list(med[s].values())) for s in C.REG_STRATEGIES},
        # §2.11 "summarized within each case before cohort-level comparison" + Abstract "reduced
        # median target displacement" (ratio of cohort medians; the paper reports this value) -> PRIMARY = 100(1 - median_cohort(med_field)/median_cohort(med_random))
        "reduction_field_vs_random_pct_ratio_of_medians": 100.0
        * (1 - med_c / med_a),
        "reduction_field_vs_random_pct_ratio_of_medians_definition": "PRIMARY (Abstract): 100·(1 − cohort_median(median per case, field) / cohort_median(median per case, random))",
        "reduction_field_vs_random_pct_median_of_case_ratios": E.summary(
            list(red.values())
        ),
        "reduction_field_vs_random_pct_median_of_case_ratios_definition": "DESCRIPTIVE: 100·(1 − med_field/med_random) computed per case and summarized across cases",
        "paired_comparisons_field_minus_others": {
            "family": "field vs 4 strategies (m=4)",
            "comparisons": fam,
        },
        "per_case": pc,
        "seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="ex.: 31,32,33")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else set()
    C.ensure_folders()
    foot_requested = not only or bool(only & {"31", "32", "33", "341", "342", "35", "37"})
    D = load(C.A4_CACHE_FOOT) if foot_requested else {}
    print("feet in the cache:", len(D), flush=True)
    foot_caches = sorted(D)
    if D:
        if not only or "31" in only:
            write("r31_reconstruction_error.json", r31(D), foot_caches)
        if only and not (only & {"32", "33", "341", "342", "35", "37"}):
            D = {}
    if D:
        r32, r33, pred_sr, pc_sr = r32_33(D)
        if not only or "32" in only:
            write("r32_association.json", r32, foot_caches)
        if not only or "33" in only:
            write("r33_localization_calibration.json", r33, foot_caches)
        if (not only or "341" in only) and all("u_ens" in D[h] for h in D):
            ch = ["u_ens", "u_mc", "u_dropens"] + (
                ["u_ds"] if all("u_ds" in D[h] for h in D) else []
            )
            write(
                "r341_intensity_uncertainty.json",
                r34(
                    D,
                    pred_sr,
                    pc_sr,
                    ch,
                    "3.4.1 Intensity-space uncertainty (Eq. 21–27, 30)",
                    "field vs intensity-space uncertainties",
                ),
                foot_caches,
            )
        if (not only or "342" in only) and all("u_geo_std" in D[h] for h in D):
            write(
                "r342_surface_uncertainty.json",
                r34(
                    D,
                    pred_sr,
                    pc_sr,
                    ["u_geo_std", "u_geo_abs", "u_geo_mc"],
                    "3.4.2 Surface-space uncertainty (Eq. 31–34)",
                    "field vs surface-space uncertainties",
                ),
                foot_caches,
            )
        if not only or "35" in only:
            write("r35_ablation.json", r35(D, pc_sr), foot_caches)
        if not only or "37" in only:
            write("r37_registration.json", r37(D, pred_sr), foot_caches)
    if not only or "36" in only:
        T = thorax_for_r36(load(C.A4_CACHE_THORAX))
        print("thoracic cases (partition %s):" % C.R36_PARTITION, len(T), flush=True)
        write("r36_thorax.json", r36(T), sorted(T))


if __name__ == "__main__":
    main()
