"""Supplementary analysis of the error measurement: sampling floor of Eq. 5, exact point-to-surface error,
tails of the error distribution, and vertices lying outside the CT volume.

Eq. 5 measures e(v) as the distance to the nearest of the 120,000 points sampled on the reference surface. The
sampled points are about 0.8 mm apart on a foot, so a vertex lying exactly on the reference surface is still
about 0.4 mm from the nearest sample. This script quantifies that floor and repeats the main association and
localization metrics against the exact point-to-triangle distance, using the leave-one-case-out predictions
already produced by a4_analyses.py (nothing is retrained):

 - per foot and surface (SR, trilinear): median, mean, P90 and P99 of e (Eq. 5) and of the exact distance;
   floor = distance from the reference's own vertices to the sampled points;
 - field vs e (Eq. 5) and vs exact e: vertex-level Spearman, AUROC of the highest-error decile; the same for the
   shape-disagreement descriptor alone;
 - calibration of the same predictions against the exact distance (slope, intercept, MAE, offset), since the
   predictions were fitted on the Eq. 5 scale;
 - vertices outside the CT volume (mesh artefacts): count per foot, and the pooled and per-foot calibration
   (Eq. 45) with and without them;
 - thorax: tails of e (Eq. 5) for SR and trilinear and artefact counts (no exact distance: the thorax cache does
   not store the closest point and the reference would have to be rebuilt).

For the SR surface the exact distance is ||v - q(v)|| with q from the cache (closest point on the reference,
computed by a4_proximity.nearest_exact for the registration experiment); for the trilinear surface the
reference surface is rebuilt from the thin-slice volume and the exact distance computed (CPU, ~25 s per foot);
vertices farther than EXACT_MAX_MM from the sampled points (mesh artefacts) keep their Eq. 5 value.

Usage: python src/reliability/supplement/exact_error.py [--cache-foot DIR] [--cache-thorax DIR] [--results DIR]
Output: <results>/rS_exact_error.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from reliability import a4_config as C  # noqa: E402
from reliability import a4_proximity as P  # noqa: E402
from reliability import a4_statistics as E  # noqa: E402
from reliability import a4_surface as S  # noqa: E402
from reliability.a4_run_foot import prepare_foot  # noqa: E402

DS = C.FEATURE_NAMES.index("d_shape")


def tails(e):
    return {
        "median_mm": float(np.median(e)),
        "mean_mm": float(e.mean()),
        "p90_mm": float(np.percentile(e, 90)),
        "p99_mm": float(np.percentile(e, 99)),
        "max_mm": float(e.max()),
        "fraction_above_2mm_pct": float(100.0 * (e > 2.0).mean()),
    }


EXACT_MAX_MM = 20.0  # vertices farther than this from the sampled points keep the Eq. 5 value (mesh artefacts)
LARGE_TRI_MM = 5.0  # triangles with a circumradius above this are tested by brute force against every vertex


def exact_bounded(V, e_eq5, ref):
    """Exact point-to-triangle distance to `ref` for the vertices with e_eq5 <= EXACT_MAX_MM (Eq. 5 value for the rest).

    Same result as a4_proximity.nearest_exact, with a search bounded by the sampled distance e_eq5 (an upper bound
    of the exact distance): small triangles are found with a KD-tree on their centroids within e_eq5 + LARGE_TRI_MM,
    the few large triangles left by decimation are tested against every vertex. nearest_exact uses a single radius
    for all triangles and becomes very slow when the reference has one long triangle."""

    return P.nearest_exact_bounded(V, e_eq5, ref, EXACT_MAX_MM, LARGE_TRI_MM)


def grid_box(meta):
    g = meta.get("grid", meta.get("grade"))
    sp = np.asarray(g.get("spacing_mm", g.get("passo_mm")), float)
    org = np.asarray(g.get("origin_mm", g.get("origem_mm")), float)
    shape_zyx = np.asarray(g.get("shape_zyx", g.get("forma_zyx")), float)
    return org, org + sp * (shape_zyx[::-1] - 1)


def outside_volume(V, meta, margin_mm=1.0):
    lo, hi = grid_box(meta)
    return ((V < lo - margin_mm) | (V > hi + margin_mm)).any(axis=1)


def load_meta(z):
    m = z["meta"]
    return json.loads(str(m)) if m.dtype.kind in "US" else m.item()


def load_predictions(results, label):
    files = sorted(glob.glob(os.path.join(results, f"_lofo_{label}_*.npz")))
    if not files:
        raise RuntimeError(f"leave-one-case-out predictions _lofo_{label}_*.npz not found in {results}")
    z = np.load(files[-1], allow_pickle=True)
    pred = {k[5:]: z[k].astype(float) for k in z.files if k.startswith("pred_")}
    const = {k[6:]: float(z[k]) for k in z.files if k.startswith("const_")}
    return pred, const, os.path.basename(files[-1])


def metrics(pred, e):
    pos = e >= np.quantile(e, 1 - C.DECILE_FRACTION)
    return {"rho_vertex": E.spearman(pred, e), "auroc_decile": E.auroc(pred, pos)}


def foot_analysis(cache_dir, results):
    pred_sr, const_sr, f_sr = load_predictions(results, "rf9_sr")
    pred_tri, const_tri, f_tri = load_predictions(results, "rf9_tri")
    per = {}
    pooled = {"pred": [], "e": [], "keep": [], "const": []}
    for path in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        ident = os.path.basename(path)[:-4]
        z = np.load(path, allow_pickle=True)
        meta = load_meta(z)
        V, e_s, q = z["V"].astype(float), z["e"].astype(float), z["q"].astype(float)
        Vt, e_st = z["V_tri"].astype(float), z["e_tri"].astype(float)
        e_x = np.linalg.norm(V - q, axis=1)
        p_sr, p_tri = pred_sr[ident], pred_tri[ident]
        assert len(p_sr) == len(e_s) and len(p_tri) == len(e_st), ident
        # trilinear: rebuild the reference (same chain as a4_run_foot) and measure exactly
        foot_path = os.path.join(C.FOOT_FOLDER, ident)
        hu_t, _thick, grid, spacing, _ts, roi, _tr = prepare_foot(foot_path, C.K_FOOT)
        ref = S.mask_to_mesh(S.segment_bone(hu_t, roi, spacing), grid)
        pts = S.sample_reference(ref, ident)
        assert np.allclose(S.vertex_error(V, pts), e_s, atol=1e-4), f"{ident}: cache not reproduced"
        e_xt = exact_bounded(Vt, e_st, ref)
        floor = S.vertex_error(np.asarray(ref.vertices, float), pts)
        out_sr, out_tri = outside_volume(V, meta), outside_volume(Vt, meta)
        keep = ~out_sr
        d_sr = z["F"].astype(float)[:, DS]
        cal_all = E.calibration(p_sr, e_s, const_sr[ident])
        cal_keep = E.calibration(p_sr[keep], e_s[keep], const_sr[ident])
        cal_exact = E.calibration(p_sr, e_x, const_sr[ident])  # predictions on the Eq. 5 scale vs the exact target
        per[ident] = {
            "n_vertices_sr": int(len(e_s)),
            "n_vertices_tri": int(len(e_st)),
            "sampling_floor": tails(floor),
            "reference_area_mm2": float(ref.area),
            "reference_max_triangle_radius_mm": float(
                np.sqrt(((ref.triangles - ref.triangles.mean(axis=1)[:, None, :]) ** 2).sum(-1)).max()
            ),
            "n_vertices_tri_beyond_exact_max": int((e_st > EXACT_MAX_MM).sum()),
            "sr": {"eq5": tails(e_s), "exact": tails(e_x)},
            "trilinear": {"eq5": tails(e_st), "exact": tails(e_xt)},
            "rho_eq5_vs_exact_sr": E.spearman(e_s, e_x),
            "field_sr_vs_eq5": metrics(p_sr, e_s),
            "field_sr_vs_exact": metrics(p_sr, e_x),
            "field_tri_vs_eq5": metrics(p_tri, e_st),
            "field_tri_vs_exact": metrics(p_tri, e_xt),
            "d_shape_vs_eq5": metrics(d_sr, e_s),
            "d_shape_vs_exact": metrics(d_sr, e_x),
            "outside_volume": {
                "n_sr": int(out_sr.sum()),
                "n_tri": int(out_tri.sum()),
                "max_e_outside_sr_mm": float(e_s[out_sr].max()) if out_sr.any() else 0.0,
            },
            "calibration_sr_eq5": {
                "all_vertices": {k: cal_all[k] for k in ("slope", "intercept_mm", "mae_mm")},
                "without_outside_volume": {k: cal_keep[k] for k in ("slope", "intercept_mm", "mae_mm")},
            },
            "calibration_sr_vs_exact": {
                k: cal_exact[k] for k in ("slope", "intercept_mm", "mae_mm")
            }
            | {
                "median_offset_eq5_minus_exact_mm": float(np.median(e_s - e_x)),
                "mae_median_constant_mm": float(np.mean(np.abs(np.median(e_x) - e_x))),
            },
        }
        pooled["pred"].append(p_sr)
        pooled["e"].append(e_s)
        pooled["keep"].append(keep)
        pooled["const"].append(np.full(len(e_s), const_sr[ident]))
        print(
            "%s | eq5 %.3f exact %.3f | tri eq5 %.3f exact %.3f | floor %.3f | outside %d | rho field eq5 %.3f exact %.3f"
            % (
                ident[:4],
                per[ident]["sr"]["eq5"]["median_mm"],
                per[ident]["sr"]["exact"]["median_mm"],
                per[ident]["trilinear"]["eq5"]["median_mm"],
                per[ident]["trilinear"]["exact"]["median_mm"],
                per[ident]["sampling_floor"]["median_mm"],
                out_sr.sum(),
                per[ident]["field_sr_vs_eq5"]["rho_vertex"],
                per[ident]["field_sr_vs_exact"]["rho_vertex"],
            ),
            flush=True,
        )
    pp, ee, kk, cc = (np.concatenate(pooled[k]) for k in ("pred", "e", "keep", "const"))
    cal_all = E.calibration(pp, ee, float(cc.mean()))
    cal_keep = E.calibration(pp[kk], ee[kk], float(cc[kk].mean()))
    return per, {
        "predictions_files": [f_sr, f_tri],
        "pooled_calibration_sr_eq5_descriptive": {
            "all_vertices": {k: cal_all[k] for k in ("slope", "intercept_mm")},
            "without_outside_volume": {k: cal_keep[k] for k in ("slope", "intercept_mm")},
            "n_vertices_outside_volume": int((~kk).sum()),
            "n_vertices_total": int(len(kk)),
        },
    }


def thorax_analysis(cache_dir):
    per = {}
    for path in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        ident = os.path.basename(path)[:-4]
        z = np.load(path, allow_pickle=True)
        meta = load_meta(z)
        e_s, e_st = z["e"].astype(float), z["e_tri"].astype(float)
        per[ident] = {
            "sr": {"eq5": tails(e_s)},
            "trilinear": {"eq5": tails(e_st)},
            "outside_volume": {
                "n_sr": int(outside_volume(z["V"].astype(float), meta).sum()),
                "n_tri": int(outside_volume(z["V_tri"].astype(float), meta).sum()),
            },
        }
    return per


def cohort(per, path):
    """Summary across cases of the value at `path` (tuple of keys); paired SR vs trilinear when applicable."""
    vals = {}
    for ident, d in per.items():
        v = d
        for k in path:
            v = v[k]
        vals[ident] = float(v)
    return {"summary": E.summary(list(vals.values())), "per_case": vals}


def summarize(per, surfaces=("sr", "trilinear"), measures=("eq5", "exact")):
    out = {}
    for surf in surfaces:
        for meas in measures:
            if meas not in per[next(iter(per))][surf]:
                continue
            for stat in ("median_mm", "mean_mm", "p90_mm", "p99_mm", "fraction_above_2mm_pct"):
                out[f"{surf}_{meas}_{stat}"] = E.summary([d[surf][meas][stat] for d in per.values()])
    for meas in measures:
        if meas not in per[next(iter(per))]["sr"]:
            continue
        for stat in ("median_mm", "mean_mm", "p90_mm", "p99_mm", "fraction_above_2mm_pct"):
            a = {h: d["sr"][meas][stat] for h, d in per.items()}
            b = {h: d["trilinear"][meas][stat] for h, d in per.items()}
            out[f"sr_minus_trilinear_{meas}_{stat}"] = E.paired_comparison(a, b)
    return out


def main():
    if C.ERROR_DEFINITION != "sampled":
        print("exact_error.py analyses caches produced with the sampled definition of Eq. 5; with ERROR_DEFINITION = %r the caches already hold the exact error (see supplement/error_tails.py)" % (C.ERROR_DEFINITION,))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-foot", default=C.A4_CACHE_FOOT)
    ap.add_argument("--cache-thorax", default=C.A4_CACHE_THORAX)
    ap.add_argument("--results", default=C.A4_RESULTS)
    args = ap.parse_args()
    per_foot, pooled = foot_analysis(args.cache_foot, args.results)
    per_thorax = thorax_analysis(args.cache_thorax)
    r = {
        "section": "Supplement: sampling floor of Eq. 5, exact point-to-surface error, error tails, vertices "
        "outside the CT volume",
        "n_feet": len(per_foot),
        "n_thorax": len(per_thorax),
        "foot": {
            "sampling_floor_median_mm": E.summary([d["sampling_floor"]["median_mm"] for d in per_foot.values()]),
            "tails": summarize(per_foot),
            "rho_eq5_vs_exact_sr": E.summary([d["rho_eq5_vs_exact_sr"] for d in per_foot.values()]),
            "field_and_d_shape": {
                k: {
                    m: E.summary([d[k][m] for d in per_foot.values()])
                    for m in ("rho_vertex", "auroc_decile")
                }
                for k in (
                    "field_sr_vs_eq5",
                    "field_sr_vs_exact",
                    "field_tri_vs_eq5",
                    "field_tri_vs_exact",
                    "d_shape_vs_eq5",
                    "d_shape_vs_exact",
                )
            },
            "outside_volume": {
                "n_feet_with_vertices_outside_sr": int(sum(d["outside_volume"]["n_sr"] > 0 for d in per_foot.values())),
                "n_vertices_outside_sr": int(sum(d["outside_volume"]["n_sr"] for d in per_foot.values())),
                "n_vertices_outside_tri": int(sum(d["outside_volume"]["n_tri"] for d in per_foot.values())),
                "max_e_outside_mm": max(d["outside_volume"]["max_e_outside_sr_mm"] for d in per_foot.values()),
            },
            "calibration_sr_vs_exact": {
                k: E.summary([d["calibration_sr_vs_exact"][k] for d in per_foot.values()])
                for k in ("slope", "intercept_mm", "mae_mm", "median_offset_eq5_minus_exact_mm", "mae_median_constant_mm")
            },
            "calibration_sr_eq5": {
                sub: {
                    k: E.summary([d["calibration_sr_eq5"][sub][k] for d in per_foot.values()])
                    for k in ("slope", "intercept_mm", "mae_mm")
                }
                for sub in ("all_vertices", "without_outside_volume")
            },
            **pooled,
            "per_case": per_foot,
        },
        "thorax": {
            "tails": summarize(per_thorax, measures=("eq5",)),
            "outside_volume": {
                "n_cases_with_vertices_outside_sr": int(sum(d["outside_volume"]["n_sr"] > 0 for d in per_thorax.values())),
                "n_vertices_outside_sr": int(sum(d["outside_volume"]["n_sr"] for d in per_thorax.values())),
            },
            "per_case": per_thorax,
        },
        "environment": C.environment_record() if hasattr(C, "environment_record") else {},
    }
    os.makedirs(args.results, exist_ok=True)
    out = os.path.join(args.results, "rS_exact_error.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1, ensure_ascii=False)
    print("saved", out)


if __name__ == "__main__":
    main()
