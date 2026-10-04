"""SMOKE TEST: 1 foot, few iterations, CPU. Runs the complete pass
(a4_run_foot.process_foot with a tiny prior) and writes a JSON with the REAL statistics
of the vectors. It is not a result of the paper.
Usage: python src/reliability/a4_smoke.py [--iters 60] [--gpu]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reliability import a4_config as C
from reliability import a4_statistics as E
from reliability import a4_uncertainty as U
from reliability import a4_registration as REG
from reliability import a4_run_foot as RP
from reliability import a4_sr as SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument(
        "--thorax", action="store_true", help="smoke test of the thoracic path (1 case)"
    )
    args = ap.parse_args()
    C.ensure_folders()
    dev = SR.pick_device(require_cuda=True) if args.gpu else torch.device("cpu")
    if args.thorax:
        return thorax_smoke(args, dev)
    listing = json.load(open(C.FOOT_LIST_N48))
    filepath = listing[0]
    ident = os.path.basename(filepath)
    group = RP.partitions(listing)
    g = group[filepath]
    t0 = time.time()
    pool, n_feet = RP.pool_prior(
        listing[:8], {c: group[c] for c in listing[:8]}, g, C.K_FOOT
    )  # tiny prior (smoke test)
    pnet, phys, anc, info_prior = U.train_prior(
        pool, C.K_FOOT, dev, C.DS_PRIOR_SEED_BASE + g, iters=args.iters, verbose=False
    )
    info_prior = {k2: v for k2, v in info_prior.items() if k2 != "loss_history"}
    info_prior["n_slices_pool"] = len(pool)
    info_prior["n_feet_pool"] = n_feet
    out = RP.process_foot(
        filepath,
        dev,
        iters=args.iters,
        prior=(pnet, phys, anc),
        iters_refinement=max(60, args.iters),
        verbose=False,
        check_determinism=True,
    )
    # population of the deciles (Eq. 46) and size of the 25 regions (Eq. 36) with a trial score (d_shape) — exercises the
    # code
    lab = E.regions(out["V"].astype(float))
    size = np.bincount(lab, minlength=C.K_REGIONS)
    dec = E.calibration(
        out["F"][:, 5].astype(float), out["e"].astype(float), float(out["e"].mean())
    )["deciles"]
    F, e, V, q = out["F"], out["e"], out["V"], out["q"]
    # record with a trial ê (= d_shape, since there is no LOO model with 1 foot): only to exercise the 5 strategies
    reg = REG.record_case(
        V, q, F[:, 5], F[:, 5], e, seed=C.REG_SEED, trials=20, labels=lab
    )
    # BH with a known example (Benjamini & Hochberg 1995, m = 4)
    bh = E.benjamini_hochberg({"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.5})
    h, _ = np.histogram(F[:, 1], bins=6, range=(0, 1.5))
    est = {
        "description": __doc__,
        "foot": ident,
        "pick_device": str(dev),
        "iters": args.iters,
        "seconds": round(time.time() - t0, 1),
        "faces_after_decimation": out["faces"],
        "target_faces": C.TARGET_FACES,
        "faces_before_decimation": out["faces_before"],
        "mesh_components": out["components"],
        "crop_box": "does not apply to the foot (crop only in the thorax, §2.2.2)",
        "roi_thin_voxels": out["roi_thin_voxels"],
        "roi_thick_voxels": out["roi_thick_voxels"],
        "morphological_cleaning": out["cleaning"],
        "d_shape_outliers": out["d_shape_outliers"],
        "sr_evidence": out["sr_evidence"],
        "kmeans_regions": {
            "n": len(size),
            "min": int(size.min()),
            "max": int(size.max()),
            "empty": int((size == 0).sum()),
        },
        "population_deciles": [d["n"] for d in dec],
        "segmentation_parameters": {
            "threshold_hu": C.BONE_THRESHOLD_HU,
            "closing": C.CLEANING_CLOSING_ITER,
            "min_mm3": C.CLEANING_MIN_MM3,
            "fill_cavities": C.CLEANING_FILL_CAVITIES,
            "gauss_mm": C.GAUSS_MM,
            "mc": C.MARCHING_CUBES_METHOD,
            "taubin_half_steps": C.TAUBIN_HALF_STEPS,
            "lambda_nu": [C.TAUBIN_LAMBDA, C.TAUBIN_NU],
            "target_faces": C.TARGET_FACES,
            "roi": {
                "ar_hu": C.ROI_AR_HU,
                "opening": C.ROI_OPENING_ITER,
                "dilate_mm": C.ROI_DILATE_MM,
            },
        },
        "sr_training": {
            "n_slices": out["info_sr"]["n_slices"],
            "n_training": out["info_sr"]["n_training"],
            "n_reserved": out["info_sr"]["n_reserved"],
            "effective_batch_min": out["info_sr"]["effective_batch_min"],
            "effective_batch_mode": out["info_sr"]["effective_batch_mode"],
            "n_iterations_run": out["info_sr"]["n_iterations_run"],
        },
        "dropsembles_prior": info_prior,
        "n_vertices_sr": len(V),
        "n_vertices_tri": len(out["V_tri"]),
        "n_columns_F": int(F.shape[1]),
        "grid": out["grid"],
        "k": out["k"],
        "d_f_mm": {
            "min": float(F[:, 1].min()),
            "max": float(F[:, 1].max()),
            "histogram_6_bins_0_1p5": (h / h.sum()).round(3).tolist(),
            "max_le_1p5": bool(F[:, 1].max() <= 1.5 + 1e-6),
        },
        "angular_defect_rad": {
            "min": float(F[:, 2].min()),
            "max": float(F[:, 2].max()),
            "negative_fraction": float((F[:, 2] < 0).mean()),
        },
        "roughness_mm": {
            "min": float(F[:, 3].min()),
            "median": float(np.median(F[:, 3])),
            "max": float(F[:, 3].max()),
        },
        "d_shape_mm": {
            "min": float(F[:, 5].min()),
            "median": float(np.median(F[:, 5])),
            "max": float(F[:, 5].max()),
            "nan": int(np.isnan(F[:, 5]).sum()),
        },
        "constants": {
            "delta_z": float(F[0, 6]),
            "t_slice": float(F[0, 7]),
            "delta_xy": float(F[0, 8]),
            "constants_in_foot": bool(np.ptp(F[:, 6:9], axis=0).max() == 0),
        },
        "nz_abs": {"min": float(F[:, 0].min()), "max": float(F[:, 0].max())},
        "e_mm": {
            "median_sr": float(np.median(e)),
            "p90_sr": float(np.percentile(e, 90)),
            "median_tri": float(np.median(out["e_tri"])),
        },
        "q_correspondence": {
            "median_|v-q|": float(np.median(np.linalg.norm(V - q, axis=1))),
            "q_le_e_always": bool((np.linalg.norm(V - q, axis=1) <= e + 1e-3).all()),
            "tolerance_mm": 1e-3,
            "note": "V, q, e in float32 with coordinates up to ~1400 mm (precision ~1e-4 mm)",
        },
        "intensity_uncertainty_HU": {
            k2: {
                "min": float(out[k2].min()),
                "median": float(np.median(out[k2])),
                "max": float(out[k2].max()),
                "positive": bool(out[k2].min() >= 0),
            }
            for k2 in ("u_ens", "u_mc", "u_dropens", "u_ds")
        },
        "surface_uncertainty_mm": {
            k2: {
                "min": float(out[k2].min()),
                "median": float(np.median(out[k2])),
                "max": float(out[k2].max()),
            }
            for k2 in ("u_geo_std", "u_geo_abs", "u_geo_mc")
        },
        "d_geo_members_shape": list(out["d_geo_members"].shape),
        "d_geo_mc_shape": list(out["d_geo_mc"].shape),
        "signed_d_geo": bool(
            out["d_geo_members"].min() < 0 < out["d_geo_members"].max()
        ),
        "lambdas_ewc": out.get("lambdas_ewc"),
        "registration_strategies": sorted(reg.keys()),
        "registration_median_mm": {s: float(np.median(v)) for s, v in reg.items()},
        "benjamini_hochberg_example": bh,
        "expected_bh": {"a": 0.04, "b": 0.04, "c": 0.04, "d": 0.5},
        "times_s": out["times_s"],
        "info_sr": {k2: v for k2, v in out["info_sr"].items() if k2 != "loss_history"},
        "environment": C.environment_record(),
    }
    p = os.path.join(
        C.A4_SMOKE, "smoke_%s_%s.json" % (ident, "cuda" if args.gpu else "cpu")
    )
    json.dump(est, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    # assertions — the JSON is already written for inspection even on failure
    assert (
        est["n_columns_F"] == 9
        and est["d_f_mm"]["max_le_1p5"]
        and est["angular_defect_rad"]["negative_fraction"] > 0.1
    )
    assert (
        est["sr_training"]["n_reserved"] == 0
        and est["sr_training"]["n_training"] == est["sr_training"]["n_slices"]
    )
    assert (
        est["dropsembles_prior"]["n_training"]
        == est["dropsembles_prior"]["n_slices_pool"]
    )
    assert (
        est["kmeans_regions"]["empty"] == 0
        and est["sr_evidence"]["diff_mean_yz_xz_norm"] > 0
    )
    assert (
        est["sr_evidence"]["determinism_max_weight_diff"] == 0.0
        and est["sr_evidence"]["determinism_max_reconstruction_diff_norm"] == 0.0
    )
    assert all(
        l["converged"]
        and abs(l["effective_ratio_in_adaptation_it50"] - C.DS_EWC_FRACTION)
        / C.DS_EWC_FRACTION
        <= C.DS_EWC_TOLERANCE
        for l in est["lambdas_ewc"]
    )
    assert (
        est["d_shape_mm"]["nan"] == 0
        and est["constants"]["constants_in_foot"]
        and est["signed_d_geo"]
    )
    assert (
        all(v["positive"] for v in est["intensity_uncertainty_HU"].values())
        and est["q_correspondence"]["q_le_e_always"]
    )
    assert (
        abs(min(est["d_f_mm"]["histogram_6_bins_0_1p5"]) - 1 / 6) < 0.05
        and abs(max(est["d_f_mm"]["histogram_6_bins_0_1p5"]) - 1 / 6) < 0.05
    )
    assert all(abs(bh[k2] - est["expected_bh"][k2]) < 1e-9 for k2 in bh) and est[
        "registration_strategies"
    ] == sorted(C.REG_STRATEGIES)
    for name, nf in est["faces_after_decimation"].items():
        if name != "s_interp":
            assert nf == C.TARGET_FACES, (name, nf)
    est["assertions"] = "all OK"
    json.dump(est, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(
        json.dumps(
            {
                k2: v
                for k2, v in est.items()
                if k2
                in (
                    "faces_after_decimation",
                    "n_vertices_sr",
                    "d_f_mm",
                    "angular_defect_rad",
                    "d_shape_mm",
                    "e_mm",
                    "intensity_uncertainty_HU",
                    "surface_uncertainty_mm",
                    "lambdas_ewc",
                    "registration_median_mm",
                    "times_s",
                )
            },
            indent=1,
            ensure_ascii=False,
            default=str,
        )
    )
    print("SMOKE TEST OK -> %s (%.0fs)" % (p, time.time() - t0))


def thorax_smoke(args, dev):
    """Smoke test of the thoracic path: 1 case, few iterations; JSON with faces, k, constants, d_f, box, grid, header."""
    from reliability import a4_data as D
    from reliability import a4_run_thorax as RT

    t0 = time.time()
    thin, thk, ident, part = D.rplhr_pairs()[0]
    out = RT.process_case(thin, thk, ident, dev, iters=args.iters)
    F = out["F"]
    k = out["k"]
    dmax = k * out["grid"]["spacing_mm"][2] / 2.0
    h, _ = np.histogram(F[:, 1], bins=5, range=(0, dmax))
    est = {
        "description": "smoke test of the thoracic path (§2.2.2); not a result",
        "case": ident,
        "partition": part,
        "pick_device": str(dev),
        "iters": args.iters,
        "k": k,
        "faces_after_decimation": out["faces"],
        "faces_before_decimation": out["faces_before"],
        "mesh_components": out["components"],
        "morphological_cleaning": out["cleaning"],
        "roi_thin_voxels": out["roi_thin_voxels"],
        "roi_thick_voxels": out["roi_thick_voxels"],
        "n_vertices_sr": len(out["V"]),
        "n_vertices_tri": len(out["V_tri"]),
        "n_columns_F": int(F.shape[1]),
        "constants": {
            "delta_z": float(F[0, 6]),
            "t_slice": float(F[0, 7]),
            "delta_xy": float(F[0, 8]),
        },
        "d_f_mm": {
            "min": float(F[:, 1].min()),
            "max": float(F[:, 1].max()),
            "theoretical_max": dmax,
            "histogram_5_bins": (h / h.sum()).round(3).tolist(),
        },
        "angular_defect_rad": {"negative_fraction": float((F[:, 2] < 0).mean())},
        "d_shape_mm": {
            "median": float(np.median(F[:, 5])),
            "nan": int(np.isnan(F[:, 5]).sum()),
        },
        "e_mm": {
            "median_sr": float(np.median(out["e"])),
            "median_tri": float(np.median(out["e_tri"])),
        },
        "info": out["info"],
        "grid": out["grid"],
        "sr_training": {
            k2: out["info_sr"][k2]
            for k2 in (
                "n_slices",
                "n_training",
                "n_reserved",
                "effective_batch_min",
                "effective_batch_mode",
            )
        },
        "seconds": round(time.time() - t0, 1),
        "environment": C.environment_record(),
    }
    p = os.path.join(
        C.A4_SMOKE, "thorax_smoke_%s_%s.json" % (ident, "cuda" if args.gpu else "cpu")
    )
    json.dump(est, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    c = est["constants"]
    assert (
        est["k"] == 5
        and abs(c["delta_z"] - 1.0) < 1e-5
        and abs(c["t_slice"] - 5.0) < 1e-5
        and abs(c["delta_xy"] - 0.7) < 1e-5
        and est["n_columns_F"] == 9
    )
    assert all(
        v == C.TARGET_FACES
        for nm, v in est["faces_after_decimation"].items()
        if nm != "s_interp"
    )
    assert (
        est["d_f_mm"]["max"] <= dmax + 1e-6
        and est["d_shape_mm"]["nan"] == 0
        and est["sr_training"]["n_reserved"] == 0
    )
    assert (
        abs(min(est["d_f_mm"]["histogram_5_bins"]) - 0.2) < 0.06
        and abs(max(est["d_f_mm"]["histogram_5_bins"]) - 0.2) < 0.06
    )
    est["assertions"] = "all OK"
    json.dump(est, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(
        json.dumps(
            {
                k2: est[k2]
                for k2 in (
                    "faces_after_decimation",
                    "constants",
                    "d_f_mm",
                    "e_mm",
                    "n_vertices_sr",
                )
            },
            indent=1,
            default=str,
        )
    )
    print("THORAX SMOKE TEST OK -> %s (%.0fs)" % (p, est["seconds"]))


if __name__ == "__main__":
    main()
