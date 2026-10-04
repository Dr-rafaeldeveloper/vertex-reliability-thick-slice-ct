"""Sensitivity of the foot results to the choices without a standard in the literature — Gaussian smoothing,
decimation target, morphological cleaning (closing, components < V mm3, ROI: air threshold and dilation) and closing
scale on the thick mask of S_interp. Outside the package hash (subfolder).

Design:
- foot only (48); the SR network is trained ONCE per foot (SR_LR/SR_ITERS/seed of the caches) and all variants use the
 same reconstructed volume — none of the varied parameters acts before the SR;
- each variant redoes, with the same functions of the package: reference (0.5 mm) and its 120 000 points, S_interp,
 trilinear mesh and SR mesh, e(v) and the 9 features; the "base" variant must reproduce foot_cache bit by bit
 (control);
- reliability field per variant: LOFO with the forest configuration selected PER FOLD in the main model
 (rf_selection_rf9_sr_tpe.json), without a new search (same reading as the ablation, §2.10);
- per-foot metrics: median e(v) SR and trilinear, rho per vertex and regional, AUROC of the top decile; paired
 comparison (Wilcoxon) of each variant against the base.
Values: gauss 0.5/1.2 mm and faces 30 000/120 000 and S_interp closing 0/1 iterations;
cleaning one at a time, two values that bracket the adopted one:
closing 1/3 iterations (integers neighboring 2), components 25/100 mm3 (half/double), air ROI -200/-400 HU
(+-100 HU), dilation 1.5/6 mm (half/double). Components and ROI also act on the thick mask of S_interp
(same sequence). The in-plane crop box does not apply to the foot (prepare_foot does not use it).

Usage:
 python a4c_sensitivity.py --process [--n-feet N] [--control] (GPU; writes _sensitivity/<foot>.npz)
 python a4c_sensitivity.py --analyze (CPU; writes results/rC_sensitivity.json)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
from reliability import a4_features as X  # noqa: E402
from reliability import a4_config as C  # noqa: E402
from reliability import a4_data as D  # noqa: E402
from reliability import a4_roi as R  # noqa: E402
from reliability import a4_sr as SR  # noqa: E402
from reliability import a4_surface as S  # noqa: E402

FOLDER = os.path.join(C.A4_OUT, "_sensitivity")

VARIANTS = {
    "base": {},
    "gauss_0.5": {"GAUSS_MM": 0.5},
    "gauss_1.2": {"GAUSS_MM": 1.2},
    "faces_30k": {"TARGET_FACES": 30000},
    "faces_120k": {"TARGET_FACES": 120000},
    "closing_1": {"CLEANING_CLOSING_ITER": 1},
    "closing_3": {"CLEANING_CLOSING_ITER": 3},
    "min_25mm3": {"CLEANING_MIN_MM3": 25.0},
    "min_100mm3": {"CLEANING_MIN_MM3": 100.0},
    "roi_ar_-200": {"ROI_AR_HU": -200.0},
    "roi_ar_-400": {"ROI_AR_HU": -400.0},
    "roi_dilate_1.5": {"ROI_DILATE_MM": 1.5},
    "roi_dilate_6": {"ROI_DILATE_MM": 6.0},
    "sinterp_closing_0": {"SINTERP_CLOSING": 0},
    "sinterp_closing_1": {"SINTERP_CLOSING": 1},
}
GLOBAL = ("CLEANING_CLOSING_ITER", "CLEANING_MIN_MM3")


@contextlib.contextmanager
def override(**kw):
    """Swaps constants of a4_config read at call time and restores them on exit (even with an exception)."""
    old = {k: getattr(C, k) for k in kw}
    try:
        for k, v in kw.items():
            setattr(C, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(C, k, v)


def thick_mask_without_closing(thick, thick_roi, thick_spacing):
    """Variant 'sinterp_closing_0': the same sequence as a4_surface.segment_bone WITHOUT the closing. The package does
 not accept 0 iterations (the crop [it:-it] becomes empty and, in scipy, iterations=0 means 'until convergence'); so
 the sequence is reproduced here with the same functions: threshold, components < CLEANING_MIN_MM3, cavities
 according to CLEANING_FILL_CAVITIES, ROI."""
    from scipy import ndimage as ndi

    assert C.SINTERP_MASK == "cleaning", C.SINTERP_MASK
    m0 = thick > C.BONE_THRESHOLD_HU
    m2 = S._remove_small(m0, C.CLEANING_MIN_MM3, thick_spacing)
    m3 = ndi.binary_fill_holes(m2) if C.CLEANING_FILL_CAVITIES else m2
    return m3 & thick_roi


def variant(v, ident, hu_t, thick, grid, spacing, thick_spacing, roi, thick_roi, rec, tri, k):
    mesh_kw = {
        "gauss_mm": v.get("GAUSS_MM", C.GAUSS_MM),
        "target_faces": v.get("TARGET_FACES", C.TARGET_FACES),
    }
    if "ROI_AR_HU" in v or "ROI_DILATE_MM" in v:
        roi = R.thick_roi(
            thick,
            k,
            spacing,
            ar_hu=v.get("ROI_AR_HU", C.ROI_AR_HU),
            dilate_mm=v.get("ROI_DILATE_MM", C.ROI_DILATE_MM),
        )
        thick_roi = R.roi_on_thick_grid(roi, k)
    with override(**{g: v[g] for g in GLOBAL if g in v}):
        mesh_ref = S.mask_to_mesh(
            S.segment_bone(hu_t, roi, spacing), grid, **mesh_kw
        )
        pts_ref = S.sample_reference(mesh_ref, ident)
        synth = (
            {"CLEANING_CLOSING_ITER": v["SINTERP_CLOSING"]}
            if "SINTERP_CLOSING" in v
            else {}
        )
        if synth.get("CLEANING_CLOSING_ITER") == 0:
            m_thick = thick_mask_without_closing(thick, thick_roi, thick_spacing)
        else:
            with override(**synth):
                m_thick = X.thick_mask_sinterp(thick, thick_roi, thick_spacing)
        s_interp = X.interpolated_surface(m_thick, k, grid)
        mesh_tri = S.mask_to_mesh(
            S.segment_bone(tri, roi, spacing), grid, **mesh_kw
        )
        e_tri = S.vertex_error_target(np.asarray(mesh_tri.vertices, float), mesh_ref, pts_ref)
        mesh_sr = S.mask_to_mesh(
            S.segment_bone(rec, roi, spacing), grid, **mesh_kw
        )
        V = np.asarray(mesh_sr.vertices, float)
        e = S.vertex_error_target(V, mesh_ref, pts_ref)
        F = X.features(mesh_sr, grid, k, s_interp)
    return {
        "e": e.astype(np.float32),
        "F": F.astype(np.float32),
        "V": V.astype(np.float32),
        "e_tri": e_tri.astype(np.float32),
        "faces": [len(mesh_ref.faces), len(mesh_tri.faces), len(mesh_sr.faces)],
    }


def run_processing(filepath, dev, folder=FOLDER):
    from reliability.a4_run_foot import prepare_foot

    ident = os.path.basename(filepath)
    output = os.path.join(folder, ident + ".npz")
    if os.path.exists(output):
        m_prev = json.loads(str(np.load(output, allow_pickle=False)["meta"]))
        if (
            m_prev.get("sha256_script") != sha_script()
            or m_prev.get("environment", {}).get("code_hash_reliability") != C.code_hash()
        ):
            raise RuntimeError(
                "%s generated by another version of the script/package; move it to an archive"
                % output
            )
        return output, None
    t0 = time.time()
    k = C.K_FOOT
    hu_t, thick, grid, spacing, thick_spacing, roi, thick_roi = prepare_foot(filepath, k)
    tn, lo, esc = SR.normalize(thick)
    slices = SR.in_plane_slices(tn)
    network, info = SR.train(
        slices, k, C.SR_ITERS, dev, seed=C.SR_SEED_EVALUATED, verbose=False
    )
    rec_n = 0.5 * (
        SR.apply(network, tn, k, orientations=("yz",))
        + SR.apply(network, tn, k, orientations=("xz",))
    )  # identical to process_foot (§2.4, two families, mean)
    del network
    rec = SR.denormalize(rec_n, lo, esc)
    del rec_n
    tri = D.interpolate_trilinear_z(thick, k)
    arrays, meta = (
        {},
        {
            "ident": ident,
            "faces": {},
            "seconds": {},
            "info_sr": {kk: v for kk, v in info.items() if kk != "loss_history"},
        },
    )
    for name, v in VARIANTS.items():
        t1 = time.time()
        r = variant(
            v, ident, hu_t, thick, grid, spacing, thick_spacing, roi, thick_roi, rec, tri, k
        )
        if C.REMOVE_VERTICES_OUTSIDE_VOLUME:  # same rule as save_cache (e_tri has no vertices here: kept)
            keep = ~S.outside_volume(r["V"], grid.as_dict())
            for kk in ("e", "F", "V"):
                r[kk] = np.asarray(r[kk])[keep]
        for kk in ("e", "F", "V", "e_tri"):
            arrays["%s__%s" % (name, kk)] = r[kk]
        meta["faces"][name] = r["faces"]
        meta["seconds"][name] = round(time.time() - t1, 1)
    meta["seconds"]["total"] = round(time.time() - t0, 1)
    meta["variants"] = VARIANTS
    meta["sha256_script"] = sha_script()
    meta["environment"] = C.environment_record()
    os.makedirs(folder, exist_ok=True)
    tmp = output + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, meta=json.dumps(meta, default=str), **arrays)
    os.replace(tmp, output)
    return output, meta


def sha_script():
    import hashlib

    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def base_control(output):
    """The base variant must reproduce foot_cache (e, F, V, e_tri) bit by bit."""
    z = np.load(output, allow_pickle=False)
    ident = os.path.basename(output)[:-4]
    c = np.load(os.path.join(C.A4_CACHE_FOOT, ident + ".npz"), allow_pickle=False)
    res = {}
    for kk in ("e", "F", "V", "e_tri"):
        a, b = z["base__" + kk], c[kk]
        res[kk] = bool(
            a.shape == b.shape
            and np.array_equal(a.astype(np.float32), b.astype(np.float32))
        )
    return res


def feet_list():
    return json.load(open(C.FOOT_LIST_N48, encoding="utf-8"))


def process(args):
    import torch

    dev = torch.device("cpu") if args.cpu else SR.pick_device(require_cuda=True)
    feet = feet_list()
    if args.n_feet:
        feet = feet[: args.n_feet]
    t0 = time.time()
    failures = []
    for i, fpath in enumerate(feet, 1):
        try:
            output, meta = run_processing(fpath, dev)
        except Exception as exc:  # noqa: BLE001 — record and go on; the resume completes it later
            failures.append(os.path.basename(fpath))
            print(
                "[%d/%d] %s FAILURE %s: %s"
                % (i, len(feet), os.path.basename(fpath), type(exc).__name__, exc),
                flush=True,
            )
            continue
        msg = "[%d/%d] %s %s" % (
            i,
            len(feet),
            os.path.basename(fpath),
            "already done" if meta is None else "ok (%.0fs)" % meta["seconds"]["total"],
        )
        if args.control or i == 1:
            cb = base_control(output)
            msg += " | base reproduces the cache: %s" % cb
            if not all(cb.values()):
                print(msg, flush=True)
                raise SystemExit("base control FAILED")
        print(msg + " | %.1f h" % ((time.time() - t0) / 3600), flush=True)
    if failures:
        raise SystemExit("feet with failure: %s" % failures)


def analyze(args):

    from reliability import a4_analyses as A
    from reliability import a4_statistics as E
    from reliability import a4_optimize_rf as O

    sel = json.load(
        open(
            os.path.join(C.A4_RESULTS, "rf_selection_rf9_sr_tpe.json"),
            encoding="utf-8",
        )
    )
    params = sel["params_per_case"]
    # the reused selection must match the current caches (same guard as _selected_params)
    Fc, ec = {}, {}
    for fpath in feet_list():
        z = np.load(
            os.path.join(C.A4_CACHE_FOOT, os.path.basename(fpath) + ".npz"),
            allow_pickle=False,
        )
        Fc[os.path.basename(fpath)], ec[os.path.basename(fpath)] = (
            z["F"],
            z["e"].astype(np.float32),
        )
    assert sel["key"] == A.memo_key(Fc, ec, None, "rf9_sr"), (
        "rf_selection_rf9_sr_tpe does not correspond to foot_cache"
    )
    fnames = {
        os.path.basename(p)[:-4]: p
        for p in (
            os.path.join(FOLDER, os.path.basename(c) + ".npz") for c in feet_list()
        )
    }
    assert all(os.path.exists(p) for p in fnames.values()), "processed feet are missing"
    ids = sorted(fnames)
    per_var = {}
    faces = {
        h: json.loads(str(np.load(fnames[h], allow_pickle=False)["meta"]))["faces"]
        for h in ids
    }
    for name in VARIANTS:
        Dv, F, e = {}, {}, {}
        for h in ids:
            z = np.load(fnames[h], allow_pickle=False)
            Dv[h] = {
                "e": z[name + "__e"],
                "V": z[name + "__V"],
                "e_tri": z[name + "__e_tri"],
            }
            F[h], e[h] = z[name + "__F"], z[name + "__e"]
        A._LABS.clear()
        pred, _, _, _ = O.nested_lofo(F, e, fixed_params=params, verbose=False)
        pc = A.per_case_full(pred, Dv)
        per_var[name] = {
            h: {
                "e_median_sr": float(np.median(Dv[h]["e"].astype(np.float64))),
                "e_median_tri": float(np.median(Dv[h]["e_tri"].astype(np.float64))),
                "rho_vertex": pc[h]["rho_vertex"],
                "rho_region": pc[h]["rho_region"],
                "auroc_decile": pc[h].get("auroc_decile"),
            }
            for h in ids
        }
        print(
            name,
            "rho %.3f" % np.median([v["rho_vertex"] for v in per_var[name].values()]),
            flush=True,
        )
    base = per_var["base"]
    ref = json.load(
        open(os.path.join(C.A4_RESULTS, "r32_association.json"), encoding="utf-8")
    )["per_case_sr"]
    summary = {}
    for name, pv in per_var.items():
        r = {}
        for m in (
            "e_median_sr",
            "e_median_tri",
            "rho_vertex",
            "rho_region",
            "auroc_decile",
        ):
            x = np.array([pv[h][m] for h in ids], float)
            r[m] = E.summary(list(x))
            if (
                name != "base"
            ):  # same paired comparison as the package (excludes NaN; two-sided)
                r[m]["diff_vs_base"] = E.paired_comparison(
                    {h: pv[h][m] for h in ids}, {h: base[h][m] for h in ids}
                )
        r["median_faces_ref_tri_sr"] = [
            float(np.median([faces[h][name][j] for h in ids])) for j in range(3)
        ]  
        r["sr_lower_than_tri_in"] = int(
            sum(pv[h]["e_median_sr"] < pv[h]["e_median_tri"] for h in ids)
        )
        summary[name] = r
    # required control — the base reproduces r31, r32 and r33 case by case
    r31 = json.load(
        open(
            os.path.join(C.A4_RESULTS, "r31_reconstruction_error.json"),
            encoding="utf-8",
        )
    )["per_case"]
    r33 = json.load(
        open(
            os.path.join(C.A4_RESULTS, "r33_localization_calibration.json"),
            encoding="utf-8",
        )
    )["per_case_sr"]
    tol = 1e-9
    control = {
        "base_rho_equals_r32": bool(
            all(
                abs(base[h]["rho_vertex"] - ref[h]["rho_vertex"]) < 1e-12 for h in ids
            )
        ),
        "base_equals_r31": bool(
            all(
                abs(base[h]["e_median_sr"] - r31[h]["sr"]["median_mm"]) < tol
                and abs(base[h]["e_median_tri"] - r31[h]["trilinear"]["median_mm"])
                < tol
                for h in ids
            )
        ),
        "base_region_auroc_equals_r33": bool(
            all(
                abs(base[h]["rho_region"] - r33[h]["rho_region"]) < 1e-12
                and abs(base[h]["auroc_decile"] - r33[h]["auroc_decile"]) < 1e-12
                for h in ids
            )
        ),
    }
    if not all(control.values()):
        raise SystemExit("base control FAILED: %s" % control)
    out = {
        "section": "Supplement: sensitivity",
        "n_feet": len(ids),
        "variants": VARIANTS,
        "summary": summary,
        "per_case": per_var,
        "control": control,
        "forest_readout": "per-fold configuration reused from rf_selection_rf9_sr_tpe.json (no new search)",
        "_traceability": {
            "environment": C.environment_record(with_torch=False),
            "selection_key": sel["key"],
        },
    }
    p = os.path.join(C.A4_RESULTS, "rC_sensitivity.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("control:", control, "->", p, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--n-feet", type=int, default=0)
    ap.add_argument(
        "--control",
        action="store_true",
        help="checks the base against foot_cache on all feet",
    )
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    if args.process:
        process(args)
    if args.analyze:
        analyze(args)


if __name__ == "__main__":
    main()
