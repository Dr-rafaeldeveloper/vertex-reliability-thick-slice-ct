"""ONE pass per foot (§2.3–§2.8; data for §2.11): everything the foot experiments need,
on the SAME reconstructed surface, in a single cache `foot_cache/<id>.npz`.

Order per foot:
 1. 0.5 mm volume -> thick slice (Eq. 1); thin grid = exam grid (origin/direction/spacing);
 2. soft-tissue ROI of the THICK volume;
 3. reference: 0.5 mm mask with the same ROI -> mesh -> 120 000 points (§2.5, §2.6);
 4. S_interp of the thick mask (§2.7.2);
 5. trilinear -> mask -> mesh -> e(v), F(v);
 6. SR seed 0 (§2.4, two families of planes) -> mask -> EVALUATED mesh -> e(v), F(v), q(v);
 7. members 1–5 with dropout, 4 passes: maps u_ens/u_MC/u_dropens (§2.8.1) and per-member SDF
 (§2.8.3), all from the same reconstructions;
 8. Dropsembles with the prior of the foot's partition (§2.8.2).
Cache: all vertices. Seeds, pick_device, versions and times recorded.
Usage: python src/reliability/a4_run_foot.py [--n-feet N] [--iters N (default a4_config.SR_ITERS)] [--cpu] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reliability import a4_features as X
from reliability import a4_config as C
from reliability import a4_data as D
from reliability import a4_uncertainty as U
from reliability import a4_roi as R
from reliability import a4_sr as SR
from reliability import a4_surface as S
from reliability.a4_grid import Grid


def partitions(listing):
    """§2.8.2: 4 partitions; the prior of partition g is trained on the feet OUTSIDE g (36 feet × 12 slices)."""
    return {filepath: i % C.DS_PARTITIONS for i, filepath in enumerate(listing)}


def pool_prior(listing, group, g, k):
    """432 slices: 12 in-plane slices (normalized) from each of the 36 feet outside partition g."""
    rng = np.random.default_rng(1000 + g)
    pool = []
    n_feet = 0
    for filepath in listing:
        if group[filepath] == g:
            continue
        hu = sitk.GetArrayFromImage(sitk.ReadImage(filepath)).astype(np.float32)
        _, thick = D.thick_slice(hu, k)
        tn, _, _ = SR.normalize(thick)
        sel = rng.choice(
            tn.shape[0], size=min(C.DS_PRIOR_SLICES_PER_FOOT, tn.shape[0]), replace=False
        )
        pool += [tn[s] for s in sel]
        n_feet += 1
    return pool, n_feet


def prepare_foot(filepath, k=C.K_FOOT):
    """Common geometric chain of the foot up to the ROI (§2.3, Eq. 1–2; grid; ROI): truncated thin volume, thick
 volume, thin grid, spacings (thin and thick), thin ROI and ROI on the thick grid. Used by process_foot and by
 other scripts, so that the recomputed S_interp comes from the same geometry that generated the caches."""
    img = sitk.ReadImage(filepath)
    hu = sitk.GetArrayFromImage(img).astype(np.float32)
    hu_t, thick = D.thick_slice(hu, k)  # §2.3, Eq. 1–2
    # THICK grid (z spacing = k·Δz, center of slice 0 at thin index (k-1)/2) and thin grid derived from it:
    # the thin grid derives structurally from the thick volume; the 0.5 mm header is only used to check equality
    g_exam = Grid.from_sitk(img, shape=hu_t.shape)
    thick_grid = Grid(
        (g_exam.spacing[0], g_exam.spacing[1], g_exam.spacing[2] * k),
        g_exam.index_to_physical(np.array([[0.0, 0.0, (k - 1) / 2.0]]))[0],
        g_exam.D.reshape(-1),
        thick.shape,
    )
    grid = thick_grid.refined_z(k)
    assert (
        np.allclose(grid.origin, g_exam.origin)
        and np.allclose(grid.spacing, g_exam.spacing)
        and grid.shape == hu_t.shape
    )
    spacing = tuple(grid.spacing)
    thick_spacing = (spacing[0], spacing[1], spacing[2] * k)
    roi = R.thick_roi(thick, k, spacing)
    thick_roi = R.roi_on_thick_grid(roi, k)  
    return hu_t, thick, grid, spacing, thick_spacing, roi, thick_roi


def process_foot(
    filepath,
    dev,
    k=C.K_FOOT,
    iters=C.SR_ITERS,
    prior=None,
    iters_refinement=C.DS_REFINEMENT_ITERS,
    with_uncertainty=True,
    verbose=True,
    check_determinism=False,
):
    ident = os.path.basename(filepath)
    t0 = time.time()
    times = {}
    hu_t, thick, grid, spacing, thick_spacing, roi, thick_roi = prepare_foot(filepath, k)
    # 3. reference (uses the 0.5 mm ONLY here and in the error)
    st_ref, st_tri, st_sr = {}, {}, {}
    mesh_ref = S.mask_to_mesh(
        S.segment_bone(hu_t, roi, spacing, stats=st_ref), grid
    )  # §2.5
    pts_ref = S.sample_reference(mesh_ref, ident)  # §2.6
    times["reference"] = time.time() - t0
    # 4. S_interp
    st_synth = {}
    m_thick = X.thick_mask_sinterp(thick, thick_roi, thick_spacing, stats=st_synth)
    s_interp = X.interpolated_surface(m_thick, k, grid)  # §2.7.2
    # 5. trilinear
    tri = D.interpolate_trilinear_z(thick, k)  # §2.4 baseline
    mesh_tri = S.mask_to_mesh(
        S.segment_bone(tri, roi, spacing, stats=st_tri), grid
    )
    V_tri = np.asarray(mesh_tri.vertices, float)
    e_tri = S.vertex_error_target(V_tri, mesh_ref, pts_ref)
    F_tri = X.features(mesh_tri, grid, k, s_interp)
    times["trilinear"] = time.time() - t0
    # 6. evaluated SR
    tn, lo, esc = SR.normalize(thick)
    slices = SR.in_plane_slices(tn)
    network0, info_sr = SR.train(
        slices, k, iters, dev, seed=C.SR_SEED_EVALUATED, verbose=verbose
    )
    rec_yz = SR.apply(network0, tn, k, orientations=("yz",))
    rec_xz = SR.apply(network0, tn, k, orientations=("xz",))
    rec_n = 0.5 * (rec_yz + rec_xz)  # §2.4 (two families, mean)
    tri_n = D.interpolate_trilinear_z(tn, k)
    evid_sr = {
        "diff_mean_yz_xz_norm": float(np.abs(rec_yz - rec_xz).mean()),
        "diff_mean_sr_trilinear_norm": float(np.abs(rec_n - tri_n).mean()),
        "diff_mean_yz_xz_HU": float(np.abs(rec_yz - rec_xz).mean() * esc),
        "diff_mean_sr_trilinear_HU": float(np.abs(rec_n - tri_n).mean() * esc),
    }
    del rec_yz, rec_xz, tri_n
    rec = SR.denormalize(rec_n, lo, esc)
    if check_determinism:  # run-to-run on the GPU
        network_b, _ = SR.train(
            slices, k, iters, dev, seed=C.SR_SEED_EVALUATED, verbose=False
        )
        evid_sr["determinism_max_weight_diff"] = float(
            max(
                (a - b).abs().max().item()
                for a, b in zip(
                    network0.state_dict().values(),
                    network_b.state_dict().values(),
                    strict=True,
                )
            )
        )
        evid_sr["determinism_max_reconstruction_diff_norm"] = float(
            np.abs(SR.apply(network_b, tn, k) - rec_n).max()
        )
        del network_b
    del network0
    mesh_sr = S.mask_to_mesh(
        S.segment_bone(rec, roi, spacing, stats=st_sr), grid
    )
    V = np.asarray(mesh_sr.vertices, float)
    N = np.asarray(mesh_sr.vertex_normals, float)
    e = S.vertex_error_target(V, mesh_ref, pts_ref)  # Eq. 5, or exact (C.ERROR_DEFINITION)
    border_sr = S.open_border(mesh_sr)  # evidence of the open border
    F = X.features(mesh_sr, grid, k, s_interp)  # Eq. 7–14
    q = S.reference_correspondence(V, mesh_ref)  # §2.11 exact matches
    times["sr"] = time.time() - t0
    out = {
        "ident": ident,
        "V": V.astype(np.float32),
        "N": N.astype(np.float32),
        "F": F,
        "e": e.astype(np.float32),
        "q": q.astype(np.float32),
        "V_tri": V_tri.astype(np.float32),
        "F_tri": F_tri,
        "e_tri": e_tri.astype(np.float32),
        "faces": {
            "reference": len(mesh_ref.faces),
            "trilinear": len(mesh_tri.faces),
            "sr": len(mesh_sr.faces),
            "s_interp": len(s_interp.faces),
        },
        "faces_before": {
            nm: m.metadata.get("faces_before_decimation")
            for nm, m in (
                ("reference", mesh_ref),
                ("trilinear", mesh_tri),
                ("sr", mesh_sr),
            )
        },
        "components": {
            nm: m.metadata.get("components")
            for nm, m in (
                ("reference", mesh_ref),
                ("trilinear", mesh_tri),
                ("sr", mesh_sr),
            )
        },
        "sr_evidence": evid_sr,
        "roi_thin_voxels": int(roi.sum()),
        "roi_thick_voxels": int(thick_roi.sum()),
        "cleaning": {
            "reference": st_ref,
            "trilinear": st_tri,
            "sr": st_sr,
            "s_interp": st_synth,
        },
        "d_shape_outliers": _outliers_dshape(F[:, 5], V, grid, hu_t.shape),
        "info_sr": info_sr,
        "lo": lo,
        "esc": esc,
        "grid": grid.as_dict(),
        "k": k,
    }
    if with_uncertainty:
        # 7. ensemble + MC + surface, from the SAME reconstructions
        inc = U.ensemble_mc(
            slices, tn, k, dev, lo, esc, grid, roi, V, iters, verbose=verbose
        )  # §2.8.1, §2.8.3
        out.update(inc)
        times["ensemble"] = time.time() - t0
        # 8. Dropsembles
        if prior is not None:
            pnet, phys, anc = prior
            u_ds, lambdas = U.dropsembles(
                pnet,
                phys,
                anc,
                slices,
                tn,
                k,
                dev,
                lo,
                esc,
                grid,
                V,
                iters_refinement=iters_refinement,
                verbose=verbose,
            )
            out["u_ds"] = u_ds
            out["lambdas_ewc"] = lambdas
        times["dropsembles"] = time.time() - t0
    out["open_border"] = _border_stats(border_sr, F, mesh_ref)
    out[
        "area_mm2"
    ] = {  # density of the 120 000 points = 120000 / reference area
        "reference": float(mesh_ref.area),
        "sr": float(mesh_sr.area),
        "trilinear": float(mesh_tri.area),
    }
    out["times_s"] = {kk: round(v, 1) for kk, v in times.items()}
    return out


def _border_stats(border, F, mesh_ref):
    """fraction of border vertices in the SR mesh and distribution of x3 (angular defect) and
 x4 (roughness) on them vs in the interior; border fraction in the reference. Declared consequence of the removal
 of padding."""
    b = np.asarray(border, bool)
    br = S.open_border(mesh_ref)

    def est(x):
        x = np.asarray(x, float)
        return {
            "border_median": float(np.median(x[b])) if b.any() else None,
            "border_p90": float(np.percentile(x[b], 90)) if b.any() else None,
            "interior_median": float(np.median(x[~b])) if (~b).any() else None,
            "interior_p90": float(np.percentile(x[~b], 90)) if (~b).any() else None,
        }

    return {
        "fraction_vertices_border_sr": float(b.mean()),
        "n_vertices_border_sr": int(b.sum()),
        "fraction_vertices_border_ref": float(br.mean()),
        "x3_angular_defect_rad": est(F[:, 2]),
        "x4_roughness_mm": est(F[:, 3]),
    }


def _outliers_dshape(dshape, V, grid, shape_zyx, threshold_mm=3.0, border_mm=2.0):
    """Evidence: fraction of vertices with d_shape > 3 mm and, among them, the fraction
 within 2 mm of the volume extremes in z."""
    high = dshape > threshold_mm
    if not high.any():
        return {"fraction_above_3mm": 0.0, "fraction_of_those_near_z_border": None}
    zi = grid.z_index(V[high])
    dz = float(grid.spacing[2])
    d_border = np.minimum(zi, shape_zyx[0] - 1 - zi) * dz
    return {
        "fraction_above_3mm": float(high.mean()),
        "n_above_3mm": int(high.sum()),
        "fraction_of_those_near_z_border": float((d_border <= border_mm).mean()),
        "median_mm_of_high": float(np.median(dshape[high])),
    }


def save_cache(out, folder):
    os.makedirs(folder, exist_ok=True)
    p = os.path.join(folder, out["ident"] + ".npz")
    if "grid" in out and "V" in out:
        out = S.remove_outside_volume(out)
    arrays = {kk: v for kk, v in out.items() if isinstance(v, np.ndarray)}
    meta = {kk: v for kk, v in out.items() if not isinstance(v, np.ndarray)}
    meta["environment"] = C.environment_record()
    tmp = p + ".tmp"  # atomic write (a restart midway does not leave a truncated .npz)
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f, meta=json.dumps(meta, ensure_ascii=False, default=str), **arrays
        )
    os.replace(tmp, p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-feet", type=int, default=0, help="0 = all 48")
    ap.add_argument("--iters", type=int, default=C.SR_ITERS)
    ap.add_argument("--iters-prior", type=int, default=C.DS_PRIOR_ITERS)
    ap.add_argument("--iters-refinement", type=int, default=C.DS_REFINEMENT_ITERS)
    ap.add_argument("--cpu", action="store_true", help="ONLY for the smoke test")
    ap.add_argument("--without-uncertainty", action="store_true")
    ap.add_argument("--cache-dir", default=C.A4_CACHE_FOOT)
    args = ap.parse_args()
    C.ensure_folders()
    dev = torch.device("cpu") if args.cpu else SR.pick_device(require_cuda=True)  
    listing = json.load(open(C.FOOT_LIST_N48))
    if args.n_feet > 0:
        run_list = listing[: args.n_feet]
    else:
        run_list = listing
    group = partitions(listing)  # partitions over the 48 (§2.8.2)
    priors = {}
    infos_prior = {}
    print(
        "pick_device:",
        dev,
        "| feet:",
        len(run_list),
        "| cache:",
        args.cache_dir,
        flush=True,
    )
    t0 = time.time()
    for i, filepath in enumerate(run_list, 1):
        ident = os.path.basename(filepath)
        cp = os.path.join(args.cache_dir, ident + ".npz")
        if os.path.exists(cp):
            print(
                "[%d/%d] %s (existing cache)" % (i, len(run_list), ident), flush=True
            )
            continue
        prior = None
        if not args.without_uncertainty:
            g = group[filepath]
            if g not in priors:
                pool, n_feet = pool_prior(listing, group, g, C.K_FOOT)
                print(
                    "  prior of partition %d: %d slices from %d feet"
                    % (g, len(pool), n_feet),
                    flush=True,
                )
                pnet, phys, anc, info = U.train_prior(
                    pool,
                    C.K_FOOT,
                    dev,
                    C.DS_PRIOR_SEED_BASE + g,
                    iters=args.iters_prior,
                    verbose=False,
                )
                priors[g] = (pnet, phys, anc)
                infos_prior[g] = {
                    kk: v for kk, v in info.items() if kk != "loss_history"
                } | {"partition": g, "n_feet_pool": n_feet}
                print("  prior %d ready (%.0fs)" % (g, time.time() - t0), flush=True)
            prior = priors[g]
        out = process_foot(
            filepath,
            dev,
            iters=args.iters,
            prior=prior,
            iters_refinement=args.iters_refinement,
            with_uncertainty=not args.without_uncertainty,
            verbose=False,
        )
        out["partition"] = group[filepath]
        if not args.without_uncertainty:
            out["info_prior"] = infos_prior[
                group[filepath]
            ]  # lr/iters/pool of the prior used
        save_cache(out, args.cache_dir)
        print(
            "[%d/%d] %s ok | %d vertices | median e SR %.3f tri %.3f mm | faces %s | %s (%.0fs)"
            % (
                i,
                len(run_list),
                ident,
                len(out["V"]),
                np.median(out["e"]),
                np.median(out["e_tri"]),
                out["faces"],
                out["times_s"],
                time.time() - t0,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
