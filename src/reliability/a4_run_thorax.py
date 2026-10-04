"""One pass per thoracic case (§2.2.2): 5 mm = input, regridded 1 mm = reference; same
protocol as the foot (§2.3–§2.7: ROI of the thick volume, segmentation, mesh at 60 000 faces, 120 000 points,
S_interp, nine features, SR in both families of planes and trilinear). No uncertainties or registration (the
text only defines them for the foot). Cache `thorax_cache/<id>.npz` with all vertices.
Usage: python src/reliability/a4_run_thorax.py [--n-cases N] [--iters N (default a4_config.SR_ITERS)]
 [--partition all|val|test] [--cpu]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reliability import a4_features as X
from reliability import a4_config as C
from reliability import a4_data as D
from reliability import a4_roi as R
from reliability import a4_sr as SR
from reliability import a4_surface as S
from reliability.a4_grid import Grid
from reliability.a4_run_foot import _border_stats  


def prepare_thorax(thin, thk):
    """Common geometric chain of the thorax up to the ROI (§2.2.2 regrid/crops; grid; ROI): regridded reference,
 thick volume, thin grid, spacings, thin ROI and ROI on the thick grid, k and info. Used by process_case."""
    hu_ref, thick, spacing, k, info = D.load_rplhr_pair(
        thin, thk
    )  # §2.2.2 regrid, HU, 0.7 mm, crop in Z
    # real header of the 5 mm: origin/direction recorded and identity direction asserted (the grid below assumes it)
    import SimpleITK as sitk

    _thk = sitk.ImageFileReader()
    _thk.SetFileName(thk)
    _thk.ReadImageInformation()
    info["thick_header"] = {
        "origin": list(_thk.GetOrigin()),
        "direction": list(_thk.GetDirection()),
        "spacing": list(_thk.GetSpacing()),
    }
    assert np.allclose(_thk.GetDirection(), np.eye(3).reshape(-1)), (
        "RPLHR direction is not the identity: %s" % (_thk.GetDirection(),)
    )
    assert np.allclose(_thk.GetOrigin(), 0.0), "RPLHR origin is not zero: %s" % (
        _thk.GetOrigin(),
    )
    assert info["coverage_after_crop"] == 1.0, (
        "reference sub-slice outside the thin volume after the crop"
    )
    box = R.plane_box(thick)  # in-plane crop (same box in both)
    hu_ref, thick = D.crop_plane(hu_ref, thick, box)
    info["box_yx"] = list(box)
    y0, _, x0, _ = box
    z0 = info["crop_z_blocks"][0]
    # thick-volume grid after the crops (RPLHR header: origin 0, identity direction, assumed spacing)
    thick_grid = Grid(
        (spacing[0], spacing[1], spacing[2] * k),
        (x0 * spacing[0], y0 * spacing[1], z0 * spacing[2] * k),
        None,
        thick.shape,
    )
    grid = thick_grid.refined_z(k)  # regridded reference grid
    assert grid.shape == hu_ref.shape, (grid.shape, hu_ref.shape)
    spacing_f = tuple(grid.spacing)
    thick_spacing = (spacing_f[0], spacing_f[1], spacing_f[2] * k)
    roi = R.thick_roi(thick, k, spacing_f)
    thick_roi = R.roi_on_thick_grid(roi, k)  # (in thorax: trunk)
    return hu_ref, thick, grid, spacing_f, thick_spacing, roi, thick_roi, k, info


def process_case(thin, thk, ident, dev, iters=C.SR_ITERS, verbose=False):
    t0 = time.time()
    times = {}
    hu_ref, thick, grid, spacing_f, thick_spacing, roi, thick_roi, k, info = prepare_thorax(
        thin, thk
    )
    st_ref, st_tri, st_sr = {}, {}, {}
    mesh_ref = S.mask_to_mesh(
        S.segment_bone(hu_ref, roi, spacing_f, stats=st_ref), grid
    )  # §2.5
    pts_ref = S.sample_reference(mesh_ref, ident)  # §2.6
    times["reference"] = time.time() - t0
    st_synth = {}
    m_thick = X.thick_mask_sinterp(
        thick, thick_roi, thick_spacing, stats=st_synth
    )  # §2.7.2
    s_interp = X.interpolated_surface(m_thick, k, grid)  # §2.7.2
    tri = D.interpolate_trilinear_z(thick, k)
    mesh_tri = S.mask_to_mesh(
        S.segment_bone(tri, roi, spacing_f, stats=st_tri), grid
    )
    V_tri = np.asarray(mesh_tri.vertices, float)
    e_tri = S.vertex_error(V_tri, pts_ref)
    F_tri = X.features(mesh_tri, grid, k, s_interp)
    times["trilinear"] = time.time() - t0
    tn, lo, esc = SR.normalize(thick)
    slices = SR.in_plane_slices(tn)
    network0, info_sr = SR.train(
        slices, k, iters, dev, seed=C.SR_SEED_EVALUATED, verbose=verbose
    )
    rec = SR.denormalize(SR.apply(network0, tn, k), lo, esc)
    del network0
    mesh_sr = S.mask_to_mesh(
        S.segment_bone(rec, roi, spacing_f, stats=st_sr), grid
    )
    V = np.asarray(mesh_sr.vertices, float)
    e = S.vertex_error(V, pts_ref)
    border_sr = S.open_border(mesh_sr)  
    F = X.features(mesh_sr, grid, k, s_interp)
    times["sr"] = time.time() - t0
    return {
        "ident": ident,
        "V": V.astype(np.float32),
        "F": F,
        "e": e.astype(np.float32),
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
        "cleaning": {
            "reference": st_ref,
            "trilinear": st_tri,
            "sr": st_sr,
            "s_interp": st_synth,
        },
        "roi_thin_voxels": int(roi.sum()),
        "roi_thick_voxels": int(thick_roi.sum()),
        "info": info,
        "info_sr": info_sr,
        "grid": grid.as_dict(),
        "k": k,
        "times_s": {kk: round(v, 1) for kk, v in times.items()},
        "open_border": _border_stats(border_sr, F, mesh_ref),
        "area_mm2": {
            "reference": float(mesh_ref.area),
            "sr": float(mesh_sr.area),
            "trilinear": float(mesh_tri.area),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=0)
    ap.add_argument("--iters", type=int, default=C.SR_ITERS)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--cache-dir", default=C.A4_CACHE_THORAX)
    ap.add_argument(
        "--partition",
        default="everything",
        choices=("everything", "val", "test"),
        help="Section 3.6 uses only the 100 test cases (the validation cases were used in the SR selection)",
    )
    args = ap.parse_args()
    C.ensure_folders()
    dev = torch.device("cpu") if args.cpu else SR.pick_device(require_cuda=True)
    pairs = D.rplhr_pairs()
    if args.partition != "everything":
        pairs = [p for p in pairs if p[3] == args.partition]
    if args.n_cases > 0:
        pairs = pairs[: args.n_cases]
    from reliability.a4_run_foot import save_cache

    print("pick_device:", dev, "| cases:", len(pairs), flush=True)
    t0 = time.time()
    for i, (thin, thk, ident, part) in enumerate(pairs, 1):
        cp = os.path.join(args.cache_dir, ident + ".npz")
        if os.path.exists(cp):
            print("[%d/%d] %s (existing cache)" % (i, len(pairs), ident), flush=True)
            continue
        out = process_case(thin, thk, ident, dev, iters=args.iters)
        out["partition"] = part
        save_cache(out, args.cache_dir)
        print(
            "[%d/%d] %s ok | %d vertices | median e SR %.3f tri %.3f mm | faces %s | %s (%.0fs)"
            % (
                i,
                len(pairs),
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
