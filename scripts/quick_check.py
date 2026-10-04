"""Quick check on the sample foot shipped with the repository (data/sample/z002_foot.nii.gz).

Runs the reconstruction chain of the paper on one foot, without the uncertainty baselines and without the regressor:
thick-slice simulation -> trilinear baseline and self-supervised super-resolution -> bone mask -> surface ->
measured error e(v) against the thin-slice reference -> nine vertex descriptors.

With the default settings on a CUDA GPU the medians should reproduce the reference run
(reference_results/r31_reconstruction_error.json, case z002_foot.nii.gz). With --cpu or fewer iterations the
script only checks that the chain runs; the numbers will differ.

Usage: python scripts/quick_check.py [--iters 3046] [--cpu]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from reliability import a4_features as X
from reliability import a4_config as C
from reliability import a4_data as D
from reliability import a4_sr as SR
from reliability import a4_surface as S
from reliability.a4_run_foot import prepare_foot

CASE = "z002_foot.nii.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=C.SR_ITERS)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    filepath = os.path.join(ROOT, "data", "sample", CASE)
    dev = torch.device("cpu") if args.cpu else SR.pick_device(require_cuda=True)
    t0 = time.time()
    k = C.K_FOOT
    hu_t, thick, grid, spacing, thick_spacing, roi, thick_roi = prepare_foot(filepath, k)
    ref = S.mask_to_mesh(S.segment_bone(hu_t, roi, spacing), grid)
    pts_ref = S.sample_reference(ref, CASE)
    s_interp = X.interpolated_surface(
        X.thick_mask_sinterp(thick, thick_roi, thick_spacing), k, grid
    )

    tri = S.mask_to_mesh(
        S.segment_bone(D.interpolate_trilinear_z(thick, k), roi, spacing), grid
    )
    e_tri = S.vertex_error(np.asarray(tri.vertices, float), pts_ref)

    tn, lo, esc = SR.normalize(thick)
    network, _info = SR.train(
        SR.in_plane_slices(tn),
        k,
        args.iters,
        dev,
        seed=C.SR_SEED_EVALUATED,
        verbose=False,
    )
    rec = 0.5 * (
        SR.apply(network, tn, k, orientations=("yz",))
        + SR.apply(network, tn, k, orientations=("xz",))
    )
    sr = S.mask_to_mesh(
        S.segment_bone(SR.denormalize(rec, lo, esc), roi, spacing), grid
    )
    V = np.asarray(sr.vertices, float)
    e_sr = S.vertex_error(V, pts_ref)
    F = X.features(sr, grid, k, s_interp)

    out = {
        "case": CASE,
        "device": str(dev),
        "sr_iterations": args.iters,
        "vertices_sr": len(V),
        "median_error_sr_mm": float(np.median(e_sr)),
        "median_error_trilinear_mm": float(np.median(e_tri)),
        "median_d_shape_mm": float(
            np.median(F[:, C.FEATURE_NAMES.index("d_shape")])
        ),
        "features_shape": list(F.shape),
        "seconds": round(time.time() - t0, 1),
    }
    with open(
        os.path.join(ROOT, "reference_results", "r31_reconstruction_error.json"),
        encoding="utf-8",
    ) as f:
        expected = json.load(f)["per_case"][CASE]
    out["reference_median_error_sr_mm"] = expected["sr"]["median_mm"]
    out["reference_median_error_trilinear_mm"] = expected["trilinear"]["median_mm"]
    out["reference_vertices_sr"] = expected["sr"]["n_vertices"]
    print(json.dumps(out, indent=1))
    ok_tri = (
        abs(
            out["median_error_trilinear_mm"]
            - out["reference_median_error_trilinear_mm"]
        )
        < 1e-4
    )
    print("trilinear baseline matches the reference run:", ok_tri)
    if dev.type == "cuda" and args.iters == C.SR_ITERS:
        ok_sr = (
            abs(out["median_error_sr_mm"] - out["reference_median_error_sr_mm"]) < 1e-3
        )
        print("super-resolved surface matches the reference run:", ok_sr)
    return 0 if ok_tri else 1


if __name__ == "__main__":
    sys.exit(main())
