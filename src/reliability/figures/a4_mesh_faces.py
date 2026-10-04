"""Re-extracts the SR mesh (with faces) of a foot through the SAME chain as a4_run_foot.process_foot (prepare_foot ->
SR trained with the evaluated seed -> segmentation -> mask_to_mesh) and checks it against the vertices of the Phase 2
cache. Only to render the shaded surface in the figures (route B); writes nothing to foot_cache. Output:
figs/data/meshes/<foot>.npz with V (regen), faces, idx_cache (cache vertex closest to each regenerated vertex) and
the check distances.

Usage: python src/reliability/figures/a4_mesh_faces.py --foot z002_foot.nii.gz [--foot z001_foot.nii.gz]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import a4_config as C
import a4_sr as SR
import a4_surface as S
from a4_run_foot import prepare_foot

FIGS = os.path.join(
    C.ROOT, "output", "figures", "figs"
)
OUTPUT = os.path.join(FIGS, "data", "meshes")


def regenerate(ident, dev):
    listing = json.load(open(C.FOOT_LIST_N48, encoding="utf-8"))
    filepath = [p for p in listing if os.path.basename(p) == ident][0]
    t0 = time.time()
    hu_t, thick, grid, spacing, thick_spacing, roi, thick_roi = prepare_foot(filepath, C.K_FOOT)
    del hu_t
    tn, lo, esc = SR.normalize(thick)
    slices = SR.in_plane_slices(tn)
    network, info = SR.train(
        slices, C.K_FOOT, C.SR_ITERS, dev, seed=C.SR_SEED_EVALUATED, verbose=False
    )
    rec = 0.5 * (
        SR.apply(network, tn, C.K_FOOT, orientations=("yz",))
        + SR.apply(network, tn, C.K_FOOT, orientations=("xz",))
    )
    rec = SR.denormalize(rec, lo, esc)
    del network
    mesh = S.mask_to_mesh(S.segment_bone(rec, roi, spacing), grid)
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, np.int64)
    # check against the foot cache (same chain; SR deterministic on the GPU)
    z = np.load(os.path.join(C.A4_CACHE_FOOT, ident + ".npz"), allow_pickle=False)
    Vc = z["V"].astype(float)
    d_regen_to_cache, idx_cache = cKDTree(Vc).query(V)
    d_cache_to_regen, _ = cKDTree(V).query(Vc)
    identical = len(V) == len(Vc) and np.allclose(V, Vc, atol=1e-3)
    conf = {
        "foot": ident,
        "n_vertices_regen": int(len(V)),
        "n_vertices_cache": int(len(Vc)),
        "n_faces": int(len(F)),
        "identical_vertices_in_same_order": bool(identical),
        "dist_regen_to_cache_mm": {
            "max": float(d_regen_to_cache.max()),
            "p99": float(np.percentile(d_regen_to_cache, 99)),
            "median": float(np.median(d_regen_to_cache)),
        },
        "dist_cache_to_regen_mm": {
            "max": float(d_cache_to_regen.max()),
            "p99": float(np.percentile(d_cache_to_regen, 99)),
        },
        "code_hash_reliability": C.code_hash() if hasattr(C, "code_hash") else None,
        "cache_hash": z["meta"]
        and json.loads(str(z["meta"]))["environment"].get("code_hash_reliability"),
        "sr_iters": int(C.SR_ITERS),
        "time_s": round(time.time() - t0, 1),
        "info_sr": {k: v for k, v in info.items() if isinstance(v, (int, float, str))}
        if isinstance(info, dict)
        else None,
    }
    os.makedirs(OUTPUT, exist_ok=True)
    np.savez_compressed(
        os.path.join(OUTPUT, ident + ".npz"),
        V=V.astype(np.float32),
        faces=F.astype(np.int32),
        idx_cache=idx_cache.astype(np.int32),
        d_regen_to_cache=d_regen_to_cache.astype(np.float32),
        verification=json.dumps(conf, ensure_ascii=False),
    )
    with open(
        os.path.join(OUTPUT, ident + ".verification.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(conf, f, indent=1, ensure_ascii=False)
    print(json.dumps(conf, ensure_ascii=False), flush=True)
    return conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--foot", action="append", required=True)
    args = ap.parse_args()
    dev = SR.pick_device(require_cuda=True)
    for ident in args.foot:
        regenerate(ident, dev)


if __name__ == "__main__":
    main()
