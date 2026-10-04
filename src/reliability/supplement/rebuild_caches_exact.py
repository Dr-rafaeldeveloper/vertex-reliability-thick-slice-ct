"""Rebuild the per-case caches with the exact point-to-triangle error as target.

The super-resolved and trilinear surfaces, the descriptors and the uncertainty baselines do not depend on how the
error is measured, so they are copied from the caches of the reference run. Only two things change:

 1. e(v) and e_tri(v) become the exact distance from the vertex to the reference surface (point to triangle),
    instead of the distance to the nearest of the 120,000 sampled points (Eq. 5). For the super-resolved
    surface the closest point q(v) was already stored; for the trilinear surface the reference surface is rebuilt
    from the thin-slice volume with the same chain and the distance computed with the bounded search of
    supplement/exact_error.py. Vertices farther than EXACT_MAX_MM from the sampled points keep the Eq. 5 value.
 2. vertices lying outside the CT volume (mesh artefacts, see exact_error.py) are removed from every per-vertex
    array of the case.

Usage: python src/reliability/supplement/rebuild_caches_exact.py --old-foot DIR --old-thorax DIR [--only foot,thorax]
Output: the cache folders of the current configuration (set A4_OUTPUT_DIR to a new location first).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from reliability import a4_config as C  # noqa: E402
from reliability import a4_data as D  # noqa: E402
from reliability import a4_surface as S  # noqa: E402
from reliability.a4_run_foot import prepare_foot, save_cache  # noqa: E402
from reliability.a4_run_thorax import prepare_thorax  # noqa: E402
from reliability.supplement.exact_error import EXACT_MAX_MM, exact_bounded, load_meta, outside_volume  # noqa: E402


def rebuild(path, mesh_ref, folder, check_q=False):
    z = np.load(path, allow_pickle=True)
    meta = load_meta(z)
    ident = os.path.basename(path)[:-4]
    V, e_old = z["V"].astype(float), z["e"].astype(float)
    Vt, et_old = z["V_tri"].astype(float), z["e_tri"].astype(float)
    if "q" in z.files:
        e_new = np.linalg.norm(V - z["q"].astype(float), axis=1)
        if check_q:
            chk = exact_bounded(V, e_old, mesh_ref)
            near = e_old <= EXACT_MAX_MM
            assert np.abs(chk[near] - e_new[near]).max() < 1e-3, ident
    else:
        e_new = exact_bounded(V, e_old, mesh_ref)
    et_new = exact_bounded(Vt, et_old, mesh_ref)
    assert np.all(e_new <= e_old + 1e-3) and np.all(et_new <= et_old + 1e-3), ident
    keep, keep_t = ~outside_volume(V, meta), ~outside_volume(Vt, meta)
    n, nt = len(V), len(Vt)
    out = {"ident": ident}
    for k in z.files:
        if k == "meta":
            continue
        a = z[k]
        if k in ("V_tri", "F_tri", "e_tri"):
            a = (et_new.astype(np.float32) if k == "e_tri" else a)[keep_t]
        elif k == "e":
            a = e_new.astype(np.float32)[keep]
        elif a.ndim >= 1 and a.shape[0] == n:
            a = a[keep]
        elif a.ndim == 2 and a.shape[1] == n:
            a = a[:, keep]
        out[k] = a
    meta_old = dict(meta)
    meta_old.pop("environment", None)
    out.update(meta_old)
    out["error_definition"] = "exact point-to-triangle distance to the reference surface (supplement/rebuild_caches_exact.py)"
    out["rebuilt_from"] = {
        "cache": os.path.basename(path),
        "code_hash_source": meta.get("environment", {}).get("code_hash_reliability")
        or meta.get("environment", {}).get("hash_codigo_artigo4"),
        "n_vertices_sr_before": int(n),
        "n_removed_outside_volume_sr": int((~keep).sum()),
        "n_vertices_tri_before": int(nt),
        "n_removed_outside_volume_tri": int((~keep_t).sum()),
        "n_sr_kept_eq5_value_beyond_exact_max": int((e_old > EXACT_MAX_MM).sum()),
        "n_tri_kept_eq5_value_beyond_exact_max": int((et_old > EXACT_MAX_MM).sum()),
        "median_eq5_mm": float(np.median(e_old)),
        "median_exact_mm": float(np.median(e_new)),
    }
    save_cache(out, folder)
    return out["rebuilt_from"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-foot", required=True)
    ap.add_argument("--old-thorax", required=True)
    ap.add_argument("--only", default="foot,thorax")
    args = ap.parse_args()
    only = set(args.only.split(","))
    if "foot" in only:
        for i, path in enumerate(sorted(glob.glob(os.path.join(args.old_foot, "*.npz")))):
            ident = os.path.basename(path)[:-4]
            dst = os.path.join(C.A4_CACHE_FOOT, ident + ".npz")
            if os.path.exists(dst):
                continue
            t0 = time.time()
            hu_t, _thick, grid, spacing, _ts, roi, _tr = prepare_foot(os.path.join(C.FOOT_FOLDER, ident), C.K_FOOT)
            mesh_ref = S.mask_to_mesh(S.segment_bone(hu_t, roi, spacing), grid)
            r = rebuild(path, mesh_ref, C.A4_CACHE_FOOT, check_q=(i < 2))
            print("foot %s | eq5 %.3f -> exact %.3f | removed %d/%d | %.0f s" % (
                ident[:4], r["median_eq5_mm"], r["median_exact_mm"], r["n_removed_outside_volume_sr"],
                r["n_removed_outside_volume_tri"], time.time() - t0), flush=True)
    if "thorax" in only:
        pairs = {name: (thin, thk) for thin, thk, name, _p in D.rplhr_pairs()}
        for path in sorted(glob.glob(os.path.join(args.old_thorax, "*.npz"))):
            ident = os.path.basename(path)[:-4]
            dst = os.path.join(C.A4_CACHE_THORAX, ident + ".npz")
            if os.path.exists(dst):
                continue
            t0 = time.time()
            thin, thk = pairs[ident]
            hu_ref, _thick, grid, spacing_f, _ts, roi, _tr, _k, _info = prepare_thorax(thin, thk)
            mesh_ref = S.mask_to_mesh(S.segment_bone(hu_ref, roi, spacing_f), grid)
            r = rebuild(path, mesh_ref, C.A4_CACHE_THORAX)
            print("thorax %s | eq5 %.3f -> exact %.3f | removed %d/%d | %.0f s" % (
                ident, r["median_eq5_mm"], r["median_exact_mm"], r["n_removed_outside_volume_sr"],
                r["n_removed_outside_volume_tri"], time.time() - t0), flush=True)
    with open(os.path.join(C.A4_OUT, "rebuild_caches_exact.json"), "w", encoding="utf-8") as f:
        json.dump({"old_foot": args.old_foot, "old_thorax": args.old_thorax, "environment": C.environment_record()}, f, indent=1)
    print("done")


if __name__ == "__main__":
    main()
