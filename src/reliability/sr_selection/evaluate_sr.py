"""Evaluator of the SR selection (main environment, GPU). For one SR configuration (lr, iters) (§2.4), trains the network per
case on the RPLHR-CT VALIDATION cases, extracts the surface (§2.5) and measures e(v) against the 120 000 points of
the 1 mm reference (§2.6). Objective = median across cases of the per-case median of e(v)
(sr_config.SR_SEARCH_AGGREGATION).

The geometric chain that does not depend on (lr, iters) — regridded reference, ROI, reference points, trilinear
baseline — is computed ONCE per case (`--prepare`) and stored in `_sr_selection/cache_val/<case>.npz` (~16 MB/case),
with the package hash checked at every use. Everything else is identical to `a4_run_thorax.process_case` (same
functions, same arguments; seed SR_SEED_EVALUATED; default orientations and normalization).
A case that fails (divergent SR -> empty mask / invalid mesh) receives e_median = e_tri_median of the case
(sr_config.FAILURE_RULE, fixed a priori) and is recorded in `per_case[ident]["failure"]`.

Usage: python evaluate_sr.py --prepare [--n-cases N]
 python evaluate_sr.py --lr 1e-3 --iters 2500 --output eval.json [--n-cases N] [--cpu]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from reliability import a4_config as C  # noqa: E402
from reliability import a4_data as D  # noqa: E402
from reliability import a4_sr as SR  # noqa: E402
from reliability import a4_surface as S  # noqa: E402
from reliability.a4_grid import Grid  # noqa: E402
from reliability.sr_selection import sr_config as CS  # noqa: E402

FOLDER = os.path.join(C.A4_OUT, "_sr_selection")
CACHE_VAL = os.path.join(FOLDER, "cache_val")


def validation_cases(pairs=None):
    """Cases of the RPLHR-CT validation partition (text: 50), in the order of a4_data.rplhr_pairs."""
    real = pairs is None
    pairs = D.rplhr_pairs() if real else pairs
    assert all(p[3] in ("val", "test") for p in pairs), (
        "undetermined partition in rplhr_pairs"
    )
    sel = [p for p in pairs if p[3] == CS.SR_SEARCH_PARTITION]
    ids = [p[2] for p in sel]
    assert len(ids) == len(set(ids)), "repeated identifiers in the validation set"
    if real:
        assert len(sel) == C.N_THORAX_VAL, (len(sel), C.N_THORAX_VAL)
    return sel


def prepare_case(thin: str, thk: str, ident: str, folder: str = CACHE_VAL) -> str:
    """Part of the chain independent of (lr, iters): thick volume, ROI, grid, reference points and e_tri."""
    from reliability.a4_run_thorax import prepare_thorax

    cp = os.path.join(folder, ident + ".npz")
    if os.path.exists(cp):
        return cp
    os.makedirs(folder, exist_ok=True)
    t0 = time.time()
    hu_ref, thick, grid, spacing_f, thick_spacing, roi, thick_roi, k, info = prepare_thorax(
        thin, thk
    )
    st_ref, st_tri = {}, {}
    mesh_ref = S.mask_to_mesh(
        S.segment_bone(hu_ref, roi, spacing_f, stats=st_ref), grid
    )  # §2.5
    pts_ref = S.sample_reference(
        mesh_ref, ident
    )  # §2.6 (float64, as in process_case)
    tri = D.interpolate_trilinear_z(thick, k)
    mesh_tri = S.mask_to_mesh(
        S.segment_bone(tri, roi, spacing_f, stats=st_tri), grid
    )
    e_tri = S.vertex_error(np.asarray(mesh_tri.vertices, float), pts_ref)
    meta = {
        "ident": ident,
        "partition": CS.SR_SEARCH_PARTITION,
        "k": int(k),
        "grid": grid.as_dict(),
        "info": info,
        "cleaning": {"reference": st_ref, "trilinear": st_tri},
        "faces_ref": int(len(mesh_ref.faces)),
        "e_tri_median": float(np.median(e_tri)),
        "n_vertices_tri": int(len(e_tri)),
        "thick_dtype": str(thick.dtype),
        "seconds": round(time.time() - t0, 1),
        "environment": C.environment_record(with_torch=False),
        "sr_selection_hash": CS.sr_selection_hash(),
    }
    tmp = cp + ".tmp.npz"
    np.savez_compressed(
        tmp,
        thick=thick,
        roi=roi,
        pts_ref=pts_ref,
        meta=json.dumps(meta, default=_json_default),
    )
    os.replace(tmp, cp)
    return cp


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_cache(cp: str):
    """Reads the cache and checks the package hash against the current one: reference/ROI from old code do not pass."""
    z = np.load(cp, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    h_cache, h_current = meta["environment"]["code_hash_reliability"], C.code_hash()
    if h_cache != h_current:
        raise RuntimeError(
            "cache_val %s generated under hash %s; current package %s: regenerate it (--prepare)"
            % (os.path.basename(cp), h_cache, h_current)
        )
    g = meta["grid"]
    grid = Grid(g["spacing_mm"], g["origin_mm"], g["direction"], g["shape_zyx"])
    return z["thick"], z["roi"], z["pts_ref"].astype(float), int(meta["k"]), grid, meta


def evaluate_case(cp: str, lr: float, iters: int, dev) -> dict:
    """Trains/applies the SR with (lr, iters) and measures e(v) of the SR surface against pts_ref (same functions as
process_case)."""
    thick, roi, pts_ref, k, grid, meta = load_cache(cp)
    spacing_f = tuple(grid.spacing)
    t0 = time.time()
    tn, lo, esc = SR.normalize(thick)
    slices = SR.in_plane_slices(tn)
    network, info = SR.train(
        slices,
        k,
        int(iters),
        dev,
        seed=C.SR_SEED_EVALUATED,
        lr=float(lr),
        verbose=False,
    )
    rec = SR.denormalize(SR.apply(network, tn, k), lo, esc)
    del network
    mesh = S.mask_to_mesh(S.segment_bone(rec, roi, spacing_f), grid)
    e = S.vertex_error(np.asarray(mesh.vertices, float), pts_ref)
    return {
        "e_median": float(np.median(e)),
        "e_mean": float(np.mean(e)),
        "e_p90": float(np.percentile(e, 90)),
        "n_vertices": int(len(e)),
        "e_tri_median": meta["e_tri_median"],
        "last_l1_loss": float(info["loss_history"][-1][1])
        if info["loss_history"]
        else None,
        "n_iterations_run": int(info["n_iterations_run"]),
        "seconds": round(time.time() - t0, 1),
        "failure": None,
        "code_hash_cache": meta["environment"]["code_hash_reliability"],
        "sr_selection_hash_cache": meta["sr_selection_hash"],
    }


def evaluate_case_with_rule(cp: str, lr: float, iters: int, dev) -> dict:
    """sr_config.FAILURE_RULE: exception in the case -> e_median = e_tri_median of the case, failure recorded."""
    try:
        return evaluate_case(cp, lr, iters, dev)
    except Exception as exc:  # noqa: BLE001 — any failure of the case becomes 'not better than the baseline'
        _, _, _, _, _, meta = load_cache(cp)
        return {
            "e_median": meta["e_tri_median"],
            "e_mean": None,
            "e_p90": None,
            "n_vertices": 0,
            "e_tri_median": meta["e_tri_median"],
            "last_l1_loss": None,
            "n_iterations_run": None,
            "seconds": None,
            "failure": "%s: %s" % (type(exc).__name__, str(exc)[:300]),
            "traceback": traceback.format_exc()[-1500:],
            "code_hash_cache": meta["environment"]["code_hash_reliability"],
            "sr_selection_hash_cache": meta["sr_selection_hash"],
        }


def objective(per_case: dict) -> float:
    """Median across cases of the per-case median of e(v) (sr_config.SR_SEARCH_AGGREGATION)."""
    return float(np.median([v["e_median"] for v in per_case.values()]))


def evaluate_config(
    lr: float, iters: int, cases, dev, folder: str = CACHE_VAL, verbose=True
) -> dict:
    t0 = time.time()
    per_case = {}
    for i, (thin, thk, ident, _) in enumerate(cases, 1):
        cp = prepare_case(thin, thk, ident, folder)
        per_case[ident] = evaluate_case_with_rule(cp, lr, iters, dev)
        if verbose:
            r = per_case[ident]
            print(
                "  [%d/%d] %s e_med SR %.3f | tri %.3f mm | %d vert | %s s%s"
                % (
                    i,
                    len(cases),
                    ident,
                    r["e_median"],
                    r["e_tri_median"],
                    r["n_vertices"],
                    r["seconds"],
                    " | FAILURE " + r["failure"] if r["failure"] else "",
                ),
                flush=True,
            )
    med = np.array([v["e_median"] for v in per_case.values()])
    tri = np.array([v["e_tri_median"] for v in per_case.values()])
    hashes_cache = {
        "code_hash_reliability": sorted(
            {v["code_hash_cache"] for v in per_case.values()}
        ),
        "sr_selection_hash": sorted(
            {v["sr_selection_hash_cache"] for v in per_case.values()}
        ),
    }
    return {
        "lr": float(lr),
        "iters": int(iters),
        "objective_mm": objective(per_case),
        "aggregated": {
            "n_cases": len(per_case),
            "n_failures": int(sum(1 for v in per_case.values() if v["failure"])),
            "e_median_iqr": [
                float(np.percentile(med, 25)),
                float(np.percentile(med, 75)),
            ],
            "e_median_mean": float(med.mean()),
            "sr_better_than_tri_in": int((med < tri).sum()),
            "tri_objective_mm": float(np.median(tri)),
        },
        "cases": sorted(per_case),
        "per_case": per_case,
        "seconds": round(time.time() - t0, 1),
        "aggregation": CS.SR_SEARCH_AGGREGATION,
        "failure_rule": CS.FAILURE_RULE,
        "sr_seed": C.SR_SEED_EVALUATED,
        "environment": C.environment_record(),
        "sr_selection_hash": CS.sr_selection_hash(),
        "hashes_cache": hashes_cache,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prepare", action="store_true", help="only builds cache_val (references)"
    )
    ap.add_argument("--lr", type=float)
    ap.add_argument("--iters", type=int)
    ap.add_argument("--output", default="")
    ap.add_argument("--n-cases", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    C.ensure_folders()
    cases = validation_cases()
    if args.n_cases > 0:
        cases = cases[: args.n_cases]
    if args.prepare:
        t0 = time.time()
        for i, (thin, thk, ident, _) in enumerate(cases, 1):
            cp = prepare_case(thin, thk, ident)
            print(
                "[%d/%d] %s -> %s (%.0fs)"
                % (i, len(cases), ident, os.path.basename(cp), time.time() - t0),
                flush=True,
            )
        return
    assert args.lr is not None and args.iters is not None and args.output, (
        "--lr, --iters and --output"
    )
    import torch

    dev = torch.device("cpu") if args.cpu else SR.pick_device(require_cuda=True)
    print(
        "evaluating lr=%g iters=%d on %d cases (%s)"
        % (args.lr, args.iters, len(cases), dev),
        flush=True,
    )
    r = evaluate_config(args.lr, args.iters, cases, dev)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1, ensure_ascii=False)
    os.replace(tmp, args.output)
    print(
        "objective %.4f mm (tri %.4f; failures %d) in %.0fs -> %s"
        % (
            r["objective_mm"],
            r["aggregated"]["tri_objective_mm"],
            r["aggregated"]["n_failures"],
            r["seconds"],
            args.output,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
