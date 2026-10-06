"""Post-hoc comparison of the adopted super-resolution configuration (selected on the RPLHR-CT validation cases) with
the original configuration of the design (learning rate 1e-3, 2500 iterations), case by case, on the same surfaces
pipeline: per-case median reconstruction error e(v) (super-resolved and trilinear meshes) and the number of cases in
which the super-resolved mesh has the lower error. Outside the package hash (subfolder).

The caches of the original configuration are produced by the main pipeline with ``SR_LR = 1e-3`` and
``SR_ITERS = 2500`` (a4_run_foot.py / a4_run_thorax.py into a separate A4_OUTPUT_DIR); under the exact error
definition they must hold the same target as the current caches (see rebuild_caches_exact.py). The vertex-level
correlation is compared only when the result files of both runs exist, because it depends on the regressor and
on the number of training cases of each run.

Output: results/rS_original_sr_config.json
Usage: python src/reliability/supplement/compare_original_sr_config.py --original-foot DIR --original-thorax DIR
       [--original-results DIR] [--label "lr 1e-3, 2500 iterations"]
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
from reliability import a4_statistics as E  # noqa: E402


def per_case_median(folder: str, key: str) -> dict:
    out = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        z = np.load(f, allow_pickle=False)
        out[os.path.basename(f)[:-4]] = float(np.median(z[key].astype(np.float64)))
    return out


def error_definition(folder: str) -> str:
    f = sorted(glob.glob(os.path.join(folder, "*.npz")))[0]
    meta = json.loads(str(np.load(f, allow_pickle=False)["meta"]))
    return str(meta.get("error_definition", "sampled (Eq. 5 against the sampled reference points)"))


def paired(current: dict, original: dict, n_expected: int) -> dict:
    ids = sorted(set(current) & set(original))
    if len(ids) != n_expected:
        raise RuntimeError("%d common cases, expected %d" % (len(ids), n_expected))
    cur = {h: current[h] for h in ids}
    orig = {h: original[h] for h in ids}
    out = E.paired_comparison(cur, orig)  # current minus original
    out["current"] = E.summary(list(cur.values()))
    out["original"] = E.summary(list(orig.values()))
    return out


def sr_vs_tri(folder: str, ids) -> dict:
    sr, tri = per_case_median(folder, "e"), per_case_median(folder, "e_tri")
    ids = sorted(ids)
    out = E.paired_comparison({h: sr[h] for h in ids}, {h: tri[h] for h in ids})
    out["sr_lower_than_tri_in"] = int(sum(sr[h] < tri[h] for h in ids))
    return out


def rho_per_case(results_dir: str, name: str, key: str) -> dict | None:
    p = os.path.join(results_dir, name)
    if not os.path.exists(p):
        return None
    j = json.load(open(p, encoding="utf-8"))
    return {h: v["rho_vertex"] for h, v in j[key].items()}


def cohort(current_cache: str, original_cache: str, n: int, results_cur: str, results_orig: str | None, r_name: str) -> dict:
    cur = per_case_median(current_cache, "e")
    orig = per_case_median(original_cache, "e")
    ids = sorted(set(cur) & set(orig))
    out = {
        "n": len(ids),
        "e_median_sr_mm": paired(cur, orig, n),
        "e_median_tri_mm": paired(per_case_median(current_cache, "e_tri"), per_case_median(original_cache, "e_tri"), n),
        "sr_vs_tri_current": sr_vs_tri(current_cache, ids),
        "sr_vs_tri_original": sr_vs_tri(original_cache, ids),
        "error_definition_current": error_definition(current_cache),
        "error_definition_original": error_definition(original_cache),
    }
    if results_orig:
        a = rho_per_case(results_cur, r_name, "per_case_sr")
        b = rho_per_case(results_orig, r_name, "per_case_sr")
        if a and b:
            out["rho_vertex_sr"] = paired(a, b, n)
            out["rho_note"] = "depends on the regressor of each run (training cases per fold may differ)"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original-foot", required=True)
    ap.add_argument("--original-thorax", required=True)
    ap.add_argument("--original-results", default="")
    ap.add_argument("--label", default="learning rate 1e-3, 2500 iterations (original design)")
    args = ap.parse_args()
    res = {
        "description": "adopted SR configuration (SR_LR=%g, SR_ITERS=%d) minus the original configuration (%s), paired by case; "
        "two-sided Wilcoxon on the per-case median error" % (C.SR_LR, C.SR_ITERS, args.label),
        "foot": cohort(C.A4_CACHE_FOOT, args.original_foot, C.N_FEET, C.A4_RESULTS, args.original_results or None, "r32_association.json"),
        "thorax_test": cohort(C.A4_CACHE_THORAX, args.original_thorax, C.N_THORAX_TEST, C.A4_RESULTS, args.original_results or None, "r36_thorax.json"),
        "environment": C.environment_record(with_torch=False),
    }
    p = os.path.join(C.A4_RESULTS, "rS_original_sr_config.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    for g in ("foot", "thorax_test"):
        d = res[g]["e_median_sr_mm"]
        print(g, "n=%d current %.4f original %.4f diff %+.4f p=%.2g | SR<tri: current %d, original %d" % (
            res[g]["n"], d["current"]["median"], d["original"]["median"], d["difference"]["median"], d["p_wilcoxon"],
            res[g]["sr_vs_tri_current"]["sr_lower_than_tri_in"], res[g]["sr_vs_tri_original"]["sr_lower_than_tri_in"]))
    print("->", p)


if __name__ == "__main__":
    main()
