"""Region-level association of the field for other numbers of k-means regions (Section 2.9.2 uses K = 25): the
leave-one-case-out predictions of the memo are kept, only the partition of the surface changes. Reports, per K, the
median across cases of the Spearman correlation between regional means of the prediction and of the error, for the
foot cohort (super-resolved meshes). Outside the package hash (subfolder).

Output: results/rS_region_count.json
Usage: python src/reliability/supplement/region_count_sensitivity.py [--k 10,25,50]
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


def memo(label: str) -> dict:
    sel = json.load(open(os.path.join(C.A4_RESULTS, "rf_selection_%s_%s.json" % (label, C.RF_SEARCH_SAMPLER)), encoding="utf-8"))
    z = np.load(os.path.join(C.A4_RESULTS, "_lofo_%s_%s.npz" % (label, sel["key"])), allow_pickle=False)
    return {k[5:]: z[k].astype(float) for k in z.files if k.startswith("pred_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="10,25,50")
    args = ap.parse_args()
    Ks = [int(x) for x in args.k.split(",")]
    pred = memo("rf9_sr")
    rho = {K: {} for K in Ks}
    for f in sorted(glob.glob(os.path.join(C.A4_CACHE_FOOT, "*.npz"))):
        h = os.path.basename(f)[:-4]
        z = np.load(f, allow_pickle=False)
        V, e = z["V"].astype(float), z["e"].astype(float)
        for K in Ks:
            lab = E.regions(V, K)
            rho[K][h] = float(E.spearman(E.regional_means(pred[h], lab, K), E.regional_means(e, lab, K)))
        print(h[:4], {K: round(rho[K][h], 3) for K in Ks}, flush=True)
    r32 = json.load(open(os.path.join(C.A4_RESULTS, "r32_association.json"), encoding="utf-8"))["per_case_sr"]
    out = {
        "description": "region-level Spearman correlation of the field (foot, super-resolved) for several K; LOFO predictions unchanged",
        "K": {str(K): {"summary": E.summary(list(rho[K].values())), "per_case": rho[K]} for K in Ks},
        "check_K25_equals_r32": bool(25 in Ks and max(abs(rho[25][h] - r32[h]["rho_region"]) for h in rho[25]) < 1e-9),
        "environment": C.environment_record(with_torch=False),
    }
    p = os.path.join(C.A4_RESULTS, "rS_region_count.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    for K in Ks:
        s = out["K"][str(K)]["summary"]
        print("K=%d: rho_region %.3f (%.3f-%.3f)" % (K, s["median"], *s["iqr"]))
    print("K=25 reproduces r32:", out["check_K25_equals_r32"], "->", p)


if __name__ == "__main__":
    main()
