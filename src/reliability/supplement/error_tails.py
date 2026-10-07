"""Per-case tails of the reconstruction error that Section 3.1 reports beyond the median: mean, 90th percentile and
fraction of vertices above 2 mm of e(v), for the super-resolved and the trilinear meshes, with paired comparisons
(foot cohort and thoracic test cases). Reads the caches of the run. Outside the package hash (subfolder).

Output: results/rS_error_tails.json
Usage: python src/reliability/supplement/error_tails.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from reliability import a4_config as C  # noqa: E402
from reliability import a4_statistics as E  # noqa: E402

ABOVE_MM = 2.0


def per_case(folder: str) -> dict:
    out = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        z = np.load(f, allow_pickle=False)
        a, b = z["e"].astype(float), z["e_tri"].astype(float)
        out[os.path.basename(f)[:-4]] = {
            "sr": {"median_mm": float(np.median(a)), "mean_mm": float(a.mean()), "p90_mm": float(np.percentile(a, 90)), "above_2mm_pct": float(100.0 * (a > ABOVE_MM).mean())},
            "trilinear": {"median_mm": float(np.median(b)), "mean_mm": float(b.mean()), "p90_mm": float(np.percentile(b, 90)), "above_2mm_pct": float(100.0 * (b > ABOVE_MM).mean())},
        }
    return out


def cohort(folder: str) -> dict:
    pc = per_case(folder)
    out = {"n": len(pc), "per_case": pc}
    for m in ("median_mm", "mean_mm", "p90_mm", "above_2mm_pct"):
        sr = {h: v["sr"][m] for h, v in pc.items()}
        tri = {h: v["trilinear"][m] for h, v in pc.items()}
        out[m] = {"sr": E.summary(list(sr.values())), "trilinear": E.summary(list(tri.values())), "sr_minus_trilinear": E.paired_comparison(sr, tri)}
        out[m]["sr_minus_trilinear"]["sr_lower_in"] = int(sum(sr[h] < tri[h] for h in sr))
    return out


def main():
    out = {
        "description": "per-case median, mean, 90th percentile and fraction above %.0f mm of e(v); paired SR minus trilinear (two-sided Wilcoxon)" % ABOVE_MM,
        "foot": cohort(C.A4_CACHE_FOOT),
        "thorax_test": cohort(C.A4_CACHE_THORAX),
        "environment": C.environment_record(with_torch=False),
    }
    p = os.path.join(C.A4_RESULTS, "rS_error_tails.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for g in ("foot", "thorax_test"):
        for m in ("median_mm", "mean_mm", "p90_mm", "above_2mm_pct"):
            d = out[g][m]
            print("%s %s: SR %.3f tri %.3f | SR lower in %d/%d p=%.2g" % (g, m, d["sr"]["median"], d["trilinear"]["median"], d["sr_minus_trilinear"]["sr_lower_in"], out[g]["n"], d["sr_minus_trilinear"]["p_wilcoxon"]))
    print("->", p)


if __name__ == "__main__":
    main()
