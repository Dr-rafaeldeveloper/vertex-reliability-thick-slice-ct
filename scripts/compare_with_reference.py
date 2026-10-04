"""Compare the result files produced by a run with the reference results of the paper.

Every numeric value of each result JSON (summary statistics and per-case values) is compared with the file of the
same name in reference_results/. Provenance fields (dates, timings, environment, code hash, paths) are ignored.

Usage: python scripts/compare_with_reference.py [--tol 1e-6]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from reliability import a4_config as C

REF = os.path.join(ROOT, "reference_results")
IGNORE = (
    "_traceability",
    "environment",
    "seconds",
    "time",
    "data",
    "hash",
    "description",
    "section",
    "failure",
    "file",
)


def leaves(o, filepath=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if any(k.startswith(p) or p in k for p in IGNORE):
                continue
            yield from leaves(v, filepath + "/" + k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, filepath + "[%d]" % i)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield filepath, float(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()
    all_ok = True
    for name in sorted(os.listdir(REF)):
        new = os.path.join(C.A4_RESULTS, name)
        if not name.endswith(".json") or not os.path.exists(new):
            print("%-40s not produced by this run" % name)
            continue
        with open(os.path.join(REF, name), encoding="utf-8") as f:
            a = dict(leaves(json.load(f)))
        with open(new, encoding="utf-8") as f:
            b = dict(leaves(json.load(f)))
        common = sorted(set(a) & set(b))
        diff = [
            (abs(a[k] - b[k]), k)
            for k in common
            if not (math.isnan(a[k]) and math.isnan(b[k]))
        ]
        worst = max(diff) if diff else (0.0, "")
        ok = worst[0] <= args.tol and len(common) == len(a)
        all_ok &= ok
        print(
            "%-40s %6d values, %d missing, max |diff| = %.3g %s%s"
            % (
                name,
                len(common),
                len(a) - len(common),
                worst[0],
                "OK" if ok else "DIFFERS",
                "" if ok else "  at " + worst[1],
            )
        )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
