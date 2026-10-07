"""Outcome of the EWC coefficient calibration of the adapted Dropsembles (Section 2.8.2), read from the foot caches:
for every subnetwork, whether the penalty/loss ratio converged to the target band, the ratio reached, and the number
of calibration probes used. Outside the package hash (subfolder).

Output: results/rS_ewc_calibration.json
Usage: python src/reliability/supplement/ewc_calibration_check.py
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

# key names of the cache meta (the published caches keep the names of the original run)
KEYS = {
    "converged": ("converged", "convergiu"),
    "ratio": ("effective_ratio_at_adaptation_it50", "razao_efetiva_na_adaptacao_it50"),
    "n_probes": ("n_probes", "n_sondas"),
    "lambda": ("lambda_ewc",),
    "entries": ("lambdas_ewc",),
}


def get(d, name):
    for k in KEYS[name]:
        if k in d:
            return d[k]
    raise KeyError(name)


def main():
    lo, hi = C.DS_EWC_FRACTION * (1 - C.DS_EWC_TOLERANCE), C.DS_EWC_FRACTION * (1 + C.DS_EWC_TOLERANCE)
    rows = []
    for f in sorted(glob.glob(os.path.join(C.A4_CACHE_FOOT, "*.npz"))):
        meta = json.loads(str(np.load(f, allow_pickle=False)["meta"]))
        for t, e in enumerate(get(meta, "entries")):
            rows.append({"case": os.path.basename(f)[:-4], "subnetwork": t, "lambda": float(get(e, "lambda")), "ratio": float(get(e, "ratio")), "n_probes": int(get(e, "n_probes")), "converged": bool(get(e, "converged"))})
    ratios = np.array([r["ratio"] for r in rows])
    probes = np.array([r["n_probes"] for r in rows])
    out = {
        "description": "EWC calibration per subnetwork: target ratio %.2f, band [%.3f, %.3f], at most %d probes" % (C.DS_EWC_FRACTION, lo, hi, C.DS_EWC_MAX_PROBES),
        "n_subnetworks": len(rows),
        "n_converged": int(sum(r["converged"] for r in rows)),
        "n_outside_band": int(((ratios < lo) | (ratios > hi)).sum()),
        "ratio": {"min": float(ratios.min()), "median": float(np.median(ratios)), "max": float(ratios.max())},
        "probes": {"min": int(probes.min()), "median": float(np.median(probes)), "max": int(probes.max())},
        "per_subnetwork": rows,
        "environment": C.environment_record(with_torch=False),
    }
    p = os.path.join(C.A4_RESULTS, "rS_ewc_calibration.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print("subnetworks %d, converged %d, outside band %d, ratio %.3f-%.3f, probes %d-%d" % (len(rows), out["n_converged"], out["n_outside_band"], ratios.min(), ratios.max(), probes.min(), probes.max()))
    print("->", p)


if __name__ == "__main__":
    main()
