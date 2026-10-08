"""Execution times recorded in the caches (per foot: reference surface, trilinear surface, super-resolution and its
surface, ensemble and MC-dropout uncertainty, Dropsembles) and in the result files (analyses), summarized across
cases. Outside the package hash (subfolder).

Output: results/rS_run_times.json
Usage: python src/reliability/supplement/run_times.py
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

STAGES = {"reference": ("reference", "referencia"), "trilinear": ("trilinear",), "sr": ("sr",), "ensemble": ("ensemble",), "dropsembles": ("dropsembles",)}
RESULT_FILES = ("r31_reconstruction_error.json", "r32_association.json", "r33_localization_calibration.json", "r341_intensity_uncertainty.json",
                "r342_surface_uncertainty.json", "r35_ablation.json", "r36_thorax.json", "r37_registration.json")


def summary(v):
    v = np.asarray(v, float)
    return {"n": int(len(v)), "median": float(np.median(v)), "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))], "min": float(v.min()), "max": float(v.max())}


def cohort(folder):
    per_stage, sr_train = {k: [] for k in STAGES}, []
    for f in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        meta = json.loads(str(np.load(f, allow_pickle=False)["meta"]))
        times = meta.get("times_s", meta.get("tempos_s", {}))
        for stage, keys in STAGES.items():
            for k in keys:
                if k in times:
                    per_stage[stage].append(float(times[k]))
        info = meta.get("info_sr", {})
        sec = info.get("seconds", info.get("segundos"))
        if sec is not None:
            sr_train.append(float(sec))
    out = {stage: summary(v) for stage, v in per_stage.items() if v}
    if sr_train:
        out["sr_training_only"] = summary(sr_train)
    return out


def main():
    out = {
        "description": "seconds per case recorded in the cache meta (stages of a4_run_foot / a4_run_thorax) and per analysis in the result files",
        "device": C.environment_record().get("gpu", "unknown"),
        "foot": cohort(C.A4_CACHE_FOOT),
        "thorax_test": cohort(C.A4_CACHE_THORAX),
        "analyses_seconds": {},
        "environment": C.environment_record(with_torch=False),
    }
    for name in RESULT_FILES:
        p = os.path.join(C.A4_RESULTS, name)
        if os.path.exists(p):
            j = json.load(open(p, encoding="utf-8"))
            sec = j.get("seconds", j.get("_traceability", {}).get("seconds"))
            if sec is not None:
                out["analyses_seconds"][name] = sec
    p = os.path.join(C.A4_RESULTS, "rS_run_times.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for g in ("foot", "thorax_test"):
        print(g, {k: round(v["median"], 1) for k, v in out[g].items()})
    print("analyses:", out["analyses_seconds"])
    print("->", p)


if __name__ == "__main__":
    main()
