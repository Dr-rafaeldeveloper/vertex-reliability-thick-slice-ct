"""Constant baseline equal to the MEDIAN of the training error (Section 2.9.4 uses the mean): mean absolute error per
foot of the field, of the mean constant and of the median constant, feet in which the field has the lower error, and
paired Wilcoxon tests. Uses the leave-one-foot-out predictions of the full model (results/_lofo_rf9_sr_<key>.npz, the
same memo as r32/r33). Outside the package hash (subfolder).

Output: results/rS_median_baseline.json
Usage: python src/reliability/supplement/median_constant_baseline.py
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


def main():
    sel = json.load(open(os.path.join(C.A4_RESULTS, "rf_selection_rf9_sr_%s.json" % C.RF_SEARCH_SAMPLER), encoding="utf-8"))
    memo = os.path.join(C.A4_RESULTS, "_lofo_rf9_sr_%s.npz" % sel["key"])
    z = np.load(memo, allow_pickle=False)
    pred = {k[5:]: z[k].astype(float) for k in z.files if k.startswith("pred_")}
    const_mean = {k[6:]: float(z[k]) for k in z.files if k.startswith("const_")}
    e = {
        os.path.basename(p)[:-4]: np.load(p, allow_pickle=False)["e"].astype(float)
        for p in sorted(glob.glob(os.path.join(C.A4_CACHE_FOOT, "*.npz")))
    }
    ids = sorted(e)
    if set(ids) != set(pred):
        raise RuntimeError("the LOFO memo does not cover the caches")
    mae_model, mae_mean, mae_median = {}, {}, {}
    for h in ids:
        train = np.concatenate([e[j] for j in ids if j != h])
        med = float(np.median(train))
        mae_model[h] = float(np.mean(np.abs(pred[h] - e[h])))
        mae_mean[h] = float(np.mean(np.abs(const_mean[h] - e[h])))
        mae_median[h] = float(np.mean(np.abs(med - e[h])))
    out = {
        "description": "MAE per foot of the field (LOFO) against a constant equal to the mean (Section 2.9.4) or to the "
        "median of the training error; paired differences field minus constant, two-sided Wilcoxon",
        "n": len(ids),
        "mae_model": E.summary(list(mae_model.values())),
        "mae_constant_mean": E.summary(list(mae_mean.values())),
        "mae_constant_median": E.summary(list(mae_median.values())),
        "model_minus_constant_mean": E.paired_comparison(mae_model, mae_mean),
        "model_minus_constant_median": E.paired_comparison(mae_model, mae_median),
        "lofo_memo": os.path.basename(memo),
        "environment": C.environment_record(with_torch=False),
    }
    for k in ("model_minus_constant_mean", "model_minus_constant_median"):
        out[k]["model_lower_in"] = int(sum(mae_model[h] < (mae_mean if k.endswith("mean") else mae_median)[h] for h in ids))
    p = os.path.join(C.A4_RESULTS, "rS_median_baseline.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for k in ("mae_model", "mae_constant_mean", "mae_constant_median"):
        print(k, "%.4f" % out[k]["median"], ["%.4f" % x for x in out[k]["iqr"]])
    for k in ("model_minus_constant_mean", "model_minus_constant_median"):
        d = out[k]
        print(k, "diff %.4f, model lower in %d, p %.3g" % (d["difference"]["median"], d["model_lower_in"], d["p_wilcoxon"]))
    print("->", p)


if __name__ == "__main__":
    main()
