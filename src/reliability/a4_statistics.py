"""Evaluation protocol (§2.9), ablation (§2.10) and statistics (§2.12).

§2.9.1 Eq. 35: Spearman per case; median, IQR, number of cases with ρ > 0.
§2.9.2 Eq. 36–39: k-means with K = 25 per case (Lloyd), means per region, regional Spearman per case and
 pooled (descriptive); ALL regions are included (no size filter).
§2.9.3 Eq. 40–43: y(v) = 1[e ≥ Q0.90(e)]; AUROC (Mann–Whitney) with ê as score; P0.10 = {ê ≥ Q0.90(ê)};
 precision; e_flag; residual error without the predicted decile.
§2.9.4 Eq. 44–46: MAE per case; constant = mean over ALL training vertices; e = α + β ê; deciles.
§2.10 Eq. 47–50: single feature; leave-one-feature-out; Δ_ij = ρ_full − ρ_−j per case.
§2.12: paired two-sided Wilcoxon per case; Benjamini–Hochberg per family; median and IQR; no bootstrap.
§2.7.4: training with up to 4000 vertices per case, seed from the identifier; evaluation on the full surface.
"""

from __future__ import annotations

import zlib

import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

from reliability import a4_config as C


# ----------------------------------------------------------------------------- §2.7.3 / §2.7.4
def case_seed(identifier: str) -> int:
    return zlib.crc32(identifier.encode("utf-8"))


def training_subsample(
    n_vertices: int, identifier: str, n: int = C.N_TRAIN_PER_CASE
) -> np.ndarray:
    rng = np.random.default_rng(case_seed(identifier))
    return rng.choice(n_vertices, size=min(n, n_vertices), replace=False)


def forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        **C.RF_PARAMS, n_jobs=-1, random_state=C.RF_SEED
    )  # §2.7.3, Eq. 19


def constant_prediction(e: dict, tr, y) -> float:
    """§2.9.4 constant baseline: mean of the measured error over ALL vertices of the training cases
 (CONSTANT_OVER = "full_surfaces") or over the training subsample y. Single implementation
 (used by lofo and by a4_optimize_rf.nested_lofo)."""
    if C.CONSTANT_OVER == "full_surfaces":
        return float(np.concatenate([e[j] for j in tr]).mean())
    return float(np.asarray(y).mean())


def lofo(F: dict, e: dict, columns=None, n_training: int = C.N_TRAIN_PER_CASE) -> dict:
    """Leave-one-case-out (§2.7.4): F[id] (n_i, 9), e[id] (n_i,). Training = subsample of each training
 case (seed from the identifier); prediction on ALL vertices of the held-out case. columns: indices
 of the features used (None = all 9). Returns {id: (pred, mean_training)}."""
    ids = sorted(F)
    cols = list(range(C.N_FEATURES)) if columns is None else list(columns)
    sub = {i: training_subsample(len(e[i]), i, n_training) for i in ids}
    out = {}
    for h in ids:
        tr = [j for j in ids if j != h]
        X = np.vstack([F[j][sub[j]][:, cols] for j in tr])
        y = np.concatenate([e[j][sub[j]] for j in tr])
        mdl = forest().fit(X, y)
        const = constant_prediction(e, tr, y)  # §2.9.4
        out[h] = (mdl.predict(F[h][:, cols]), const)  # Eq. 20
    return out


# ----------------------------------------------------------------------------- §2.9.1
def spearman(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:  # constant (exact)
        return float("nan")
    return float(spearmanr(a, b)[0])


# ----------------------------------------------------------------------------- §2.9.2
def regions(V: np.ndarray, K: int = C.K_REGIONS) -> np.ndarray:
    return KMeans(
        n_clusters=K, n_init=C.KMEANS_N_INIT, random_state=C.KMEANS_SEED
    ).fit_predict(V)  # Eq. 36–37


def regional_means(
    x: np.ndarray, lab: np.ndarray, K: int = C.K_REGIONS
) -> np.ndarray:
    return np.array([x[lab == c].mean() for c in range(K)])  # Eq. 38–39


# ----------------------------------------------------------------------------- §2.9.3
def auroc(score, positive) -> float:
    p = int(positive.sum())
    n = int((~positive).sum())
    if p == 0 or n == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[positive].sum() - p * (p + 1) / 2.0) / (p * n))


def localization(
    pred: np.ndarray, e: np.ndarray, fraction: float = C.DECILE_FRACTION
) -> dict:
    nan = {
        "auroc_decile": float("nan"),
        "precision_10": float("nan"),
        "flagged_error_mm": float("nan"),
        "overall_error_mm": float(e.mean()),
        "residual_error_mm": float("nan"),
    }
    if (
        np.ptp(pred) == 0
    ):  # constant predictor (max = min, exact): there is no "10 % with the largest ê"
        return nan
    pos = e >= np.quantile(e, 1 - fraction)  # Eq. 40
    sel = pred >= np.quantile(pred, 1 - fraction)  # Eq. 41
    if (
        sel.all() or not sel.any()
    ):  # degenerate selection: Eq. 40 remains defined; Eq. 41-43 do not
        return dict(nan, auroc_decile=auroc(pred, pos))
    return {
        "auroc_decile": auroc(pred, pos),
        "precision_10": float((pos & sel).sum() / max(1, sel.sum())),  # Eq. 42
        "flagged_error_mm": float(e[sel].mean()),  # Eq. 43
        "overall_error_mm": float(e.mean()),
        "residual_error_mm": float(e[~sel].mean()),
    }


# ----------------------------------------------------------------------------- §2.9.4
def calibration(
    pred: np.ndarray, e: np.ndarray, const: float, ndec: int = C.N_CALIBRATION_DECILES
) -> dict:
    mae = float(np.mean(np.abs(pred - e)))  # Eq. 44
    mae_c = float(np.mean(np.abs(const - e)))
    if np.ptp(pred) > 0:
        beta, alpha = np.polyfit(pred, e, 1)  # Eq. 45: e = α + β ê
    else:
        beta, alpha = float("nan"), float("nan")
    q = np.quantile(pred, np.linspace(0, 1, ndec + 1))
    dec = []
    for d in range(ndec):  # Eq. 46
        m = (pred >= q[d]) & (
            (pred <= q[d + 1]) if d == ndec - 1 else (pred < q[d + 1])
        )
        dec.append(
            {
                "decile": d + 1,
                "predicted_mm": float(pred[m].mean()) if m.any() else float("nan"),
                "observed_mm": float(e[m].mean()) if m.any() else float("nan"),
                "n": int(m.sum()),
            }
        )
    return {
        "mae_mm": mae,
        "constant_mae_mm": mae_c,
        "constant_mm": const,
        "slope": float(beta),
        "intercept_mm": float(alpha),
        "deciles": dec,
    }


# ----------------------------------------------------------------------------- §2.12
def wilcoxon_two_sided(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if np.all(a == b):
        return float("nan")  # degenerate (all differences zero): no p-value
    return float(wilcoxon(a, b, alternative="two-sided")[1])


def benjamini_hochberg(pvals: dict) -> dict:
    """Benjamini & Hochberg (1995): p_adj_(i) = min_{j>=i} (m/j) p_(j), truncated at 1. Keys with p = NaN (degenerate
 comparison) stay NaN and do NOT count in m."""
    valid = [k for k in pvals if np.isfinite(pvals[k])]
    out = {k: float("nan") for k in pvals}
    p = np.array([pvals[k] for k in valid], float)
    m = len(p)
    if m == 0:
        return out
    order = np.argsort(p)
    adj = np.empty(m)
    current = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        current = min(current, p[i] * m / rank)
        adj[i] = current
    for k, v in zip(valid, adj, strict=True):
        out[k] = float(min(1.0, v))
    return out


def summary(values) -> dict:
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {
            "median": float("nan"),
            "iqr": [float("nan"), float("nan")],
            "n": 0,
            "positives": 0,
        }
    return {
        "median": float(np.median(v)),
        "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
        "n": len(v),
        "positives": int((v > 0).sum()),
        "positive_proportion": float((v > 0).mean()),
    }  # §2.9.1 "number and proportion"


def paired_comparison(a: dict, b: dict) -> dict:
    """a, b: {id: value}; difference a − b per case, median/IQR, two-sided Wilcoxon (raw p; BH is
 applied per family by the caller)."""
    ids = sorted(
        i for i in set(a) & set(b) if np.isfinite(a[i]) and np.isfinite(b[i])
    )  # only cases with both values
    n_excluded = len(set(a) & set(b)) - len(ids)
    da = np.array([a[i] for i in ids])
    db = np.array([b[i] for i in ids])
    if len(ids) < 2:
        return {
            "n": len(ids),
            "n_excluded_nan": n_excluded,
            "difference": summary([]),
            "p_wilcoxon": float("nan"),
            "degenerate": True,
        }
    return {
        "n": len(ids),
        "n_excluded_nan": n_excluded,
        "difference": summary(da - db),
        "p_wilcoxon": wilcoxon_two_sided(da, db),
        "degenerate": bool(np.all(da == db)),
    }
