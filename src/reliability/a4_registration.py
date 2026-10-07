"""Reliability-guided rigid registration (§2.11).

Text: "local surface regions were defined around candidate registration locations, and
correspondences were sampled from the selected region. Rigid transformations were estimated without
scaling using the least-squares formulation of Umeyama [45]. For each candidate region, K = 200
surface correspondences were used... Registration regions were selected according to alternative
strategies based on the predicted geometric reliability field, the shape-disagreement descriptor,
random surface selection, and the measured geometric error used only as an oracle reference. A
global-surface registration condition was additionally included... The reliability-guided
condition selected regions with the lowest predicted geometric error, whereas the oracle condition
selected regions with the lowest measured error.... For each registration location, the target was
selected from surface locations separated from the registration region by a distance exceeding the
80th percentile of the within-surface distance distribution... 300 repeated registration trials...
Correspondences were treated as noiseless... d_target = ||T_r(x_target) − T_ref(x_target)||_2...
summarized within each case before cohort-level comparison."

Declared readings: "regions" = the 25 k-means regions of §2.9.2 on the
evaluated surface; "lowest" = region with the lowest mean of the criterion (argmin); K = 200 correspondences drawn
INSIDE the region at each trial; random = uniform vertex on the surface -> its region. Previous branches
("region_mean": 200 neighbors of the center, 20 %; "vertex") kept as alternatives. Global condition =
K = 200 random vertices of the whole surface, with the target by the SAME 80th-percentile criterion; T_ref =
identity with exact correspondences and x_target = q (reference correspondence of the target), hence
d = ||T_r(q) − q||; the target is drawn relative to the region of EACH strategy: among the vertices
whose distance to the region exceeds the 80th percentile of the distribution of those distances.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from reliability import a4_config as C
from reliability import a4_statistics as E


def rigid_umeyama(src: np.ndarray, dst: np.ndarray):
    """Umeyama (1991) without scaling: R, t minimizing Σ ||R s_i + t − d_i||^2."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    U, S, Vt = np.linalg.svd(cov)
    Dm = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        Dm[2, 2] = -1
    R = U @ Dm @ Vt
    t = mu_d - R @ mu_s
    return R, t


def record_case(
    V: np.ndarray,
    q: np.ndarray,
    e_pred: np.ndarray,
    d_shape: np.ndarray,
    e_med: np.ndarray,
    seed: int,
    K: int = C.REG_K,
    trials: int = C.REG_TRIALS,
    p_target: float = C.REG_TARGET_PERCENTILE,
    quantile: float = C.REG_SELECTION_QUANTILE,
    n_global: int = C.REG_GLOBAL_N,
    labels=None,
) -> dict:
    """V (n,3) vertices of the reconstructed mesh; q (n,3) exact correspondences on the reference;
 e_pred = ê(v) of the LOO model; d_shape = x6; e_med = measured e(v) (only for the oracle).
 Region selection by C.REG_SELECTION: "kmeans" or the previous branches
 "region_mean" and "vertex". Returns {strategy: array(trials) of d_target in mm}."""
    rng = np.random.default_rng(seed)
    N = len(V)
    tree = cKDTree(V)
    if N <= K or N <= n_global:
        raise ValueError("degenerate mesh: %d vertices for K = %d" % (N, K))
    if (
        C.REG_SELECTION == "kmeans"
    ):  # regions = k-means of §2.9.2; see a4_config.REG_SELECTION
        return _record_kmeans(
            V,
            q,
            e_pred,
            d_shape,
            e_med,
            rng,
            tree,
            K,
            trials,
            p_target,
            n_global,
            labels,
        )
    if C.REG_SELECTION == "region_mean":
        # "selected regions with the lowest... error" — REGION criterion = mean over the K
        # neighbors of each candidate center (all vertices are candidates)
        _, viz = tree.query(V, k=K)

        def criterion(x):
            return np.asarray(x, float)[viz].mean(axis=1)
    else:  # "vertex" (original): pointwise value of the center

        def criterion(x):
            return np.asarray(x, float)

    def smallest(x):  # the 20 % with the lowest criterion
        c = criterion(x)
        return np.where(c <= np.quantile(c, quantile))[0]

    pool = {
        "field": smallest(e_pred),
        "d_shape": smallest(d_shape),
        "random": np.arange(N),
        "oracle": smallest(e_med),
    }
    out = {s: np.empty(trials) for s in C.REG_STRATEGIES}
    for r in range(trials):
        for s in C.REG_STRATEGIES:
            if s == "global":
                loc = rng.choice(N, size=min(n_global, N), replace=False)
                # §2.11 "For each registration location, the target was selected... separated from the
                # registration region": the "region" of the global condition is the set of the K drawn
                # correspondences; the same 80th-percentile criterion applies
                d_reg = cKDTree(V[loc]).query(V)[0]
                far = np.where(d_reg > np.percentile(d_reg, p_target))[0]
                if len(far) == 0:
                    raise ValueError(
                        "global strategy: no vertex beyond percentile %d of the distance to the region"
                        % p_target
                    )
                target = int(far[rng.integers(len(far))])
            else:
                center = int(pool[s][rng.integers(len(pool[s]))])
                _, loc = tree.query(V[center], k=K)
                loc = np.atleast_1d(loc)
                if C.REG_TARGET_MODE == "region":
                    d_reg = cKDTree(V[loc]).query(V)[
                        0
                    ]  # distance of each vertex to the REGION
                    far = np.where(d_reg > np.percentile(d_reg, p_target))[0]
                else:
                    dist = np.linalg.norm(V - V[center], axis=1)
                    far = np.where(dist > np.percentile(dist, p_target))[0]
                if len(far) == 0:
                    raise ValueError(
                        "strategy %s: no vertex beyond percentile %d"
                        % (s, p_target)
                    )
                target = int(far[rng.integers(len(far))])
            R, t = rigid_umeyama(
                V[loc], q[loc]
            )  # T_r: reconstructed mesh -> reference
            x = q[target]  # x_target = reference correspondence
            out[s][r] = float(
                np.linalg.norm(R @ x + t - x)
            )  # ||T_r(x) − T_ref(x)||, T_ref = identity
    return out


def _record_kmeans(
    V, q, e_pred, d_shape, e_med, rng, tree, K, trials, p_target, n_global, labels
):
    """The reading of "regions" is the 25 k-means regions (Eq. 36-37) of the
 evaluated surface; the strategy selects the region with the lowest mean of the criterion (Eq. 39 for ê; analogous for
 d_shape and e); each trial draws K correspondences inside the region; random draws a uniform vertex
 on the surface and uses its region;
 global: K correspondences on the whole surface; target: beyond the p80 of the distance to the region."""
    N = len(V)
    if labels is None:
        labels = E.regions(V)
    labels = np.asarray(labels)
    regs = np.unique(labels)
    members = {r: np.where(labels == r)[0] for r in regs}
    # Regions with fewer than K vertices cannot provide "K = 200 surface correspondences" (§2.11): they become
    # ineligible for selection and for the random draw (consequence of the quoted K, not a new parameter).
    # A k-means region may hold only a few vertices (isolated fragment); regions with fewer than K vertices are
    # ineligible. In the exact-error run no region was ineligible; in the sampled run, 2 of 48 feet had one.
    eligible = np.array([r for r in regs if len(members[r]) >= K])
    if len(eligible) == 0:
        raise ValueError(
            "no k-means region with K = %d vertices or more" % K
        )
    eligible_vert = np.concatenate([members[r] for r in eligible])
    n_ineligible = int(len(regs) - len(eligible))

    def smallest(x):
        x = np.asarray(x, float)
        means = np.array([x[members[r]].mean() for r in eligible])
        return eligible[int(np.argmin(means))]

    sel = {"field": smallest(e_pred), "d_shape": smallest(d_shape), "oracle": smallest(e_med)}
    d_reg_cache = {}

    def far_from(r):
        if r not in d_reg_cache:
            d_reg = cKDTree(V[members[r]]).query(V)[0]
            d_reg_cache[r] = np.where(d_reg > np.percentile(d_reg, p_target))[0]
        return d_reg_cache[r]

    out = {s: np.empty(trials) for s in C.REG_STRATEGIES}
    out["_ineligible_regions"] = n_ineligible
    for t in range(trials):
        for s in C.REG_STRATEGIES:
            if s == "global":
                loc = rng.choice(N, size=min(n_global, N), replace=False)
                d_reg = cKDTree(V[loc]).query(V)[0]
                far = np.where(d_reg > np.percentile(d_reg, p_target))[0]
            else:
                # random: "random surface selection" = uniform vertex on the surface -> its region
                r = (
                    labels[eligible_vert[rng.integers(len(eligible_vert))]]
                    if s == "random"
                    else sel[s]
                )
                loc = rng.choice(members[r], size=K, replace=False)
                far = far_from(r)
            if len(far) == 0:
                raise ValueError(
                    "strategy %s: no vertex beyond percentile %d"
                    % (s, p_target)
                )
            target = int(far[rng.integers(len(far))])
            R, tt = rigid_umeyama(V[loc], q[loc])
            x = q[target]
            out[s][t] = float(np.linalg.norm(R @ x + tt - x))
    return out


def case_summary(res: dict) -> dict:
    return {
        s: {
            "median_mm": float(np.median(v)),
            "iqr_mm": [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
            "n_trials": len(v),
        }
        for s, v in res.items()
        if not str(s).startswith("_")
    } | {"ineligible_regions": int(res.get("_ineligible_regions", 0))}


def _old_case_summary(res: dict) -> dict:
    return {
        s: {
            "median_mm": float(np.median(v)),
            "iqr_mm": [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
            "n_trials": len(v),
        }
        for s, v in res.items()
    }
