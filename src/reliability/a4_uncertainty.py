"""Predictive uncertainty baselines (§2.8): intensity space (§2.8.1, Eq. 21–27),
adapted Dropsembles (§2.8.2, Eq. 28–30) and surface space (§2.8.3, Eq. 31–34).

§2.8.1: M = 5 additional networks with dropout p = 0.1, trained independently; S = 4 stochastic passes
each; Ī_m (Eq. 22); u_ens = sample SD across the M means (Eq. 23); u_MC = sample SD
across the S passes of one network (Eq. 25); u_dropens = SD of the M·S reconstructions (Eq. 26; sample
divisor N−1, as in Eq. 23 and 25); "each intensity-space uncertainty map was sampled at the physical coordinates of
the
evaluated surface vertices... trilinear interpolation" (Eq. 27) — the MAPS are computed in the voxel grid and
only then sampled.
§2.8.2: dense prior (6000 it, 432 slices from 36 feet, 4 partitions), 5 subnetworks by fixed
dropout masks (Eq. 28), adaptation 600 it, lr 2e-4, EWC with diagonal Fisher (Eq. 29), λ adjusted to ≈ 10 %
of the loss after ≈ 50 it; u_DS = SD of the 5 reconstructions (Eq. 30).
§2.8.3: "the same stochastic reconstructions were then converted into bone surfaces": for each
reconstruction I_r, mask (§2.5) -> SDF in mm -> d_r(v) = φ_r(x_v) (Eq. 31); u_geo,std (Eq. 32),
u_geo,abs (Eq. 33) over R = 5 reconstructions = the per-member means Ī_m; u_geo,MC (Eq. 34)
over the S passes of the selected member (the same one as in Eq. 25).
Everything in HU (intensity) or mm (surface); voxel-by-voxel accumulators in float64.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn

from reliability import a4_config as C
from reliability import a4_sr as SR
from reliability.a4_features import sdf_mm
from reliability.a4_sr import degrade
from reliability.a4_surface import segment_bone


class _Accum:
    """Voxel-by-voxel sum and sum of squares (float64) for the variance of maps."""

    def __init__(self, shape):
        self.s = np.zeros(shape, np.float64)
        self.s2 = np.zeros(shape, np.float64)
        self.n = 0

    def add(self, r):
        r = r.astype(np.float64)
        self.s += r
        self.s2 += r * r
        self.n += 1

    def mean(self):
        return self.s / self.n

    def sd(self, ddof):
        var = (self.s2 - self.s * self.s / self.n) / max(1, self.n - ddof)
        return np.sqrt(np.maximum(var, 0.0))


# ----------------------------------------------------------------------------- §2.8.1 + §2.8.3
def ensemble_mc(
    slices, thick_norm, k, dev, lo, esc, thin_grid, thin_roi, V, iters, verbose=True
):
    """Returns maps in HU (u_ens, u_mc, u_dropens) sampled at V (Eq. 27) and surface
 distances d_members (M, n), d_mc (S, n) in mm (Eq. 31), using the SAME reconstructions."""
    shape = (thick_norm.shape[0] * k,) + thick_norm.shape[1:]
    spacing = tuple(thin_grid.spacing)
    ac_all = _Accum(shape)
    ac_means = _Accum(shape)
    ac_mc = _Accum(shape)
    d_members, d_mc = [], []
    t0 = time.time()
    for m, seed in enumerate(C.ENS_SEEDS[: C.ENS_M]):
        network, _ = SR.train(
            slices, k, iters, dev, seed=seed, dropout=C.ENS_DROPOUT_P, verbose=False
        )  # §2.8.1
        ac_m = _Accum(shape)
        for s in range(C.ENS_S):
            torch.manual_seed(
                C.ENS_PASSES_SEED_BASE * (m + 1) + s
            )  # seeds of the passes
            r = SR.apply(network, thick_norm, k, mc_dropout=True)  # I_{m,s} (Eq. 21)
            ac_m.add(r)
            ac_all.add(r)
            if m == C.ENS_MEMBER_MC:
                ac_mc.add(r)
                mask = segment_bone(
                    SR.denormalize(r, lo, esc), thin_roi, spacing
                )  # §2.8.3 (§2.5)
                d_mc.append(
                    thin_grid.sample(sdf_mm(mask, spacing), V)
                )  # d_s(v), Eq. 31/34
        Im = ac_m.mean()  # Ī_m (Eq. 22)
        ac_means.add(Im)
        mask = segment_bone(
            SR.denormalize(Im.astype(np.float32), lo, esc), thin_roi, spacing
        )  # I_r = Ī_m
        d_members.append(thin_grid.sample(sdf_mm(mask, spacing), V))  # d_r(v), Eq. 31
        del network
        if verbose:
            print(
                "    member %d/%d ok (%.0fs)" % (m + 1, C.ENS_M, time.time() - t0),
                flush=True,
            )
    u_ens = (
        thin_grid.sample(ac_means.sd(ddof=C.UNCERTAINTY_DDOF).astype(np.float32), V)
        * esc
    )  # Eq. 23 -> Eq. 27
    u_mc = (
        thin_grid.sample(ac_mc.sd(ddof=C.UNCERTAINTY_DDOF).astype(np.float32), V) * esc
    )  # Eq. 25 -> Eq. 27
    u_dropens = (
        thin_grid.sample(ac_all.sd(ddof=C.ENS_DROPENS_DDOF).astype(np.float32), V)
        * esc
    )  # Eq. 26 -> Eq. 27
    D = np.stack(d_members)
    Dmc = np.stack(d_mc)
    return {
        "u_ens": u_ens.astype(np.float32),
        "u_mc": u_mc.astype(np.float32),
        "u_dropens": u_dropens.astype(np.float32),
        "d_geo_members": D.astype(np.float32),
        "d_geo_mc": Dmc.astype(np.float32),
        "u_geo_std": D.std(axis=0, ddof=C.UNCERTAINTY_DDOF).astype(np.float32),  # Eq. 32
        "u_geo_abs": np.abs(D).mean(axis=0).astype(np.float32),  # Eq. 33
        "u_geo_mc": Dmc.std(axis=0, ddof=C.UNCERTAINTY_DDOF).astype(np.float32),
    }  # Eq. 34


# ----------------------------------------------------------------------------- §2.8.2 Dropsembles
class FixedMask(nn.Module):
    """Eq. 28: θ_t = m_t ⊙ θ, "m_t binary mask" — BINARY mask without scaling.
 §2.8.2 "subnetworks induced by fixed DROPOUT masks" ->
 per-CHANNEL mask at the 8 Dropout2d positions of the body (zeroes the output rows of the preceding convolution:
 a structured subset of m_t ⊙ θ; 1st and 10th convolutions never masked, as in the prior's dropout).
 Alternative: per-parameter Bernoulli over all of θ (more literal for "⊙ θ", but it is not a "dropout mask")."""

    def __init__(self, mask, p):
        super().__init__()
        self.register_buffer("mask", mask.view(1, -1, 1, 1))
        self.scale = (
            (1.0 / (1.0 - p) if p < 1 else 1.0) if C.DS_MASK_SCALE else 1.0
        )  

    def forward(self, x):
        return x * self.mask * self.scale


def sub_network(prior, p, seed):
    gen = torch.Generator().manual_seed(seed)
    new = copy.deepcopy(prior)
    body = list(new.body)
    for i, m in enumerate(body):
        if isinstance(m, nn.Dropout2d):
            ch = body[i - 2].out_channels
            mask = (torch.rand(ch, generator=gen) > p).float()
            if (
                mask.sum() == 0
            ):  # declared guard (measure-zero EXTRA: prob. 0.1^48); the text does not describe it
                mask[torch.randint(ch, (1,), generator=gen)] = 1.0
            body[i] = FixedMask(mask, p)
    new.body = nn.Sequential(*body)
    return new


def _batch(slices, ids, k, patch, bs, rng):
    xb, yb = [], []
    for _ in range(bs):
        sl = slices[ids[rng.integers(len(ids))]]
        H, W = sl.shape
        if H <= patch or W <= patch:
            continue
        y0 = int(rng.integers(0, H - patch))
        x0 = int(rng.integers(0, W - patch))
        hr = sl[y0 : y0 + patch, x0 : x0 + patch]
        xb.append(degrade(hr, k, int(rng.integers(0, 2)))[None])
        yb.append(hr[None])
    return np.stack(xb), np.stack(yb)


def fisher_diagonal(network, slices, k, dev, batches=C.DS_FISHER_BATCHES, seed=0):
    """Eq. 29: F_j = diagonal Fisher estimate (mean of grad^2 of the L1 loss over the prior pool)."""
    rng = np.random.default_rng(seed)
    ids = np.arange(len(slices))
    phys = {n: torch.zeros_like(p) for n, p in network.named_parameters()}
    loss = nn.L1Loss()
    network.eval()  # Fisher of the DENSE prior (dropout inactive), as in the text and the old code
    for _ in range(batches):
        xb, yb = _batch(slices, ids, k, C.SR_PATCH, C.SR_BATCH, rng)
        x = torch.from_numpy(xb).float().to(dev)
        y = torch.from_numpy(yb).float().to(dev)
        network.zero_grad()
        loss(network(x), y).backward()
        for n, p in network.named_parameters():
            if p.grad is not None:
                phys[n] += p.grad.detach() ** 2
    for n in phys:
        phys[n] /= max(1, batches)
    network.zero_grad()
    network.eval()
    return phys


def train_prior(pool, k, dev, seed, iters=C.DS_PRIOR_ITERS, verbose=True):
    """§2.8.2: dense prior, with dropout p = 0.1, over the pool of slices from other feet."""
    network, info = SR.train(
        pool,
        k,
        iters,
        dev,
        seed=seed,
        dropout=C.ENS_DROPOUT_P,
        lr=C.DS_PRIOR_LR,
        verbose=verbose,
    )
    phys = fisher_diagonal(network, pool, k, dev)
    anc = {n: p.detach().clone() for n, p in network.named_parameters()}
    return network, phys, anc, info


def search_lambda(
    measure,
    lam0,
    target=C.DS_EWC_FRACTION,
    tol=C.DS_EWC_TOLERANCE,
    max_probes=C.DS_EWC_MAX_PROBES,
    factor_min=C.DS_EWC_FACTOR_MIN,
):
    """Searches for lambda such that measure(lambda) ~ target (penalty/loss ratio at t = calibrate, §2.8.2), with the
 ratio possibly NON-monotonic in lambda. Strategy: (i) geometric growth starting from the
 largest lambda already probed, factor max(factor_min, target/ratio), until there is a probe above the target;
 (ii) bisection in log(lambda) between the largest lambda below the target (with lambda smaller than the one above)
 and the
 smallest lambda above the target — by continuity there is a crossing in that interval. Every probe is measured;
 returns (lambda, ratio, probes, converged). If it does not converge, returns the probe whose ratio is closest
 to the target (in log), always measured."""
    probes = []

    def probe(lam, origin):
        r = float(measure(lam))
        probes.append({"lambda": float(lam), "effective_ratio": r, "origin": origin})
        return r

    r = probe(lam0, "probe without penalty")
    for _ in range(max_probes - 1):
        if r > 0 and abs(r - target) / target <= tol:
            return (
                float(lam0 if len(probes) == 1 else probes[-1]["lambda"]),
                r,
                probes,
                True,
            )
        above = [q for q in probes if q["effective_ratio"] > target]
        if above:
            la = min(above, key=lambda q: q["lambda"])["lambda"]
            below = [
                q for q in probes if q["effective_ratio"] < target and q["lambda"] < la
            ]
            lb = (
                max(below, key=lambda q: q["lambda"])["lambda"]
                if below
                else la / 10.0
            )
            lam = float(np.sqrt(la * lb))
            r = probe(lam, "bisection in log(lambda)")
        else:
            largest = max(probes, key=lambda q: q["lambda"])
            rm = largest["effective_ratio"]
            factor = max(factor_min, target / rm) if rm > 0 else 10.0
            lam = largest["lambda"] * factor
            r = probe(lam, "geometric growth x%.2f" % factor)
    if r > 0 and abs(r - target) / target <= tol:
        return float(probes[-1]["lambda"]), r, probes, True
    best = min(
        (q for q in probes if q["effective_ratio"] > 0),
        key=lambda q: abs(np.log(q["effective_ratio"] / target)),
        default=probes[-1],
    )
    return float(best["lambda"]), float(best["effective_ratio"]), probes, False


def adapt(
    sub,
    slices,
    k,
    dev,
    fisher,
    anchor,
    iters=C.DS_REFINEMENT_ITERS,
    lr=C.DS_REFINEMENT_LR,
    seed=0,
    fraction=C.DS_EWC_FRACTION,
    calibrate=C.DS_EWC_ITER_CALIBRATION,
):
    """Eq. 29: L_adapt = L_data + λ_EWC Σ_j F_j (θ_j − θ*_j)^2; λ adjusted after `calibrate` iterations so
 that the penalty equals `fraction` of the data loss. Returns (sub_network, lambda, info)."""
    loss = nn.L1Loss()

    def penalty(network):
        return sum(
            (fisher[n] * (p - anchor[n]) ** 2).sum()
            for n, p in network.named_parameters()
            if n in fisher
        )

    def run(network, n_it, lam, rng, record_in=None):
        """Adaptation of n_it iterations with fixed lambda (None = no penalty). In `record_in` it writes the
 data loss, the penalty (without lambda) and the effective ratio lambda*pen/L_data."""
        ids = np.arange(len(slices))
        opt = torch.optim.Adam(network.parameters(), lr=lr)
        network.train()
        reg = {}
        for it in range(n_it):
            xb, yb = _batch(slices, ids, k, C.SR_PATCH, C.SR_BATCH, rng)
            x = torch.from_numpy(xb).float().to(dev)
            y = torch.from_numpy(yb).float().to(dev)
            if record_in is not None and it == record_in:
                with torch.no_grad():
                    l0 = float(loss(network(x), y))
                    p0 = float(penalty(network))
                reg = {
                    "iter": it,
                    "data_loss": l0,
                    "penalty_without_lambda": p0,
                    "effective_ratio": (lam * p0 / max(l0, 1e-30)) if lam else 0.0,
                }
                if C.DS_EWC_MODE == "after_calibration" and lam is None:
                    lam = fraction * l0 / max(p0, 1e-30)
                    reg["lambda_ewc"] = lam
            opt.zero_grad()
            L = loss(network(x), y)
            if lam is not None and lam > 0:
                L = L + lam * penalty(network)
            L.backward()
            opt.step()
        network.eval()
        return network, reg, lam

    if C.DS_EWC_MODE == "probe_and_restart":
        # "The EWC coefficient was adjusted so that, after approximately 50 adaptation
        # iterations, the regularization term contributed approximately 10% of the data-fitting loss" and
        # "Adaptation minimized a reconstruction objective regularized by an EWC penalty" -> fixed point:
        # REGULARIZED probes of `calibrate` iterations, starting from the initial subnetwork, adjusting lambda until the
        # penalty/loss ratio at t = calibrate is at `fraction` (10 % relative tolerance, up to DS_EWC_MAX_PROBES probes; geometric growth then bisection in log lambda);
        # then the full adaptation starts from the initial subnetwork with that lambda from iteration 0.
        assert iters > calibrate, (
            "the adaptation needs more than %d iterations to measure the ratio at t = %d"
            % (calibrate, calibrate)
        )
        initial = copy.deepcopy(sub)
        _, reg0, _ = run(
            copy.deepcopy(initial),
            calibrate + 1,
            None,
            np.random.default_rng(seed),
            record_in=calibrate,
        )
        lam0 = fraction * reg0["data_loss"] / max(reg0["penalty_without_lambda"], 1e-30)

        def measure(
            lam_,
        ):  # regularized probe of `calibrate` it. starting from the initial subnetwork (same seed)
            _, reg_s, _ = run(
                copy.deepcopy(initial),
                calibrate + 1,
                lam_,
                np.random.default_rng(seed),
                record_in=calibrate,
            )
            return reg_s["effective_ratio"]

        lam, r_probe, probes, converged = search_lambda(measure, lam0, target=fraction)
        sub, reg_final, _ = run(
            sub, iters, lam, np.random.default_rng(seed), record_in=calibrate
        )
        registration = {
            "mode": "probe_and_restart_fixed_point",
            "calibration_iter": calibrate,
            "lambda_ewc": lam,
            "probes": probes,
            "n_probes": len(probes),
            "converged": bool(converged),
            "effective_ratio_in_adaptation_it50": reg_final.get("effective_ratio"),
            "data_loss_it50": reg_final.get("data_loss"),
        }
        if not converged:
            print(
                "    [WARNING] lambda_EWC did not converge to %.0f%% in %d probes (ratio %.4f)"
                % (100 * fraction, len(probes), r_probe),
                flush=True,
            )
    else:
        sub, registration, lam = run(
            sub, iters, None, np.random.default_rng(seed), record_in=calibrate
        )
        registration["mode"] = "after_calibration"
        registration["lambda_ewc"] = lam
    return sub, lam, registration


def dropsembles(
    prior,
    fisher,
    anchor,
    slices,
    thick_norm,
    k,
    dev,
    lo,
    esc,
    thin_grid,
    V,
    iters_refinement=C.DS_REFINEMENT_ITERS,
    verbose=True,
):
    """Eq. 30: u_DS = sample SD of the 5 reconstructions of the adapted subnetworks, in the voxel grid, sampled at V."""
    shape = (thick_norm.shape[0] * k,) + thick_norm.shape[1:]
    ac = _Accum(shape)
    lambdas = []
    t0 = time.time()
    for t in range(C.DS_SUBNETS):
        sub = sub_network(
            prior,
            C.ENS_DROPOUT_P,
            C.DS_SUBNET_SEED_BASE + C.DS_SUBNET_SEED_STEP * t,
        ).to(dev)
        sub, lam, reg = adapt(
            sub, slices, k, dev, fisher, anchor, iters=iters_refinement, seed=t
        )
        lambdas.append(reg)
        ac.add(SR.apply(sub, thick_norm, k))  # I_t (fixed masks -> deterministic)
        del sub
        if verbose:
            print(
                "    subnetwork %d/%d ok, lambda_EWC=%s ratio it50=%s (%.0fs)"
                % (
                    t + 1,
                    C.DS_SUBNETS,
                    ("%.3g" % lam) if lam else None,
                    reg.get("effective_ratio_in_adaptation_it50"),
                    time.time() - t0,
                ),
                flush=True,
            )
    u_ds = (
        thin_grid.sample(ac.sd(ddof=C.UNCERTAINTY_DDOF).astype(np.float32), V) * esc
    )  # Eq. 30 -> sampling
    return u_ds.astype(np.float32), lambdas
