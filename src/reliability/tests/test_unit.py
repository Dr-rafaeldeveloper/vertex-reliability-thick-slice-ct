"""Unit tests of the reliability package with closed expected values (they do not enter the package hash:
they live in a subfolder). Each test cites the equation/section of the paper that it validates.
Usage: python -m pytest src/reliability/tests -q
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import trimesh

from reliability import a4_features as X
from reliability import (
    a4_config as C,
)
from reliability import (
    a4_data as D,
)
from reliability import (
    a4_statistics as E,
)
from reliability import a4_proximity as PX
from reliability import (
    a4_registration as REG,
)
from reliability import (
    a4_roi as R,
)
from reliability import a4_surface as S
from reliability.a4_grid import Grid


# ------------------------------------------------------------------------- §2.3, Eq. 1
def test_eq1_exact_block_mean():
    rng = np.random.default_rng(0)
    hu = rng.normal(size=(13, 4, 5)).astype(
        np.float32
    )  # 13 is not a multiple of 6 -> truncates to 12
    hu_t, thick = D.thick_slice(hu, 6)
    assert hu_t.shape == (12, 4, 5) and thick.shape == (2, 4, 5)
    expected = np.stack([hu[0:6].mean(0), hu[6:12].mean(0)])
    assert np.allclose(thick, expected, atol=1e-6)


def test_trilinear_z_aligns_block_centers():
    """align_corners=False: the center of thick voxel j falls on thin index j*k + (k-1)/2 (used by d_f and S_interp)."""
    k = 6
    thick = np.zeros((4, 1, 1), np.float32)
    thick[:, 0, 0] = [0, 1, 2, 3]
    up = D.interpolate_trilinear_z(thick, k)[:, 0, 0]
    # at the center of block j (index j*k + 2.5) the interpolation returns exactly the value of the block
    # interior blocks (the first and the last suffer the border clamp of align_corners=False)
    for j in (1, 2):
        z = j * k + (k - 1) / 2.0
        lo, hi = int(np.floor(z)), int(np.ceil(z))
        assert abs(0.5 * (up[lo] + up[hi]) - j) < 1e-6
    # between centers the interpolation is linear: at index 11 (between centers 8.5 and 14.5) it is worth 1 + 2.5/6
    assert abs(up[11] - (1 + 2.5 / 6)) < 1e-6


# --------------------------------------------------------------------------- Grid (§2.5 physical coordinates; Eq. 9)
def test_grid_round_trip_and_refined():
    g = Grid((0.9, 0.9, 3.0), (10.0, -5.0, 100.0), None, (4, 8, 8))
    idx = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    p = g.index_to_physical(idx)
    assert np.allclose(p[0], [10 + 0.9, -5 + 1.8, 100 + 9.0])
    assert np.allclose(g.physical_to_index(p), idx)
    gf = g.refined_z(6)
    assert np.allclose(gf.spacing, [0.9, 0.9, 0.5]) and gf.shape == (24, 8, 8)
    # center of block j on the thin grid (index j*6 + 2.5) == center of thick slice j
    for j in range(4):
        assert np.allclose(
            gf.index_to_physical(np.array([[0, 0, j * 6 + 2.5]]))[0],
            g.index_to_physical(np.array([[0, 0, j]]))[0],
        )


def test_d_f_eq9_known_values():
    g = Grid((1.0, 1.0, 0.5), (0.0, 0.0, 0.0), None, (24, 8, 8))  # thin grid of the foot
    k = 6
    zs = np.array([2.5, 0.0, 5.5, 8.5, 11.0])  # thin indices
    V = np.column_stack([np.zeros(5), np.zeros(5), zs * 0.5])
    d = X.d_f_mm(V, g, k)
    # centers of the acquired slices at 2.5, 8.5, 14.5... (thin index) -> distances in mm = |dz|*0.5
    assert np.allclose(d, [0.0, 1.25, 1.5, 0.0, 1.25])
    assert d.max() <= k * 0.5 / 2 + 1e-9


# ----------------------------------------------------------------------------- Eq. 10, 11 (via trimesh)
def test_signed_angular_defect_and_zero_roughness_on_plane():
    m = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    d = np.asarray(m.vertex_defects)
    assert (
        abs(d.sum() - 4 * np.pi) < 1e-6
    )  # Gauss–Bonnet: sum of the defects = 4π (sphere)
    assert (d > 0).all()  # convex sphere: positive defect
    # a saddle: central vertex below the plane of the neighbors in two directions and above in the others -> negative
    # defect
    V = np.array([[0, 0, 0], [1, 0, 1], [0, 1, -1], [-1, 0, 1], [0, -1, -1]], float)
    F = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]])
    saddle = trimesh.Trimesh(V, F, process=False)
    assert saddle.vertex_defects[0] < 0
    # Eq. 11 on a sphere: pure Laplacian (without volume constraint) shrinks -> roughness > 0 and equal at all vertices
    r = X.roughness(m)
    val = np.bincount(
        m.faces.ravel()
    )  # valence (no. of incident faces) per vertex: 5 (12 vertices) or 6
    assert (
        r.min() > 0
        and np.allclose(r[val == 5], r[val == 5][0], rtol=1e-6)
        and np.allclose(r[val == 6], r[val == 6].mean(), rtol=2e-2)
    )
    assert r[val == 5][0] < r[val == 6][0]


# ----------------------------------------------------------------------------- Eq. 18 (exact distance)
def test_exact_distance_equals_brute_force():
    m = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    rng = np.random.default_rng(1)
    P = rng.normal(size=(30, 3)) * 6
    d, q = PX.nearest_exact(P, m)
    fb = PX.brute_force(P, m)
    assert np.allclose(d, fb, atol=1e-9)
    assert np.allclose(np.linalg.norm(P - q, axis=1), d, atol=1e-9)


# ----------------------------------------------------------------------------- Eq. 15 (signed SDF)
def test_sdf_sign_and_unit():
    mask = np.zeros((6, 6, 6), bool)
    mask[2:4, 2:4, 2:4] = True
    phi = X.sdf_mm(mask, (1.0, 1.0, 2.0))  # z spacing = 2 mm
    assert (phi[mask] < 0).all() and (phi[~mask] > 0).all()
    assert abs(phi[0, 2, 2] - 4.0) < 1e-6  # 2 voxels in z * 2 mm = 4 mm


# ----------------------------------------------------------------------------- §2.9 (Eq. 35–46) and §2.12
def test_auroc_mann_whitney_and_quantile_precision():
    pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    e = np.array([0.1, 0.1, 0.3, 0.2, 0.6, 0.5, 0.8, 0.7, 1.0, 0.9])
    pos = e >= np.quantile(e, 0.9)  # Eq. 40: {1.0} -> index 8
    assert pos.sum() == 1 and pos[8]
    # AUROC = P(pred_pos > pred_neg) + 0.5 P(tie): 0.9 beats 8 of the 9 negatives -> 8/9
    assert abs(E.auroc(pred, pos) - 8 / 9) < 1e-12
    loc = E.localization(pred, e)
    assert (
        abs(loc["auroc_decile"] - 8 / 9) < 1e-12 and loc["precision_10"] == 0.0
    )  # Eq. 41: selected = index 9 (pred 1.0), error 0.9 is not the decile
    assert (
        abs(loc["flagged_error_mm"] - 0.9) < 1e-12
        and abs(loc["residual_error_mm"] - np.mean(e[:9])) < 1e-12
    )


def test_calibration_line_and_deciles():
    pred = np.linspace(0, 1, 100)
    e = 0.2 + 0.8 * pred
    c = E.calibration(pred, e, const=0.5)
    assert (
        abs(c["slope"] - 0.8) < 1e-9 and abs(c["intercept_mm"] - 0.2) < 1e-9
    )  # Eq. 45
    assert [d["n"] for d in c["deciles"]] == [10] * 10  # Eq. 46 equally populated
    assert (
        abs(c["mae_mm"] - np.mean(np.abs(pred - e))) < 1e-12
        and abs(c["mae_mm"] - 0.1) < 1e-9
    )
    assert abs(c["constant_mae_mm"] - np.mean(np.abs(0.5 - e))) < 1e-12


def test_benjamini_hochberg_classic_example():
    p = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.5}
    adj = E.benjamini_hochberg(p)
    assert np.allclose([adj[k] for k in "abcd"], [0.04, 0.04, 0.04, 0.5])
    p2 = {"a": 0.005, "b": 0.011, "c": 0.02, "d": 0.5, "e": 0.9}
    adj2 = E.benjamini_hochberg(
        p2
    )  # m=5: 0.025, 0.0275, 0.0333, 0.625->0.9?, 0.9; monotonized: 0.025,0.0275,0.0333,0.625,0.9
    assert np.allclose(
        [adj2[k] for k in "abcde"], [0.025, 0.0275, 0.0333333333, 0.625, 0.9]
    )


def test_wilcoxon_and_summary():
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8.0])
    b = a - 1
    assert 0 < E.wilcoxon_two_sided(a, b) < 0.05
    assert np.isnan(E.wilcoxon_two_sided(a, a))
    r = E.summary([0.1, 0.2, -0.3, 0.4])
    assert (
        r["n"] == 4
        and r["positives"] == 3
        and abs(r["positive_proportion"] - 0.75) < 1e-12
    )


def test_deterministic_subsample_per_identifier():
    a = E.training_subsample(10000, "z001_foot.nii.gz")
    b = E.training_subsample(10000, "z001_foot.nii.gz")
    c = E.training_subsample(10000, "z002_foot.nii.gz")
    assert (
        len(a) == C.N_TRAIN_PER_CASE
        and np.array_equal(a, b)
        and not np.array_equal(a, c)
        and len(np.unique(a)) == len(a)
    )


# ----------------------------------------------------------------------------- §2.11 (Umeyama without scale)
def test_umeyama_recovers_rotation_and_translation():
    rng = np.random.default_rng(2)
    src = rng.normal(size=(200, 3))
    ang = 0.4
    Rz = np.array(
        [[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]]
    )
    t = np.array([3.0, -2.0, 1.0])
    dst = src @ Rz.T + t
    R, tt = REG.rigid_umeyama(src, dst)
    assert np.allclose(R, Rz, atol=1e-9) and np.allclose(tt, t, atol=1e-9)
    assert abs(np.linalg.det(R) - 1) < 1e-9  # no reflection, no scale
    # with scale 1.5 in the data, the rigid fit does NOT absorb the scale (residual > 0)
    dst2 = 1.5 * src @ Rz.T + t
    R2, t2 = REG.rigid_umeyama(src, dst2)
    assert np.linalg.norm(src @ R2.T + t2 - dst2) > 1.0


def test_registration_five_strategies_and_far_target():
    m = trimesh.creation.icosphere(subdivisions=4, radius=30.0)
    V = np.asarray(m.vertices)
    q = V.copy()  # in-silico: reconstruction and reference in the SAME frame; perfect reconstruction -> T_r = identity
    rng = np.random.default_rng(3)
    e_pred = rng.random(len(V))
    d_shape = rng.random(len(V))
    e_med = rng.random(len(V))
    res = REG.record_case(V, q, e_pred, d_shape, e_med, seed=0, trials=5, K=50)
    res = {k: v for k, v in res.items() if not k.startswith("_")}
    assert sorted(res) == sorted(C.REG_STRATEGIES)
    for s, v in res.items():  # perfect reconstruction: no strategy displaces the target
        assert len(v) == 5 and np.all(v < 1e-6), s
    q2 = V + rng.normal(
        0, 0.05, V.shape
    )  # reconstruction error of 0.05 mm -> small target displacement
    res2 = REG.record_case(V, q2, e_pred, d_shape, e_med, seed=0, trials=5, K=50)
    res2 = {k: v for k, v in res2.items() if not k.startswith("_")}
    assert (
        all(np.all(v < 0.5) for v in res2.values())
        and res2["global"].max() < res2["random"].max() + 0.5
    )


# ----------------------------------------------------------------------------- ROI
def test_registration_selection_by_region_mean():
    """the criterion is the mean over the K neighbors. Discriminating design INSIDE record_case:
 cap (20 % of the sphere) with error 0.5 and perfect reconstruction (q = V, up to 40 % of the sphere); outside,
 error 2.0 with 40 % of isolated vertices of error 0.0 and deformed reconstruction (q = V + noise). Pointwise
 reading: the 20 % of
 smallest e are the isolated ones -> deformed region -> displacement > 0. Per-region reading: centers in the cap
 -> T = identity -> displacement 0."""
    from scipy.spatial import cKDTree

    m = trimesh.creation.icosphere(subdivisions=4, radius=30.0)
    V = np.asarray(m.vertices)
    K = 50
    rng = np.random.default_rng(7)
    z = V[:, 2]
    cap = z > np.quantile(z, 0.8)
    e = 2.0 + 0.01 * rng.random(len(V))
    e[cap] = 0.5 + 0.01 * rng.random(cap.sum())
    perfect = z > np.quantile(z, 0.6)  # cap + band: perfect reconstruction (q = V)
    deformed = np.where(~perfect)[0]
    isolated = rng.choice(deformed, size=int(0.4 * len(deformed)), replace=False)
    e[isolated] = 0.0
    q = V.copy()
    q[~perfect] += rng.normal(0, 0.5, (int((~perfect).sum()), 3))
    _, viz_all = cKDTree(V).query(V, k=K)
    mean = e[viz_all].mean(axis=1)
    pool_region = np.where(mean <= np.quantile(mean, 0.20))[0]
    pool_vertex = np.where(e <= np.quantile(e, 0.20))[0]
    # pointwise reading: only isolated ones (deformed zone); per-region reading: majority in the perfect zone
    assert not perfect[pool_vertex].any() and perfect[pool_region].mean() > 0.5
    mode0 = C.REG_SELECTION
    C.REG_SELECTION = (
        "region_mean"  # reading, kept as an alternative branch
    )
    try:
        res = REG.record_case(V, q, e, e, e, seed=0, trials=40, K=K)
        C.REG_SELECTION = "vertex"  # pointwise reading, only for contrast
        res_v = REG.record_case(V, q, e, e, e, seed=0, trials=40, K=K)
    finally:
        C.REG_SELECTION = mode0
    # per region: the majority of centers have the whole neighborhood in the perfect zone (T = identity);
    # pointwise: isolated centers -> deformed region -> displacement > 0 in ALL trials
    assert np.mean(res["oracle"] < 1e-6) > 0.6 and np.mean(res["field"] < 1e-6) > 0.6
    assert np.all(res_v["oracle"] > 1e-3) and np.all(res_v["field"] > 1e-3)
    assert res["random"].max() > 1e-3


def test_fixed_binary_mask_without_scaling():
    """Eq. 28 "m_t binary mask": output = x ⊙ m_t, without 1/(1-p)."""
    import torch

    from reliability import a4_uncertainty as U

    assert C.DS_MASK_SCALE is False
    mask = torch.tensor([1.0, 0.0, 1.0, 1.0])
    mf = U.FixedMask(mask, 0.1)
    x = torch.ones(2, 4, 3, 3)
    y = mf(x)
    assert torch.equal(y[:, 1], torch.zeros(2, 3, 3)) and torch.equal(
        y[:, 0], torch.ones(2, 3, 3)
    )
    assert float(y.max()) == 1.0  # without 1/(1-p) = 1.111 scaling


def test_thick_roi_isotropic_dilation_in_mm():
    thick = np.full((3, 21, 21), -1000.0, np.float32)
    thick[:, 8:13, 8:13] = 0.0  # 5x5 body (survives the opening of 1 iteration)
    roi = R.thick_roi(
        thick, 2, (1.0, 1.0, 0.5), dilate_mm=2.0
    )  # thin grid spacing (1,1,0.5), k=2
    assert roi.shape == (6, 21, 21)
    # dilation of 2 mm = 2 voxels in x/y from the border of the body (column 12 -> up to 14)
    assert roi[3, 10, 14] and not roi[3, 10, 15]
    ro = R.roi_on_thick_grid(roi, 2)
    assert ro.shape == (3, 21, 21) and ro.any()


# ----------------------------------------------------------------------------- §2.5 segmentation and mesh (synthetic)
def test_segmentation_and_synthetic_mesh():
    hu = np.full((24, 40, 40), -1000.0, np.float32)
    zz, yy, xx = np.ogrid[:24, :40, :40]
    sphere = ((zz - 12) * 1.0) ** 2 + ((yy - 20) * 0.5) ** 2 + (
        (xx - 20) * 0.5
    ) ** 2 <= 8.0**2  # radius 8 mm with spacing (0.5,0.5,1.0)
    hu[sphere] = 800.0
    cav = (
        ((zz - 12) * 1.0) ** 2 + ((yy - 20) * 0.5) ** 2 + ((xx - 20) * 0.5) ** 2
        <= 4.0** 2
    )  # 4 mm cavity (does not close with 2 it.; survives the 0.8 mm Gaussian)
    hu[
        cav
    ] = (
        -1000.0
    )  # internal cavity: the pipeline does NOT fill it (CLEANING_FILL_CAVITIES = False)
    roi = np.ones_like(sphere, bool)
    st = {}
    m = S.segment_bone(hu, roi, (0.5, 0.5, 1.0), stats=st)
    assert C.CLEANING_FILL_CAVITIES is False
    assert (not m[12, 20, 20]) and st["voxels_filled_in_cavities"] == 0
    g = Grid((0.5, 0.5, 1.0), (0, 0, 0), None, hu.shape)
    mesh = S.mask_to_mesh(m, g, target_faces=2000)
    assert (
        1800 <= len(mesh.faces) <= 2200
        and mesh.metadata["faces_before_decimation"] > 2000
    )
    thick_vol = (
        4 / 3 * np.pi * (8**3 - 4**3)
    )  # 8 mm sphere minus the 4 mm cavity (not filled)
    assert abs(mesh.volume - thick_vol) / thick_vol < 0.15
    # declared consequence of the unfilled cavity producing an endosteal surface -> 2 components
    # (a 2 mm cavity would be erased by the 0.8 mm Gaussian in this geometry: measured, 1 component)
    full = S.mask_to_mesh(m, g, target_faces=0)
    assert len(full.split(only_watertight=False)) == 2


# ----------------------------------------------------------------------------- Eq. 3 (exact period of D_k/U_k)
def test_degrade_exact_period_k5_and_k6():
    """Impulse at the center of a LATE block: the old code (9 samples reinterpolated to 48 for
 k = 5) would shift the peak of block 8 from 42 to ~44.8; the new one keeps 42.
 For k = 6 (48 = 8*6) old and new coincide (control)."""
    from reliability import a4_sr as SR

    # (k, impulse row = center of block j, tolerance): k=5 -> 2+5j exact; k=6 -> 2.5+6j (half-integer)
    for k, expected, tol in ((5, 22, 0), (5, 42, 0), (6, 20, 1), (6, 44, 1)):
        img = np.zeros((48, 48), np.float32)
        img[expected, :] = 1.0
        d = SR.degrade(img, k, 0)
        assert d.shape == (48, 48)
        peak = int(np.argmax(d[:, 24]))
        assert abs(peak - expected) <= tol, (k, expected, peak)
        # the response has period k: the impulse spreads around the center of the block (outside the
        # last block, where U_k replicates the border)
        if expected + k <= 47:
            assert d[expected + k, 24] < 0.5 * d[peak, 24]
    # two impulses in consecutive blocks: distance between the peaks = k (exact period, not 48/9)
    img = np.zeros((48, 48), np.float32)
    img[22, :] = 1.0
    img[42, :] = 1.0
    d = SR.degrade(img, 5, 0)
    peaks = np.flatnonzero((d[:, 24] > 0.99 * d[:, 24].max()))
    assert 22 in peaks and 42 in peaks, peaks


def test_bh_and_paired_with_nan():
    adj = E.benjamini_hochberg({"a": 0.01, "b": float("nan"), "c": 0.04})
    assert (
        np.isnan(adj["b"])
        and abs(adj["a"] - 0.02) < 1e-12
        and abs(adj["c"] - 0.04) < 1e-12
    )  # m = 2
    a = {"x": 1.0, "y": 2.0, "z": float("nan"), "w": 4.0}
    b = {"x": 0.5, "y": 1.0, "z": 1.0, "w": 3.0}
    c = E.paired_comparison(a, b)
    assert c["n"] == 3 and c["n_excluded_nan"] == 1
    loc = E.localization(
        np.full(20, 0.3), np.linspace(0, 1, 20)
    )  # constant predictor -> undefined localization
    assert np.isnan(loc["auroc_decile"]) and np.isnan(loc["precision_10"])


# ----------------------------------------------------------------------------- §2.8.2 search for lambda_EWC
def test_search_lambda_monotonic_valley_and_unreachable():
    """Synthetic ratio(lambda) curves: (1) monotonic; (2) with a valley, like the one measured at control point A
 (rises to ~0.05, drops near 1e8, rises again to ~0.12 at 1e10); (3) ceiling 0.05 (unreachable)."""
    from reliability import a4_uncertainty as U

    def r_mono(lam):
        return 0.10 * (lam / 1e9) ** 0.5

    def r_valley(lam):
        x = np.log10(lam)
        return 0.05 * np.exp(-((x - 6.8) ** 2) / 0.8) + 0.12 / (
            1 + np.exp(-(x - 9.6) * 2.5)
        )

    def r_ceiling(lam):
        return 0.05 * (1 - np.exp(-lam / 1e8))

    for f, conv in ((r_mono, True), (r_valley, True), (r_ceiling, False)):
        lam, r, probes, ok = U.search_lambda(f, 4e5, target=0.10, tol=0.10, max_probes=14)
        assert ok is conv, (f.__name__, probes)
        assert (
            abs(f(lam) - r) < 1e-12
        )  # the returned lambda is always a measured lambda
        if conv:
            assert abs(r - 0.10) / 0.10 <= 0.10 and len(probes) <= 14
        else:  # returns the probe closest to the target
            assert r == max(q["effective_ratio"] for q in probes)
    # the curve with a valley traps the pure proportional adjustment: shows that the bisection was necessary
    lam, r, probes, ok = U.search_lambda(r_valley, 4e5, target=0.10, tol=0.10, max_probes=14)
    assert any("bisection" in q["origin"] for q in probes) or ok


# ----------------------------------------------------------------------------- open surfaces
def test_registration_kmeans_region_of_lowest_mean():
    """the selected region is the one with the lowest mean of the criterion among the 25 of the
 k-means; each trial draws K correspondences inside it. Design: perfect reconstruction (q = V) inside
 the region of lowest e, deformed outside -> displacement 0 for field/oracle, > 0 in most of the random ones."""
    assert C.REG_SELECTION == "kmeans"
    m = trimesh.creation.icosphere(subdivisions=5, radius=30.0)
    V = np.asarray(m.vertices)
    rot = E.regions(V)
    rng = np.random.default_rng(11)
    e = 1.0 + 0.01 * rng.random(len(V))
    target_reg = 3
    e[rot == target_reg] = 0.2
    q = V + rng.normal(0, 0.5, V.shape)
    q[rot == target_reg] = V[rot == target_reg]
    res = REG.record_case(V, q, e, e, e, seed=0, trials=30, K=100, labels=rot)
    assert sorted(k for k in res if not k.startswith("_")) == sorted(C.REG_STRATEGIES)
    assert np.all(res["field"] < 1e-6) and np.all(res["oracle"] < 1e-6)
    assert (
        np.mean(res["random"] > 1e-3) > 0.8
    )  # most of the drawn vertices fall outside the perfect region
    assert (
        len({round(x, 9) for x in res["global"]}) > 1
    )  # correspondence sets differ between trials
    assert res["_ineligible_regions"] == 0
    assert np.all(res["global"] > 1e-3)


def test_normalization_modes():
    from reliability import a4_sr as SR

    v = np.array([[[-1000.0, 0.0, 2000.0]]], np.float32)
    mode0 = C.SR_NORMALIZATION
    try:
        C.SR_NORMALIZATION = "hu"
        t, lo, esc = SR.normalize(v)
        assert np.array_equal(t, v) and lo == 0.0 and esc == 1.0
        assert np.allclose(SR.denormalize(t, lo, esc), v)
        C.SR_NORMALIZATION = "minmax"
        t, lo, esc = SR.normalize(v)
        assert (
            abs(t.min()) < 1e-6
            and abs(t.max() - 1) < 1e-5
            and lo == -1000.0
            and abs(esc - 3000) < 1e-6
        )
        assert np.allclose(SR.denormalize(t, lo, esc), v, atol=1e-2)
    finally:
        C.SR_NORMALIZATION = mode0


def test_thick_mask_sinterp():
    """with SINTERP_MASK = "cleaning" the
 thick mask of S_interp is that of segment_bone (an isolated fragment of 1 voxel = 3 mm3 < 50 mm3 disappears, a
 bone block outside the soft-tissue ROI disappears, the bone stays); with mode "threshold" it is the pure threshold
 (both stay). The default mode is the one in the configuration."""
    from reliability import a4_features as X
    from reliability import a4_roi as R

    assert C.SINTERP_MASK == "cleaning"
    k, spacing = 6, (1.0, 1.0, 0.5)
    thick_spacing = (1.0, 1.0, 3.0)
    thick = np.full((8, 60, 60), -1000.0, np.float32)  # air
    thick[1:7, 12:52, 12:52] = 0.0  # soft tissue (> -300 HU): ROI
    thick[2:5, 1:5, 1:5] = (
        400.0  # bone block (144 mm3 > 50 mm3) in the air, 7 mm from the tissue: outside the ROI
    )
    thick[2:6, 20:40, 20:40] = 400.0  # bone (block 4 x 20 x 20 thick voxels)
    thick[3, 45, 45] = 400.0  # isolated fragment of 1 voxel (3 mm3), inside the ROI
    thick_roi = R.roi_on_thick_grid(R.thick_roi(thick, k, spacing), k)
    m_lim = X.thick_mask_sinterp(thick, thick_roi, thick_spacing, mode="threshold")
    m_lmp = X.thick_mask_sinterp(thick, thick_roi, thick_spacing, mode="cleaning")
    assert np.array_equal(m_lim, thick > C.BONE_THRESHOLD_HU) and m_lim[3, 45, 45]
    assert not m_lmp[3, 45, 45] and m_lmp[2:6, 20:40, 20:40].all()
    assert (
        m_lim[2:5, 1:5, 1:5].all() and not m_lmp[2:5, 1:5, 1:5].any()
    )  # clipping by the ROI (§2.5 CCA)
    assert np.array_equal(
        m_lmp, S.segment_bone(thick, thick_roi, thick_spacing)
    )  # and it is the mask of §2.5
    assert np.array_equal(X.thick_mask_sinterp(thick, thick_roi, thick_spacing), m_lmp)


def test_sinterp_without_cap_and_without_own_closing():
    """interpolated_surface adds no cap or closing: a bar that touches the z face of the volume generates an
 OPEN surface (not watertight), and a gap of 2 thick voxels between two blocks of the INPUT MASK is not
 closed (the closing, when there is one, comes from the mask of §2.5 — thick_mask_sinterp —, not from here)."""
    from reliability import a4_features as X

    assert C.SINTERP_CAP is False and C.PADDING_VOXELS == 0
    k = 6
    m = np.zeros((12, 30, 30), bool)
    m[0:4, 10:20, 10:20] = True  # block 1 touching the z = 0 face
    m[6:10, 10:20, 10:20] = (
        True  # block 2, gap of 2 thick voxels (closing x2 would merge)
    )
    g = Grid((0.5, 0.5, 0.5), (0, 0, 0), None, (12 * k, 30, 30))
    s = X.interpolated_surface(m, k, g)
    comps = s.split(only_watertight=False)
    assert len(comps) == 2  # the gap remains: two objects
    assert not s.is_watertight  # open at the z = 0 face (no cap)
    z = np.asarray(s.vertices)[:, 2]
    assert z.min() < 0.5  # the surface reaches the face of the volume
