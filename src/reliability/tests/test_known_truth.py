"""Tests with known ground truth for the error measurement (Section 2.6).

T1  identity: a surface compared with itself must have error 0 (any floor shows up here)
T2  sphere: reference radius r, reconstructed radius r+d -> every vertex error must be d
T3  on a real case, the exact distance never exceeds the sampled one, and ||v - q|| from the cache is the exact distance

The sampled error of Eq. 5 is expected to FAIL T1 and the small offsets of T2: that is the sampling floor documented in
supplement/exact_error.py. Those cases are marked as expected failures (strict xfail) rather than hidden, and the exact
point-to-triangle distance must pass them.
"""

import os
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from reliability import a4_config as C  # noqa: E402
from reliability import a4_proximity as P  # noqa: E402
from reliability import a4_surface as S  # noqa: E402

CACHE = C.A4_CACHE_FOOT
TOL_MM = 0.02  # tolerance for exact distances (facets of the icosphere)
FLOOR = pytest.mark.xfail(strict=True, reason="sampling floor of Eq. 5")


def sphere(r, n=5):
    return trimesh.creation.icosphere(subdivisions=n, radius=r)


def measure(method, V, ref, name):
    if method == "sampled_eq5":
        return S.vertex_error(V, S.sample_reference(ref, name))  # 120,000 points as in the paper
    e, _q = P.nearest_exact(V, ref)
    return e


@pytest.mark.parametrize("method", [pytest.param("sampled_eq5", marks=FLOOR), "exact"])
def test_T1_identity(method):
    ref = sphere(30.0)
    e = measure(method, np.asarray(ref.vertices, float), ref, "sphere")
    assert np.median(e) < TOL_MM, f"{method}: median error of a surface against itself = {np.median(e):.3f} mm"


@pytest.mark.parametrize(
    "method, d",
    [
        pytest.param("sampled_eq5", 0.2, marks=FLOOR),
        pytest.param("sampled_eq5", 0.5, marks=FLOOR),
        ("sampled_eq5", 2.0),
        ("exact", 0.2),
        ("exact", 0.5),
        ("exact", 2.0),
    ],
)
def test_T2_sphere_offset(method, d):
    ref, rec = sphere(30.0), sphere(30.0 + d)
    e = measure(method, np.asarray(rec.vertices, float), ref, "sphere")
    assert abs(np.median(e) - d) < TOL_MM, f"{method}: known offset {d} mm measured as {np.median(e):.3f} mm"
    assert np.percentile(e, 99) - d < 0.1, f"{method}: P99 {np.percentile(e, 99):.3f} for a uniform offset of {d}"


def test_T3_exact_le_sampled_on_cache():
    path = os.path.join(CACHE, "z002_foot.nii.gz.npz")
    if not os.path.exists(path):
        pytest.skip("foot cache not present (run a4_run_foot.py)")
    z = np.load(path, allow_pickle=False)
    V, q, e_s = z["V"].astype(float), z["q"].astype(float), z["e"].astype(float)
    e_x = np.linalg.norm(V - q, axis=1)
    assert np.all(e_x <= e_s + 1e-4), "exact distance larger than the sampled one somewhere"  # float32 cache
    assert np.median(e_x) < 0.6 * np.median(e_s), "sampled error not dominated by the floor on this case (unexpected)"
