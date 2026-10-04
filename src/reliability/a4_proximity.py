"""Exact point-to-triangulated-surface distance (paper Eq. 18) and closest point
(noiseless correspondences of §2.11).

Method, guaranteed exact:
 (i) upper bound d_ub(p) = distance to the nearest mesh vertex (kd-tree);
 (ii) candidates = triangles whose centroid is at distance <= d_ub(p) + r_max from p, with r_max the largest
 centroid->vertex radius among the triangles: the nearest triangle contains a point at distance
 d* <= d_ub, so its centroid is at distance <= d* + r_max <= d_ub + r_max — it is among the candidates;
 (iii) exact point-triangle distance (trimesh.triangles.closest_point) in batch; minimum.
Zero-area triangles are removed beforehand (they do not belong to the surface and produce division by zero).
No dependency on `rtree`.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def without_degenerate(mesh: trimesh.Trimesh, eps: float = 1e-12) -> trimesh.Trimesh:
    m = mesh.copy()
    ok = m.area_faces > eps
    if not ok.all():
        m.update_faces(ok)
        m.remove_unreferenced_vertices()
    return m


def nearest_exact(P: np.ndarray, mesh: trimesh.Trimesh, batch: int = 2000):
    """Returns (d, q): d (n,) exact distance in mm; q (n,3) closest point on the surface."""
    P = np.asarray(P, float)
    m = without_degenerate(mesh)
    tri = m.triangles
    cent = tri.mean(axis=1)
    r_max = float(np.sqrt(((tri - cent[:, None, :]) ** 2).sum(-1)).max())
    d_ub = cKDTree(np.asarray(m.vertices)).query(P)[0]
    tree = cKDTree(cent)
    d = np.full(len(P), np.nan)
    q = np.full((len(P), 3), np.nan)
    for i0 in range(0, len(P), batch):
        i1 = min(len(P), i0 + batch)
        cand = tree.query_ball_point(P[i0:i1], d_ub[i0:i1] + r_max + 1e-9)
        for j, cj in enumerate(cand):
            cj = np.asarray(cj, int)
            p = P[i0 + j]
            qq = trimesh.triangles.closest_point(
                tri[cj], np.repeat(p[None, :], len(cj), axis=0)
            )
            dd = np.sqrt(((qq - p) ** 2).sum(-1))
            k = int(np.nanargmin(dd))
            d[i0 + j] = dd[k]
            q[i0 + j] = qq[k]
    assert not np.isnan(d).any(), "exact distance produced NaN"
    return d, q


def nearest_exact_bounded(P, upper, mesh, max_mm=20.0, large_tri_mm=5.0):
    """Exact point-to-triangle distance for the points with upper <= max_mm (the others keep `upper`).

    `upper` is an upper bound of the exact distance (e.g. the distance to the nearest sampled point). Small
    triangles are found with a KD-tree on their centroids within upper + large_tri_mm; the few large triangles left
    by decimation are tested against every point. Same result as nearest_exact, which uses one radius for all
    triangles and becomes very slow when the reference has a long sliver triangle."""
    P, upper = np.asarray(P, float), np.asarray(upper, float)
    e = upper.copy()
    near = np.flatnonzero(upper <= max_mm)
    if len(near) == 0:
        return e
    m = without_degenerate(mesh)
    tri = np.asarray(m.triangles, float)
    cent = tri.mean(axis=1)
    rad = np.sqrt(((tri - cent[:, None, :]) ** 2).sum(-1)).max(axis=1)
    small, large = np.flatnonzero(rad <= large_tri_mm), np.flatnonzero(rad > large_tri_mm)
    Pn = P[near]
    best = np.full(len(near), np.inf)
    if len(large):
        for i0 in range(0, len(near), 4000):
            blk = Pn[i0 : i0 + 4000]
            for t in large:
                qq = trimesh.triangles.closest_point(np.repeat(tri[t][None], len(blk), axis=0), blk)
                best[i0 : i0 + 4000] = np.minimum(best[i0 : i0 + 4000], np.sqrt(((qq - blk) ** 2).sum(-1)))
    tree = cKDTree(cent[small])
    ub = upper[near] + 1e-9
    for i0 in range(0, len(near), 2000):
        i1 = min(len(near), i0 + 2000)
        cand = tree.query_ball_point(Pn[i0:i1], ub[i0:i1] + large_tri_mm)
        for j, cj in enumerate(cand):
            if not cj:
                continue
            cj = small[np.asarray(cj, int)]
            p = Pn[i0 + j]
            qq = trimesh.triangles.closest_point(tri[cj], np.repeat(p[None, :], len(cj), axis=0))
            best[i0 + j] = min(best[i0 + j], float(np.sqrt(((qq - p) ** 2).sum(-1)).min()))
    assert np.all(np.isfinite(best)) and np.all(best <= ub + 1e-3), "bounded search missed the nearest triangle"
    e[near] = best
    return e


def brute_force(P: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    """Validation: exact distance against ALL the triangles (only for small samples)."""
    m = without_degenerate(mesh)
    tri = m.triangles
    out = np.empty(len(P))
    for i, p in enumerate(np.asarray(P, float)):
        qq = trimesh.triangles.closest_point(
            tri, np.repeat(p[None, :], len(tri), axis=0)
        )
        out[i] = np.sqrt(((qq - p) ** 2).sum(-1)).min()
    return out
