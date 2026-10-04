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
