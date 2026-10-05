"""Rendering of the meshes in the clean figures of the paper, in three modes:
- "points": cloud of colored points;
- "shaded": the same vertices, with Lambertian shading from the normals stored in the cache (route A);
- "surface": triangles of the mesh re-extracted by a4_mesh_faces.py, painted by the mean value of the vertices and
 shaded by the face normal (route B).
Nothing is recomputed in the values: the color is still the per-vertex value read from the CSV; only the drawing
geometry changes.
The frame (u, v, depth) of each figure is recovered by least squares from the cache vertices
(the projection is orthonormal), and checked against the CSV columns.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
import numpy as np
from matplotlib import cm
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, to_rgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from reliability import a4_config as C  # noqa: E402

CACHE_FOOT = C.A4_CACHE_FOOT  # honours A4_OUTPUT_DIR like the rest of the package
MESHES = os.path.join(
    ROOT, "output", "figures",
    "figs",
    "data",
    "meshes",
)

LIGHT = np.array(
    [0.35, 0.45, 0.82]
)  # light direction in the frame (u, v, depth); depth points toward the camera
LIGHT = LIGHT / np.linalg.norm(LIGHT)
ENVIRONMENT = 0.45  # ambient fraction; the rest is diffuse
EDGE_MAX_MM = 6.0  # sliver: edge larger than this AND...
THINNESS_MIN = 0.05  #... area/edge^2 smaller than this (equilateral triangle = 0.43)
_CACHE: dict[str, dict] = {}


def _load(foot):
    if foot not in _CACHE:
        z = np.load(os.path.join(CACHE_FOOT, foot + ".npz"), allow_pickle=False)
        d = {"V": z["V"].astype(float), "N": z["N"].astype(float)}
        p = os.path.join(MESHES, foot + ".npz")
        if os.path.exists(p):
            m = np.load(p, allow_pickle=False)
            d["V_regen"] = m["V"].astype(float)
            d["faces"] = m["faces"].astype(int)
            d["idx_cache"] = m["idx_cache"].astype(int)
            d["verification"] = json.loads(str(m["verification"]))
        _CACHE[foot] = d
    return _CACHE[foot]


def frame(foot, u, v, depth):
    """Rotation R (3x3) and center c such that [u v depth] = (V - c) @ R, fitted to the cache vertices; checks the fit."""
    d = _load(foot)
    V = d["V"]
    assert len(V) == len(u), (foot, len(V), len(u))
    Y = np.c_[u, v, depth]
    c = V.mean(axis=0)
    X = V - c
    R, *_ = np.linalg.lstsq(np.c_[X, np.ones(len(X))], Y, rcond=None)
    Rrot, t = R[:3], R[3]
    res = np.abs(X @ Rrot + t - Y).max()
    assert res < 1e-2, ("projection not reproduced", foot, res)
    assert np.allclose(Rrot @ Rrot.T, np.eye(3), atol=1e-3), "rotation is not orthonormal"
    d["R"], d["t"], d["c"] = Rrot, t, c
    return Rrot, t, c


def _shadow(n_uvp):
    n = n_uvp / np.maximum(np.linalg.norm(n_uvp, axis=1, keepdims=True), 1e-9)
    lam = np.abs(n @ LIGHT)  # open surface: normals may point to either side
    return ENVIRONMENT + (1 - ENVIRONMENT) * lam


def colors(val, vmin, vmax, cmap):
    return matplotlib.colormaps[cmap](Normalize(vmin, vmax, clip=True)(val))[:, :3]


def shaded_points(ax, foot, u, v, depth, val, vmin, vmax, cmap="viridis", s=3.0):
    R, _t, _c = frame(foot, u, v, depth)
    n = _load(foot)["N"] @ R
    rgb = colors(val, vmin, vmax, cmap) * _shadow(n)[:, None]
    o = np.argsort(depth)
    ax.scatter(u[o], v[o], c=rgb[o], s=s, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    return cm.ScalarMappable(Normalize(vmin, vmax), cmap)


def surface(ax, foot, u, v, depth, val, vmin, vmax, cmap="viridis", masks=()):
    """Triangles of the re-extracted mesh, value = mean of the 3 vertices (via the closest cache vertex), painter's
 algorithm ordered by mean depth. `masks` = [(bool per cache vertex, color),...]: faces entirely
 inside each mask are painted with the given color (no value shading, but with the relief)."""
    d = _load(foot)
    assert "faces" in d, (
        f"mesh with faces not found for{foot} (run a4_mesh_faces.py)"
    )
    R, t, c = frame(foot, u, v, depth)
    P = (d["V_regen"] - c) @ R + t  # (u, v, depth) of the regenerated vertices
    F = d["faces"]
    # degenerate faces from the decimation (long thin slivers linking distant parts) are discarded ONLY in the drawing:
    # edge larger than EDGE_MAX_MM and area/edge^2 smaller than THINNESS_MIN; recorded in the provenance
    T3 = d["V_regen"][F]
    edge = np.linalg.norm(
        np.stack([T3[:, 1] - T3[:, 0], T3[:, 2] - T3[:, 1], T3[:, 0] - T3[:, 2]]),
        axis=2,
    ).max(0)
    area = 0.5 * np.linalg.norm(
        np.cross(T3[:, 1] - T3[:, 0], T3[:, 2] - T3[:, 0]), axis=1
    )
    sliver = (edge > EDGE_MAX_MM) & (area / np.maximum(edge**2, 1e-9) < THINNESS_MIN)
    d["discarded_faces"] = {
        "edge_longer_than_mm": EDGE_MAX_MM,
        "area_over_edge2_below": THINNESS_MIN,
        "n_faces": int(sliver.sum()),
        "n_faces_total": len(F),
    }
    F = F[~sliver]
    vv = val[d["idx_cache"]]  # value per regenerated vertex
    fval = vv[F].mean(axis=1)
    tri = P[F]  # (nf, 3, 3)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    rgb = colors(fval, vmin, vmax, cmap)
    for mask, color in masks:
        inside = np.asarray(mask, bool)[d["idx_cache"]][F].all(axis=1)
        rgb[inside] = np.asarray(to_rgb(color), float)
    rgb = rgb * _shadow(normal)[:, None]
    o = np.argsort(tri[:, :, 2].mean(axis=1))  # far first
    pc = PolyCollection(
        tri[o][:, :, :2],
        facecolors=rgb[o],
        edgecolors=rgb[o],
        linewidths=0.15,  # covers the seams between neighboring triangles
        antialiaseds=True,
        rasterized=True,
    )
    ax.add_collection(pc)
    for (
        mask,
        color,
    ) in (
        masks
    ):  # hidden part of the region (opposite side): faded points on top
        k = np.asarray(mask, bool)
        ax.scatter(
            u[k],
            v[k],
            s=0.6,
            color=color,
            alpha=0.3,
            linewidths=0,
            zorder=3,
            rasterized=True,
        )
    ax.set_xlim(P[:, 0].min(), P[:, 0].max())
    ax.set_ylim(P[:, 1].min(), P[:, 1].max())
    ax.set_aspect("equal")
    return cm.ScalarMappable(Normalize(vmin, vmax), cmap)


def verification(foot):
    d = _load(foot)
    c = dict(d.get("verification") or {})
    if "discarded_faces" in d:
        c["faces_discarded_in_drawing"] = d["discarded_faces"]
    return c
