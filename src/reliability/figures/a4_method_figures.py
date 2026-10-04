"""Figures of Section 2 of the paper: Fig. 1 framework; Fig. 2 shape-disagreement
descriptor; Fig. 3 intensity- vs surface-space uncertainty; Fig. 4 reliability-guided registration.

Outside the package hash (subfolder `figures/`: `a4_config.code_hash` only covers `src/reliability/*.py`), so as not
to alter the traceability of the caches generated in Phase 2. Reads only `output/validation/reliability/foot_cache/`
(new caches) and the 0.5 mm volume of the representative foot. Labels in English with the terms of the paper.
Output in
`output/figures/figs/` (PDF + PNG) and `figs/figures_provenance.json`.

Usage: python src/reliability/figures/a4_method_figures.py [--foot z002_foot.nii.gz]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import a4_features as X
import a4_config as C
import a4_data as D
import a4_statistics as E
import a4_registration as REG
from a4_grid import Grid

FIGS = os.path.join(
    C.ROOT, "output", "figures", "figs"
)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    }
)
CMAP = "viridis"
COLOR = {
    "ref": "#1f77b4",
    "sr": "#d62728",
    "interp": "#2ca02c",
    "field": "#d62728",
    "d_shape": "#ff7f0e",
    "random": "#7f7f7f",
    "oracle": "#9467bd",
    "global": "#17becf",
    "high": "#000000",
}
NAMES = C.FEATURE_NAMES if hasattr(C, "FEATURE_NAMES") else None


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, "%s.%s" % (name, ext)), bbox_inches="tight")
    plt.close(fig)
    print("figure saved:", name, flush=True)


def box(ax, xy, w, h, text, color="#f2f2f2", border="#444444", fs=7.5, lw=0.8):
    ax.add_patch(
        FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            fc=color,
            ec=border,
            lw=lw,
        )
    )
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        wrap=True,
    )


def arrow(ax, p0, p1, color="#444444", style="-|>", lw=0.9, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=9,
            color=color,
            lw=lw,
            linestyle=ls,
            shrinkA=1,
            shrinkB=1,
        )
    )


# ----------------------------------------------------------------------------- Fig. 1
def fig1():
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")
    fs = 6.6
    ax.text(
        1,
        58.0,
        "Training / evaluation (uses the higher-resolution reference)",
        fontsize=8.5,
        weight="bold",
    )
    ax.text(
        1,
        27.0,
        "Target-case inference (no reference available)",
        fontsize=8.5,
        weight="bold",
    )
    ax.plot([0, 100], [29.3, 29.3], ls="--", color="#666666", lw=1)
    y1, h, w = 44, 9, 17
    xs = [1, 21, 41, 61, 81]
    top = [
        "Higher-resolution CT\n(0.5 mm; 48 feet)",
        "Controlled thick-slice\ngeneration\n(k = 6; Eq. 1–2)",
        "Through-plane SR\n(self-supervised;\n§2.4)",
        "Segmentation and\nsurface extraction\n(§2.5)",
        "Vertex-wise features\nx(v) ∈ ℝ⁹\n(§2.7)",
    ]
    for x, t in zip(xs, top, strict=True):
        box(ax, (x, y1), w, h, t, fs=fs)
    for a, b in zip(xs[:-1], xs[1:], strict=True):
        arrow(ax, (a + w, y1 + h / 2), (b, y1 + h / 2))
    yb, hb = 31.5, 8
    box(
        ax,
        (21, yb),
        w,
        hb,
        "Reference surface\n(same segmentation\nand meshing)",
        color="#dbe9f6",
        border=COLOR["ref"],
        fs=fs,
    )
    arrow(ax, (9.5, y1), (29.5, yb + hb), color=COLOR["ref"])
    box(
        ax,
        (41, yb),
        w,
        hb,
        "Vertex-wise geometric\nerror e(v) (Eq. 5)",
        color="#dbe9f6",
        border=COLOR["ref"],
        fs=fs,
    )
    arrow(ax, (21 + w, yb + hb / 2), (41, yb + hb / 2), color=COLOR["ref"])
    arrow(ax, (69.5, y1), (69.5, yb + hb), color=COLOR["ref"])
    box(
        ax,
        (61, yb),
        w,
        hb,
        "Random-forest\nregression (§2.7.3);\nleave-one-case-out",
        color="#fde8e8",
        border=COLOR["sr"],
        fs=fs,
    )
    arrow(ax, (41 + w, yb + hb / 2), (61, yb + hb / 2), color=COLOR["ref"])
    arrow(ax, (89.5, y1), (89.5, yb + hb))
    box(
        ax,
        (81, yb),
        18,
        hb,
        "Evaluation (§2.9–2.11):\nuncertainty baselines;\nreal paired CT;\nregistration",
        color="#eeeeee",
        fs=fs,
    )
    arrow(ax, (61 + w, yb + hb / 2), (81, yb + hb / 2))
    y2 = 13
    bot = [
        "Thick-slice CT\n(target case)",
        "Through-plane SR\n(same network\nfamily)",
        "Segmentation and\nsurface extraction",
        "Vertex-wise features\nx(v) ∈ ℝ⁹",
        "Predicted reliability\nfield ê(v) in mm\n(Eq. 19)",
    ]
    for x, t in zip(xs, bot, strict=True):
        box(
            ax,
            (x, y2),
            w,
            h,
            t,
            color="#fff7e6" if "ê" in t else "#f2f2f2",
            border=COLOR["sr"] if "ê" in t else "#444444",
            fs=fs,
        )
    for a, b in zip(xs[:-1], xs[1:], strict=True):
        arrow(ax, (a + w, y2 + h / 2), (b, y2 + h / 2))
    arrow(ax, (69.5, yb), (89.5, y2 + h), color=COLOR["sr"], ls="--")
    ax.text(76, 24.3, "trained model", color=COLOR["sr"], fontsize=6.5, ha="center")
    ax.text(
        1,
        4,
        "Reference data enter only the upper lane; the target case is processed without any reference.",
        fontsize=7,
        style="italic",
    )
    save(fig, "fig1_framework")


# ----------------------------------------------------------------------------- data of the representative foot
def load_foot(ident):
    listing = json.load(open(C.FOOT_LIST_N48))
    filepath = [p for p in listing if os.path.basename(p) == ident][0]
    z = np.load(os.path.join(C.A4_CACHE_FOOT, ident + ".npz"), allow_pickle=False)
    cache = {k: z[k] for k in z.files if k != "meta"}
    cache["meta"] = json.loads(str(z["meta"]))
    return filepath, cache


def foot_geometry(filepath, k=C.K_FOOT):
    """Geometric chain of a4_run_foot up to S_interp (without SR): calls a4_run_foot.prepare_foot, the same
 function used by the pipeline and by the recomputation of x6, instead of duplicating it."""
    from a4_run_foot import prepare_foot

    hu_t, thick, grid, spacing, thick_spacing, _roi, thick_roi = prepare_foot(filepath, k)
    thick_grid = Grid(
        thick_spacing,
        grid.index_to_physical(np.array([[0.0, 0.0, (k - 1) / 2.0]]))[0],
        grid.D.reshape(-1),
        thick.shape,
    )
    m_thick = X.thick_mask_sinterp(
        thick, thick_roi, thick_spacing
    )  # same thick mask as the pipeline
    thick_phi = X.sdf_mm(m_thick, thick_spacing)
    s_interp = X.interpolated_surface(m_thick, k, grid)
    del hu_t
    return dict(
        thick=thick,
        grid=grid,
        thick_grid=thick_grid,
        spacing=spacing,
        thick_spacing=thick_spacing,
        m_thick=m_thick,
        thick_phi=thick_phi,
        s_interp=s_interp,
        k=k,
    )


def projection_2d(V):
    """Orthographic projection onto the plane of the two principal directions (PCA) of the mesh — view of largest
 extent."""
    c = V.mean(axis=0)
    u, s, vt = np.linalg.svd(V - c, full_matrices=False)
    P = (V - c) @ vt[:2].T
    depth = (V - c) @ vt[2]
    return P, depth


def _data_dir():
    d = os.path.join(FIGS, "data")
    os.makedirs(d, exist_ok=True)
    return d


def export(name, **columns):
    """writes to figs/data/<name>.csv the columns (same length) exactly as plotted."""
    import csv

    cols = {k: np.asarray(v).reshape(-1) for k, v in columns.items()}
    n = {len(v) for v in cols.values()}
    assert len(n) == 1, (name, {k: len(v) for k, v in cols.items()})
    with open(
        os.path.join(_data_dir(), name + ".csv"), "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(list(cols))
        for i in range(n.pop()):
            w.writerow([cols[k][i] for k in cols])


def export_matrix(name, M, extent_mm):
    """2D image as CSV (rows = z, columns = y) + physical extent [y0, y1, z1, z0] in mm in the header."""
    with open(os.path.join(_data_dir(), name + ".csv"), "w", encoding="utf-8") as f:
        f.write(
            "# extent_mm (y0, y1, z_bottom, z_top) = %s\n"
            % (list(map(float, extent_mm)),)
        )
        np.savetxt(f, np.asarray(M, float), delimiter=",", fmt="%.6g")


def mesh_scatter(
    ax, P, depth, val, title, label, vmin=None, vmax=None, cmap=CMAP, s=0.6
):
    o = np.argsort(depth)  # paints first what is behind
    sc = ax.scatter(
        P[o, 0],
        P[o, 1],
        c=val[o],
        s=s,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
        linewidths=0,
    )
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label)
    cb.ax.tick_params(labelsize=7)
    return sc


# ----------------------------------------------------------------------------- Fig. 2
def fig2(ident, cache, geo):
    V = cache["V"].astype(float)
    F = cache["F"]
    d_shape = F[:, 5].astype(float)
    grid, k = geo["grid"], geo["k"]
    thick, m_thick, thick_phi = geo["thick"], geo["m_thick"], geo["thick_phi"]
    # sagittal plane (x fixed) at the median x of the vertices, in thin index
    idx = grid.physical_to_index(V)
    ix = int(np.round(np.median(idx[:, 0])))
    # sagittal slice of the thick volumes: arrays (z_thick, y)
    ct2 = thick[:, :, ix]
    m2 = m_thick[:, :, ix]
    phi2 = thick_phi[:, :, ix]
    thin_phi = D.interpolate_trilinear_z(thick_phi, k)  # Eq. 16 (same function as S_interp)
    phi2f = thin_phi[:, :, ix]
    del thin_phi
    sy, thick_sz, sz = grid.spacing[1], geo["thick_spacing"][2], grid.spacing[2]
    thick_ext = [
        0,
        ct2.shape[1] * sy,
        ct2.shape[0] * thick_sz,
        0,
    ]  # x = y (mm), y = z (mm), origin at the top
    thin_ext = [0, phi2f.shape[1] * sy, phi2f.shape[0] * sz, 0]
    # window in y/z: box of the vertices ± margin
    ymm = (
        max(0.0, idx[:, 1].min() * sy - 10),
        min(thick_ext[1], idx[:, 1].max() * sy + 10),
    )
    zmm = (
        max(0.0, idx[:, 2].min() * sz - 10),
        min(thick_ext[2], idx[:, 2].max() * sz + 10),
    )
    # SR vertices near the plane (|x − x0| ≤ 0.5 mm)
    near = np.abs(idx[:, 0] - ix) * grid.spacing[0] <= 0.5
    # S_interp: intersection with the plane
    o = grid.index_to_physical(np.array([[ix, 0, 0]], float))[0]
    n = grid.D[:, 0]
    sec = geo["s_interp"].section(plane_origin=o, plane_normal=n)
    fig, axs = plt.subplots(2, 3, figsize=(7.5, 5.8))
    ax = axs.ravel()
    ax[0].imshow(
        ct2,
        cmap="gray",
        vmin=-500,
        vmax=1500,
        extent=thick_ext,
        aspect="equal",
        interpolation="nearest",
    )
    ax[0].set_title("(a) Thick-slice CT (sagittal)")
    ax[1].imshow(
        m2, cmap="gray", extent=thick_ext, aspect="equal", interpolation="nearest"
    )
    ax[1].set_title("(b) Thick-slice bone mask (§2.5 mask, thick grid)")
    lim = (
        float(np.percentile(np.abs(phi2[m2 | (phi2 < 8)]), 99))
        if (m2 | (phi2 < 8)).any()
        else 8.0
    )
    im = ax[2].imshow(
        phi2,
        cmap="RdBu_r",
        vmin=-lim,
        vmax=lim,
        extent=thick_ext,
        aspect="equal",
        interpolation="nearest",
    )
    ax[2].set_title("(c) SDF φ, thick grid (mm)")
    plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
    im = ax[3].imshow(
        phi2f,
        cmap="RdBu_r",
        vmin=-lim,
        vmax=lim,
        extent=thin_ext,
        aspect="equal",
        interpolation="nearest",
    )
    ax[3].set_title("(d) Interpolated φ̃ (mm)")
    plt.colorbar(im, ax=ax[3], fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
    ax[4].imshow(
        ct2,
        cmap="gray",
        vmin=-500,
        vmax=1500,
        extent=thick_ext,
        aspect="equal",
        interpolation="nearest",
        alpha=0.6,
    )
    if sec is not None:
        for ent in sec.entities:
            pts = sec.vertices[ent.points]
            pi = grid.physical_to_index(pts)
            ax[4].plot(pi[:, 1] * sy, pi[:, 2] * sz, color=COLOR["interp"], lw=0.9)
    ax[4].scatter(
        idx[near, 1] * sy,
        idx[near, 2] * sz,
        s=1.2,
        color=COLOR["sr"],
        linewidths=0,
        rasterized=True,
    )
    ax[4].plot([], [], color=COLOR["interp"], lw=1.2, label="S_interp = {φ̃ = 0}")
    ax[4].scatter(
        [], [], s=6, color=COLOR["sr"], label="SR surface vertices (|Δx| ≤ 0.5 mm)"
    )
    ax[4].legend(fontsize=5.5, loc="upper left", frameon=True)
    ax[4].set_title("(e) S_interp = {φ̃ = 0} vs. SR surface")
    for a in ax[:5]:
        a.set_xlim(ymm)
        a.set_ylim(zmm[1], zmm[0])
        a.set_xticks([])
        a.set_yticks([])
    P, depth = projection_2d(V)
    mesh_scatter(
        ax[5],
        P,
        depth,
        np.clip(d_shape, 0, 2),
        "(f) d_shape(v), SR surface",
        "d_shape (mm; clipped at 2)",
        s=0.35,
    )
    fig.suptitle(
        "Shape-disagreement descriptor x₆ = d_shape (Eq. 15–18), foot %s"
        % ident.split("_")[0],
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, "fig2_shape_disagreement")
    export_matrix("fig2a_thick_ct_sagittal_HU", ct2, thick_ext)
    export_matrix("fig2b_thick_mask", m2.astype(int), thick_ext)
    export_matrix("fig2c_thick_sdf_mm", phi2, thick_ext)
    export_matrix("fig2d_interpolated_sdf_mm", phi2f, thin_ext)
    if sec is not None:
        segs = [
            (ie, grid.physical_to_index(sec.vertices[ent.points]))
            for ie, ent in enumerate(sec.entities)
        ]
        export(
            "fig2e_sinterp_section",
            segment=np.concatenate([[ie] * len(pi) for ie, pi in segs]),
            y_mm=np.concatenate([pi[:, 1] * sy for _, pi in segs]),
            z_mm=np.concatenate([pi[:, 2] * sz for _, pi in segs]),
        )
    export(
        "fig2e_sr_vertices_on_plane",
        y_mm=idx[near, 1] * sy,
        z_mm=idx[near, 2] * sz,
    )
    export(
        "fig2f_vertices_dshape",
        x_mm=V[:, 0],
        y_mm=V[:, 1],
        z_mm=V[:, 2],
        proj_u=P[:, 0],
        proj_v=P[:, 1],
        depth=depth,
        d_shape_mm=d_shape,
    )
    return {
        "sagittal_plane_index_x": ix,
        "vertices_on_plane": int(near.sum()),
        "d_shape_median_mm": float(np.median(d_shape)),
    }


# ----------------------------------------------------------------------------- Fig. 3
def loo_prediction(ident, cache):
    """ê(v) of the held-out foot read from the LOFO memo of the full model (rf9_sr) — the SAME ê(v) reported in r32/r33
 (the forest is selected per fold; redoing it here with fixed RF_PARAMS would leave the figure outside the reported
 model).
 Locates the memo by the key recorded in rf_selection_rf9_sr_<sampler>.json; without nested selection
 (RF_SELECTION = "fixed"), uses the most recent memo of rf9_sr."""
    if C.RF_SELECTION == "nested":
        j = json.load(
            open(
                os.path.join(
                    C.A4_RESULTS, "rf_selection_rf9_sr_%s.json" % C.RF_SEARCH_SAMPLER
                ),
                encoding="utf-8",
            )
        )
        p = os.path.join(C.A4_RESULTS, "_lofo_rf9_sr_%s.npz" % j["key"])
    else:
        cands = sorted(
            glob.glob(os.path.join(C.A4_RESULTS, "_lofo_rf9_sr_*.npz")),
            key=os.path.getmtime,
        )
        p = cands[-1]
    z = np.load(p, allow_pickle=False)
    return z["pred_" + ident]


def fig3(ident, cache, e_hat):
    V = cache["V"].astype(float)
    P, depth = projection_2d(V)
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15])
    axs = fig.add_subplot(gs[0, :])
    axs.set_xlim(0, 100)
    axs.set_ylim(0, 30)
    axs.axis("off")
    routes = [
        (
            "(a) Intensity-space uncertainty (§2.8.1–2.8.2)",
            [
                "Repeated reconstructions\n(ensemble M = 5; MC dropout S = 4;\nDropsembles)",
                "Voxel-wise variability\n(standard deviation)",
                "Uncertainty in HU\nsampled at vertices (Eq. 27)",
            ],
            "#e8eef7",
        ),
        (
            "(b) Surface-space uncertainty (§2.8.3)",
            [
                "Repeated reconstructions",
                "Segmentation → signed\ndistance fields (Eq. 31)",
                "Surface uncertainty in mm\n(u_geo,std, u_geo,abs, u_geo,MC)",
            ],
            "#e9f5e9",
        ),
        (
            "(c) Proposed reliability field (§2.7)",
            [
                "Single reconstructed\ntarget surface",
                "Nine vertex-wise\ndescriptors x(v)",
                "Predicted geometric\nerror ê(v) in mm",
            ],
            "#fdf0e6",
        ),
    ]
    for i, (tit, steps, color) in enumerate(routes):
        y = 21 - i * 10
        axs.text(0, y + 7.2, tit, fontsize=8, weight="bold")
        for j, t in enumerate(steps):
            x = j * 34
            box(axs, (x, y), 30, 6.5, t, color=color, fs=6.8)
            if j < 2:
                arrow(axs, (x + 30, y + 3.25), (x + 34, y + 3.25))
    vmax_hu = float(np.percentile(cache["u_ens"], 99))
    vmax_mm = float(
        max(np.percentile(cache["u_geo_std"], 99), np.percentile(e_hat, 99))
    )
    ax1 = fig.add_subplot(gs[1, 0])
    mesh_scatter(
        ax1,
        P,
        depth,
        cache["u_ens"].astype(float),
        "u_ens(v)\nintensity space",
        "HU",
        vmin=0,
        vmax=vmax_hu,
    )
    ax2 = fig.add_subplot(gs[1, 1])
    mesh_scatter(
        ax2,
        P,
        depth,
        cache["u_geo_std"].astype(float),
        "u_geo,std(v)\nsurface space",
        "mm",
        vmin=0,
        vmax=vmax_mm,
    )
    ax3 = fig.add_subplot(gs[1, 2])
    mesh_scatter(
        ax3,
        P,
        depth,
        e_hat,
        "ê(v)\npredicted geometric error",
        "mm",
        vmin=0,
        vmax=vmax_mm,
    )
    fig.suptitle(
        "Intensity-space vs. surface-space uncertainty and the predicted reliability field (foot %s, same SR surface)"
        % ident.split("_")[0],
        fontsize=8.5,
    )
    fig.tight_layout()
    save(fig, "fig3_uncertainty_spaces")
    export(
        "fig3_vertices_uncertainties_field",
        x_mm=cache["V"][:, 0],
        y_mm=cache["V"][:, 1],
        z_mm=cache["V"][:, 2],
        proj_u=P[:, 0],
        proj_v=P[:, 1],
        depth=depth,
        u_ens_HU=cache["u_ens"],
        u_geo_std_mm=cache["u_geo_std"],
        e_hat_mm=e_hat,
        e_measured_mm=cache["e"],
    )
    return {
        "u_ens_median_HU": float(np.median(cache["u_ens"])),
        "u_geo_std_median_mm": float(np.median(cache["u_geo_std"])),
        "e_hat_median_mm": float(np.median(e_hat)),
    }


# ----------------------------------------------------------------------------- Fig. 4
def fig4(ident, cache, e_hat):
    V = cache["V"].astype(float)
    q = cache["q"].astype(float)
    e = cache["e"].astype(float)
    d_shape = cache["F"][:, 5].astype(float)
    K, p_target = C.REG_K, C.REG_TARGET_PERCENTILE
    assert C.REG_SELECTION == "kmeans", (
        "the figure illustrates the k-means regions"
    )
    rng = np.random.default_rng(C.REG_SEED)
    rot = E.regions(V)  # the same 25 regions of §2.9.2 (Eq. 36-37)
    regs = np.unique(rot)
    members = {r: np.where(rot == r)[0] for r in regs}
    eligible = np.array([r for r in regs if len(members[r]) >= K])
    vert_elig = np.concatenate([members[r] for r in eligible])

    def smallest(x):  # region with the lowest mean of the criterion (Eq. 38-39), among the eligible ones
        med = np.array([x[members[r]].mean() for r in eligible])
        return eligible[int(np.argmin(med))]

    def largest(x):
        med = np.array([x[members[r]].mean() for r in eligible])
        return eligible[int(np.argmax(med))]

    def far_target(reg):  # target beyond the p80 of the distance to the WHOLE region
        d_reg = cKDTree(V[members[reg]]).query(V)[0]
        far = np.where(d_reg > np.percentile(d_reg, p_target))[0]
        return int(far[rng.integers(len(far))])

    def displ(loc, target):
        Rm, t = REG.rigid_umeyama(V[loc], q[loc])
        x = q[target]
        return float(np.linalg.norm(Rm @ x + t - x))

    sel = {"field": smallest(e_hat), "d_shape": smallest(d_shape), "oracle": smallest(e)}
    sel["random"] = rot[
        vert_elig[rng.integers(len(vert_elig))]
    ]  # "random surface selection": uniform vertex
    region_of = dict(sel)
    locs = {
        s: rng.choice(members[r], size=K, replace=False) for s, r in sel.items()
    }  # 200 drawn INSIDE
    locs["global"] = rng.choice(len(V), size=C.REG_GLOBAL_N, replace=False)
    targets = {}
    for s in locs:
        if s == "global":
            d_reg = cKDTree(V[locs[s]]).query(V)[0]
            far = np.where(d_reg > np.percentile(d_reg, p_target))[0]
            targets[s] = int(far[rng.integers(len(far))])
        else:
            targets[s] = far_target(region_of[s])
    dts = {s: displ(locs[s], targets[s]) for s in locs}
    # contrast: region of LOWEST vs HIGHEST mean e-hat, 300 trials each (200 drawn inside; target > p80)
    rng2 = np.random.default_rng(C.REG_SEED + 1)
    r_lo, r_hi = smallest(e_hat), largest(e_hat)

    def median_trials(reg, n=C.REG_TRIALS):
        ds = []
        for _ in range(n):
            loc = rng2.choice(members[reg], size=K, replace=False)
            d_reg = cKDTree(V[members[reg]]).query(V)[0]
            far = np.where(d_reg > np.percentile(d_reg, p_target))[0]
            ds.append(displ(loc, int(far[rng2.integers(len(far))])))
        return float(np.median(ds)), ds

    d_lo, ds_lo = median_trials(r_lo)
    d_hi, ds_hi = median_trials(r_hi)
    l_min, l_max = members[r_lo], members[r_hi]
    region_members = {s: members[r] for s, r in region_of.items()}
    P, depth = projection_2d(V)
    vmax = float(np.percentile(e_hat, 99))
    fig, axs = plt.subplots(2, 3, figsize=(7.5, 4.9), constrained_layout=True)
    ax = axs.ravel()
    order = [
        ("field", "Reliability-guided (lowest ê)"),
        ("d_shape", "Shape-disagreement (lowest d_shape)"),
        ("random", "Random surface selection"),
        ("oracle", "Oracle (lowest measured e)"),
        ("global", "Global surface (K = 200)"),
    ]
    o = np.argsort(depth)
    for a, (st, tit) in zip(ax[:5], order, strict=True):
        a.scatter(
            P[o, 0],
            P[o, 1],
            c=e_hat[o],
            s=0.4,
            cmap=CMAP,
            vmin=0,
            vmax=vmax,
            rasterized=True,
            linewidths=0,
        )
        if st in region_members:
            rm = region_members[st]
            a.scatter(
                P[rm, 0],
                P[rm, 1],
                s=1.5,
                color="#bbbbbb",
                linewidths=0,
                label="selected k-means region",
            )
        l = locs[st]
        a.scatter(
            P[l, 0],
            P[l, 1],
            s=2.5,
            color=COLOR[st],
            linewidths=0,
            label="K = 200 sampled correspondences",
        )
        a.scatter(
            [P[targets[st], 0]],
            [P[targets[st], 1]],
            s=40,
            marker="*",
            color="white",
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
            label="remote target (> p80)",
        )
        a.set_title("%s\nd_target = %.2f mm (one trial)" % (tit, dts[st]), fontsize=7)
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    ax[0].legend(fontsize=5.2, loc="lower left", frameon=True, handletextpad=0.3)
    a = ax[5]
    sc = a.scatter(
        P[o, 0],
        P[o, 1],
        c=e_hat[o],
        s=0.4,
        cmap=CMAP,
        vmin=0,
        vmax=vmax,
        rasterized=True,
        linewidths=0,
    )
    a.scatter(
        P[l_min, 0],
        P[l_min, 1],
        s=2.5,
        color=COLOR["field"],
        linewidths=0,
        label="lowest-ê region: median d = %.2f mm" % d_lo,
    )
    a.scatter(
        P[l_max, 0],
        P[l_max, 1],
        s=2.5,
        color=COLOR["high"],
        linewidths=0,
        label="highest-ê region: median d = %.2f mm" % d_hi,
    )
    a.set_title(
        "Contrast: lowest- vs highest-ê k-means regions\n(300 trials each; remote targets > p80)",
        fontsize=7,
    )
    a.legend(fontsize=5.2, loc="lower left", frameon=True, handletextpad=0.3)
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(sc, ax=axs, shrink=0.6, pad=0.01)
    cb.set_label("ê(v), predicted geometric error (mm)")
    cb.ax.tick_params(labelsize=7)
    fig.suptitle(
        "Reliability-guided rigid registration (§2.11): k-means regions (§2.9.2), K = 200 correspondences sampled within, Umeyama T_r, remote target (foot %s)"
        % ident.split("_")[0],
        fontsize=7.5,
    )
    save(fig, "fig4_registration")
    export(
        "fig4_vertices",
        x_mm=V[:, 0],
        y_mm=V[:, 1],
        z_mm=V[:, 2],
        proj_u=P[:, 0],
        proj_v=P[:, 1],
        depth=depth,
        e_hat_mm=e_hat,
        e_measured_mm=e,
        d_shape_mm=d_shape,
        kmeans_region=rot,
    )
    ests = [s_ for s_, _ in order]
    export(
        "fig4_correspondences_per_strategy",
        strategy=np.concatenate([[s_] * len(locs[s_]) for s_ in ests]),
        vertex_index=np.concatenate([locs[s_] for s_ in ests]),
    )
    export(
        "fig4_strategies_one_trial",
        strategy=ests,
        kmeans_region=[region_of.get(s_, -1) for s_ in ests],
        target_index=[targets[s_] for s_ in ests],
        d_target_mm=[dts[s_] for s_ in ests],
    )
    export(
        "fig4_contrast_300_trials",
        region=["lowest_e_hat"] * len(ds_lo) + ["highest_e_hat"] * len(ds_hi),
        d_target_mm=list(ds_lo) + list(ds_hi),
    )
    export(
        "fig4_contrast_regions",
        region=["lowest_e_hat", "highest_e_hat"],
        kmeans_region=[int(r_lo), int(r_hi)],
        median_d_mm=[d_lo, d_hi],
    )  
    return {
        "d_target_one_trial_mm_": dts,
        "median_contrast_300_trials_mm": {"lowest_e_hat": d_lo, "highest_e_hat": d_hi},
        "contrast_iqr_mm": {
            "lowest_e_hat": [
                float(np.percentile(ds_lo, 25)),
                float(np.percentile(ds_lo, 75)),
            ],
            "highest_e_hat": [
                float(np.percentile(ds_hi, 25)),
                float(np.percentile(ds_hi, 75)),
            ],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--foot",
        default="z002_foot.nii.gz",
        help="representative foot (median of e(v) = cohort median)",
    )
    ap.add_argument("--only", default="", help="1,2,3,4")
    args = ap.parse_args()
    which = set(args.only.split(",")) if args.only else {"1", "2", "3", "4"}
    prov = {
        "foot": args.foot,
        "environment": C.environment_record(with_torch=False),
        "figures": {},
    }
    if "1" in which:
        fig1()
        prov["figures"]["fig1"] = "schematic (no data)"
    if which & {"2", "3", "4"}:
        filepath, cache = load_foot(args.foot)
        prov["cache_hash"] = cache["meta"]["environment"]["code_hash_reliability"]
        if "2" in which:
            geo = foot_geometry(filepath)
            prov["figures"]["fig2"] = fig2(args.foot, cache, geo)
            del geo
        if which & {"3", "4"}:
            e_hat = loo_prediction(args.foot, cache)
            prov["e_hat_rho_spearman_with_e"] = E.spearman(e_hat, cache["e"])
            if "3" in which:
                prov["figures"]["fig3"] = fig3(args.foot, cache, e_hat)
            if "4" in which:
                prov["figures"]["fig4"] = fig4(args.foot, cache, e_hat)
    os.makedirs(FIGS, exist_ok=True)
    json.dump(
        prov,
        open(os.path.join(FIGS, "figures_provenance.json"), "w", encoding="utf-8"),
        indent=1,
        ensure_ascii=False,
    )
    print(json.dumps(prov, ensure_ascii=False, indent=None)[:1500])


if __name__ == "__main__":
    main()
