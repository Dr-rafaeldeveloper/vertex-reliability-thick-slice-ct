"""Final set of figures of the paper: same style as a4_clean_figures.py (Arial,
0.6 pt, white background, vector PDF + 600 dpi PNG), with a pipeline
infographic built from real foot images and a thorax figure. Reads only CSVs from figs/data/ and
results_figure_data/ and the meshes from figs/data/meshes/; no value is recomputed.

Figures (file name = content; the numbering is the manuscript's):
 fig_pipeline pipeline infographic with CT slices and surfaces of foot z002 (Methods)
 fig_field_vs_error e(v) x ê(v) on the representative foot + rho per foot
 fig_calibration calibration by deciles
 fig_uncertainty e(v), u_ens, u_geo,std and ê on the same surface
 fig_shape_disagreement construction of S_interp and d_shape
 fig_thorax thorax: 5 mm / SR / 1 mm slices, e(v) x ê(v) on the representative case, rho per case
 fig_registration five selection strategies + displacement distribution (smaller x larger ê)

Output: output/figures/figs/final/ and copy of the PDFs in output/figures/figs/final/pdf/.
Usage: python src/reliability/figures/a4_final_figures.py
 [--only pipeline,field,calibration,uncertainty,shape,thorax,registration]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a4_clean_figures as L
import a4_mesh_render as RENDER

OUTPUT = os.path.join(L.FIGS, "final")
MANUSCRIPT = os.path.join(OUTPUT, "pdf")
L.MODE = "surface"
L.FIGS_OUT = OUTPUT
LW, BLACK, GRAY, BLUE, RED = L.LW, L.BLACK, L.GRAY, L.BLUE, L.RED


def bar(fig, ax, sc, label, width=0.035, sep=0.02):
    """Color bar attached to the panel (same height as the axis box, after the layout)."""
    cax = ax.inset_axes([1.0 + sep, 0.0, width, 1.0])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6.5, width=LW, length=2.2)
    cb.outline.set_linewidth(LW)
    return cb


def window(mask, ext, margin=10.0):
    """Window (h0, h1), (v0, v1) in mm around what is true in `mask` (rows = vertical axis)."""
    nv, nh = mask.shape
    sh, sv = (ext[1] - ext[0]) / nh, (ext[2] - ext[3]) / nv
    vv, hh = np.nonzero(mask)
    return (
        (
            max(ext[0], hh.min() * sh - margin),
            min(ext[1], (hh.max() + 1) * sh + margin),
        ),
        (
            max(ext[3], vv.min() * sv - margin),
            min(ext[2], (vv.max() + 1) * sv + margin),
        ),
    )


def slice_2d(ax, M, ext, win, **kw):
    im = ax.imshow(M, extent=ext, aspect="equal", interpolation="nearest", **kw)
    ax.set_xlim(win[0])
    ax.set_ylim(win[1][1], win[1][0])
    L.no_axes(ax)
    return im


def fig_arrow(fig, a, b, text=None, color=BLACK, ls="-", invert=False):
    """Horizontal arrow between two neighboring axes (right border of `a` -> left border of `b`), label above."""
    pa, pb = a.get_position(), b.get_position()
    y = (min(pa.y1, pb.y1) + max(pa.y0, pb.y0)) / 2
    p0, p1 = (pa.x1 + 0.006, y), (pb.x0 - 0.006, y)
    if invert:
        p0, p1 = p1, p0
    fig.add_artist(
        FancyArrowPatch(
            p0,
            p1,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=7,
            color=color,
            lw=LW,
            linestyle=ls,
            shrinkA=0,
            shrinkB=0,
        )
    )
    if text:
        fig.text(
            (pa.x1 + pb.x0) / 2,
            y + 0.02,
            text,
            ha="center",
            va="bottom",
            fontsize=6,
            color=color,
            linespacing=1.15,
        )


# ----------------------------------------------------------------------------- Fig. pipeline (infographic)
def elbow_fig(fig, points, color=BLACK, ls="-"):
    """Orthogonal arrow in figure coordinates: segments through `points`, head only on the last one."""
    for p0, p1 in zip(points[:-2], points[1:-1], strict=True):
        fig.add_artist(
            Line2D(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                transform=fig.transFigure,
                color=color,
                lw=LW,
                ls=ls,
            )
        )
    fig.add_artist(
        FancyArrowPatch(
            points[-2],
            points[-1],
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=7,
            color=color,
            lw=LW,
            linestyle=ls,
            shrinkA=0,
            shrinkB=0,
        )
    )


def fig_pipeline():
    """Infographic of all the steps (Section 2) with real images of the representative foot, in three bands:
 reconstruction (2.3-2.5), field estimation (2.6-2.7) and evaluation/use (2.8-2.11)."""
    thin, ext_f = L.read_matrix("fig1_thin_ct_sagittal_HU.csv")
    thk, ext_e = L.read_matrix("fig1_thick_ct_sagittal_HU.csv")
    sr, _ = L.read_matrix("fig1_ct_sr_sagittal_HU.csv")
    msk, _ = L.read_matrix("fig1_mask_sr_sagittal.csv")
    sec = L.read_table("fig2e_sinterp_section.csv")
    vp = L.read_table("fig2e_sr_vertices_on_plane.csv")
    t = L.read_table("fig3_vertices_uncertainties_field.csv")
    d = L.read_table("fig1_vertices_descriptors.csv")
    t4 = L.read_table("fig4_vertices.csv")
    corr = L.read_table("fig4_correspondences_per_strategy.csv")
    ens = L.read_table("fig4_strategies_one_trial.csv")
    foot = L.foot_method()
    u, v, p = t["proj_u"], t["proj_v"], t["depth"]
    e_hat, rot = t["e_hat_mm"], t4["kmeans_region"].astype(int)
    win = window(msk > 0.5, ext_f)
    ct = {"cmap": "gray", "vmin": -500, "vmax": 1500}
    gray = np.full(len(u), 0.25)

    fig = plt.figure(figsize=(L.WIDTH_2COL, 4.9))
    gs = fig.add_gridspec(
        3,
        5,
        left=0.04,
        right=0.945,
        top=0.955,
        bottom=0.012,
        wspace=0.50,
        hspace=0.42,
        height_ratios=[1.15, 1.0, 0.8],
    )
    A = [fig.add_subplot(gs[0, j]) for j in range(5)]
    B = [fig.add_subplot(gs[1, j]) for j in range(5)]
    Cx = [fig.add_subplot(gs[2, j]) for j in range(5)]

    # --- band 1: reconstruction
    slice_2d(A[0], thin, ext_f, win, **ct)
    L.title(A[0], "a", "Thin-slice CT (0.5 mm)", fs=7)
    slice_2d(A[1], thk, ext_e, win, **ct)
    L.title(A[1], "b", "Thick-slice CT (3 mm)", fs=7)
    slice_2d(A[2], sr, ext_f, win, **ct)
    L.title(A[2], "c", "Super-resolved CT", fs=7)
    slice_2d(A[3], msk, ext_f, win, cmap="gray", vmin=0, vmax=1)
    L.title(A[3], "d", "Bone mask", fs=7)
    L.mesh(A[4], u, v, p, gray, 0, 1, cmap="Greys", foot=foot)
    L.title(A[4], "e", "Reconstructed surface", fs=7)

    # --- band 2: field estimation
    slice_2d(B[0], thk, ext_e, win, cmap="gray", vmin=-500, vmax=1500, alpha=0.55)
    for sgm in np.unique(sec["segment"]):
        k = sec["segment"] == sgm
        B[0].plot(sec["y_mm"][k], sec["z_mm"][k], color=BLUE, lw=0.6)
    B[0].scatter(
        vp["y_mm"], vp["z_mm"], s=0.6, color=RED, linewidths=0, rasterized=True
    )
    L.title(B[0], "f", "$\\mathcal{S}_{\\mathrm{interp}}$ and SR vertices", fs=7)
    L.mesh(B[1], u, v, p, np.clip(d["d_shape"], 0, 2), 0, 2, foot=foot)
    L.title(B[1], "g", "$d_{\\mathrm{shape}}(v)$", fs=7)
    L.mesh(B[2], u, v, p, d["nz_abs"], 0, 1, foot=foot)
    L.title(B[2], "h", "$|n_z|(v)$", fs=7)
    L.mesh(B[3], u, v, p, e_hat, 0, 2, foot=foot)
    L.title(B[3], "i", "Predicted $\\hat{e}(v)$", fs=7)
    sc = L.mesh(B[4], u, v, p, t["e_measured_mm"], 0, 2, foot=foot)
    L.title(B[4], "j", "Measured $e(v)$", fs=7)

    # --- band 3: evaluation and use
    L.mesh(Cx[0], u, v, p, (rot % 20).astype(float), 0, 19, cmap="tab20", foot=foot)
    L.title(Cx[0], "k", "Surface regions", fs=7)
    high = e_hat >= np.percentile(e_hat, 90)
    L.mesh(
        Cx[1], u, v, p, gray, 0, 1, cmap="Greys", foot=foot, masks=[(high, RED)]
    )
    L.title(Cx[1], "l", "Top decile of $\\hat{e}(v)$", fs=7)
    L.mesh(
        Cx[2], u, v, p, t["u_ens_HU"], 0, float(np.percentile(t["u_ens_HU"], 99)), foot=foot
    )
    L.title(Cx[2], "m", "$u_{\\mathrm{ens}}(v)$", fs=7)
    L.mesh(
        Cx[3],
        u,
        v,
        p,
        t["u_geo_std_mm"],
        0,
        float(np.percentile(t["u_geo_std_mm"], 99)),
        foot=foot,
    )
    L.title(Cx[3], "n", "$u_{\\mathrm{geo,std}}(v)$", fs=7)
    reg = int(ens["kmeans_region"][ens["strategy"] == "field"][0])
    target = int(ens["target_index"][ens["strategy"] == "field"][0])
    loc = corr["vertex_index"][corr["strategy"] == "field"].astype(int)
    L.mesh(Cx[4], u, v, p, e_hat, 0, 2, foot=foot, masks=[(rot == reg, L.GRAY_REGION)])
    Cx[4].scatter(u[loc], v[loc], s=1.6, color=RED, linewidths=0, zorder=4)
    Cx[4].scatter(
        [u[target]],
        [v[target]],
        s=24,
        marker="*",
        color="white",
        edgecolors=BLACK,
        linewidths=0.5,
        zorder=5,
    )
    L.title(Cx[4], "o", "Registration", fs=7)

    for ax in (*A, *B, *Cx):
        ax.set_anchor("C")
    bar(fig, B[4], sc, "mm, (g), (i), (j)", width=0.045, sep=0.03)
    fig.canvas.draw()

    fig_arrow(fig, A[0], A[1], "slice\naveraging")
    fig_arrow(fig, A[1], A[2], "super-\nresolution")
    fig_arrow(fig, A[2], A[3], "threshold,\ncleaning")
    fig_arrow(fig, A[3], A[4], "marching\ncubes")
    fig_arrow(fig, B[0], B[1], "distance")
    fig_arrow(fig, B[2], B[3], "random\nforest")
    fig_arrow(
        fig, B[3], B[4], "training\nonly", color=BLUE, ls=(0, (4, 2.5)), invert=True
    )
    pB1, pB2 = B[1].get_position(), B[2].get_position()
    fig.text(
        (pB1.x1 + pB2.x0) / 2,
        (pB1.y0 + pB1.y1) / 2,
        "+",
        ha="center",
        va="center",
        fontsize=9,
    )

    def center(ax):
        q = ax.get_position()
        return (q.x0 + q.x1) / 2

    base_A = min(a.get_position().y0 for a in A) - 0.008

    def top(ax):
        return ax.get_position().y1 + 0.05

    y1 = base_A - 0.030
    y2 = base_A - 0.052
    # (b) -> (f): thick mask, signed distance field, interpolation
    elbow_fig(
        fig,
        [
            (center(A[1]), base_A),
            (center(A[1]), y1),
            (center(B[0]), y1),
            (center(B[0]), top(B[0])),
        ],
    )
    fig.text(
        (center(B[0]) + center(A[1])) / 2,
        y1 + 0.004,
        "signed distance field",
        fontsize=6,
        ha="center",
        va="bottom",
    )
    # (e) -> (h): per-vertex geometric descriptors
    elbow_fig(
        fig,
        [
            (center(A[4]), base_A),
            (center(A[4]), y2),
            (center(B[2]), y2),
            (center(B[2]), top(B[2])),
        ],
    )
    fig.text(
        (center(B[2]) + center(A[4])) / 2,
        y2 + 0.004,
        "vertex-wise descriptors",
        fontsize=6,
        ha="center",
        va="bottom",
    )

    for axes, label in (
        (A, "Reconstruction"),
        (B, "Estimation"),
        (Cx, "Evaluation and use"),
    ):
        y = (
            min(a.get_position().y0 for a in axes)
            + max(a.get_position().y1 for a in axes)
        ) / 2
        fig.text(
            0.012,
            y,
            label,
            fontsize=7,
            weight="bold",
            va="center",
            ha="center",
            rotation=90,
            color=GRAY,
        )
    L.save(fig, "fig_pipeline")
    return {
        "foot": foot,
        "window_mm": [list(map(float, win[0])), list(map(float, win[1]))],
        "bins": 3,
        "panels": 15,
    }


# ----------------------------------------------------------------------------- Fig. shape disagreement
def fig_shape():
    ct, ext = L.read_matrix("fig2a_thick_ct_sagittal_HU.csv")
    m, _ = L.read_matrix("fig2b_thick_mask.csv")
    phi, _ = L.read_matrix("fig2c_thick_sdf_mm.csv")
    phif, extf = L.read_matrix("fig2d_interpolated_sdf_mm.csv")
    sec = L.read_table("fig2e_sinterp_section.csv")
    vp = L.read_table("fig2e_sr_vertices_on_plane.csv")
    vf = L.read_table("fig2f_vertices_dshape.csv")
    m = m > 0.5
    sy, thick_sz = (ext[1] - ext[0]) / ct.shape[1], (ext[2] - ext[3]) / ct.shape[0]
    zz, yy = np.nonzero(m)
    y_all = np.r_[yy * sy, sec["y_mm"], vp["y_mm"]]
    z_all = np.r_[zz * thick_sz, sec["z_mm"], vp["z_mm"]]
    win = (
        (max(ext[0], y_all.min() - 10), min(ext[1], y_all.max() + 10)),
        (max(ext[3], z_all.min() - 10), min(ext[2], z_all.max() + 10)),
    )
    reg = m | (phi < 8)
    lim = float(np.percentile(np.abs(phi[reg]), 99)) if reg.any() else 8.0

    fig = plt.figure(figsize=(L.WIDTH_2COL, 4.35))
    gs = fig.add_gridspec(
        2, 3, left=0.005, right=0.925, top=0.955, bottom=0.075, wspace=0.30, hspace=0.20
    )
    ax = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    slice_2d(ax[0], ct, ext, win, cmap="gray", vmin=-500, vmax=1500)
    L.title(ax[0], "a", "Thick-slice CT, sagittal section")
    slice_2d(ax[1], m, ext, win, cmap="gray", vmin=0, vmax=1)
    L.title(ax[1], "b", "Bone mask on the thick-slice grid")
    im = slice_2d(ax[2], phi, ext, win, cmap="RdBu_r", vmin=-lim, vmax=lim)
    L.title(ax[2], "c", "Signed distance field $\\phi$")
    im2 = slice_2d(ax[3], phif, extf, win, cmap="RdBu_r", vmin=-lim, vmax=lim)
    L.title(ax[3], "d", "Interpolated field $\\tilde{\\phi}$")
    slice_2d(ax[4], ct, ext, win, cmap="gray", vmin=-500, vmax=1500, alpha=0.55)
    for s in np.unique(sec["segment"]):
        k = sec["segment"] == s
        ax[4].plot(sec["y_mm"][k], sec["z_mm"][k], color=BLUE, lw=0.7)
    ax[4].scatter(
        vp["y_mm"], vp["z_mm"], s=1.0, color=RED, linewidths=0, rasterized=True
    )
    L.title(ax[4], "e", "$\\mathcal{S}_{\\mathrm{interp}}$ and super-resolved surface")
    x0, y0 = win[0][0] + 4, win[1][1] - 5
    ax[0].add_line(
        Line2D([x0, x0 + 20], [y0, y0], color="white", lw=1.2, solid_capstyle="butt")
    )
    ax[0].text(
        x0 + 10, y0 - 2, "20 mm", color="white", fontsize=6.5, ha="center", va="bottom"
    )
    ax[4].legend(
        handles=[
            Line2D(
                [],
                [],
                color=BLUE,
                lw=0.9,
                label="$\\mathcal{S}_{\\mathrm{interp}} = \\{\\tilde{\\phi} = 0\\}$",
            ),
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                ms=2.5,
                color=RED,
                label="SR vertices within 0.5 mm of the plane",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=False,
        fontsize=6.5,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.2,
    )
    sc = L.mesh(
        ax[5],
        vf["proj_u"],
        vf["proj_v"],
        vf["depth"],
        np.clip(vf["d_shape_mm"], 0, 2),
        0,
        2,
        foot=L.foot_method(),
    )
    L.title(ax[5], "f", "$d_{\\mathrm{shape}}(v)$ on the super-resolved surface")
    for a in ax:
        a.set_anchor("N")
    bar(fig, ax[2], im, "$\\phi$ (mm)")
    bar(fig, ax[3], im2, "$\\tilde{\\phi}$ (mm)")
    bar(fig, ax[5], sc, "$d_{\\mathrm{shape}}$ (mm)")
    L.save(fig, "fig_shape_disagreement")
    return {
        "limit_sdf_mm": lim,
        "d_shape_saturated_at_mm": 2.0,
        "d_shape_median_mm": float(np.median(vf["d_shape_mm"])),
    }


# ----------------------------------------------------------------------------- Fig. uncertainty
def fig_uncertainty():
    t = L.read_table("fig3_vertices_uncertainties_field.csv")
    u, v, p = t["proj_u"], t["proj_v"], t["depth"]
    foot = L.foot_method()
    vmax_hu = float(np.percentile(t["u_ens_HU"], 99))
    vmax_geo = float(np.percentile(t["u_geo_std_mm"], 99))
    fig = plt.figure(figsize=(L.WIDTH_2COL, 3.9))
    gs = fig.add_gridspec(
        2, 2, left=0.005, right=0.93, top=0.955, bottom=0.01, wspace=0.22, hspace=0.12
    )
    ax = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    panels = [
        ("a", "Measured surface error $e(v)$", t["e_measured_mm"], 2.0, "mm"),
        (
            "b",
            "Intensity-space uncertainty $u_{\\mathrm{ens}}(v)$",
            t["u_ens_HU"],
            vmax_hu,
            "HU",
        ),
        (
            "c",
            "Surface-space uncertainty $u_{\\mathrm{geo,std}}(v)$",
            t["u_geo_std_mm"],
            vmax_geo,
            "mm",
        ),
        ("d", "Predicted field $\\hat{e}(v)$", t["e_hat_mm"], 2.0, "mm"),
    ]
    for a, (letter, tit, val, vmax, un) in zip(ax, panels, strict=True):
        sc = L.mesh(a, u, v, p, val, 0, vmax, foot=foot)
        L.title(a, letter, tit)
        a.set_anchor("N")
        bar(fig, a, sc, un, width=0.03)
    L.save(fig, "fig_uncertainty")
    return {
        "foot": foot,
        "vmax_e_mm": 2.0,
        "vmax_u_ens_HU_p99": vmax_hu,
        "vmax_u_geo_std_mm_p99": vmax_geo,
    }


# ----------------------------------------------------------------------------- Fig. calibration
def fig_calibration():
    d = L.read_table("foot_calibration_deciles.csv", L.DATA_RES)
    px, px1, px3 = d["predicted_median_mm"], d["predicted_q1"], d["predicted_q3"]
    oy, oy1, oy3 = d["observed_median_mm"], d["observed_q1"], d["observed_q3"]
    n = d["n_feet"].astype(int)
    fig, ax = plt.subplots(figsize=(L.WIDTH_1COL, 3.3), constrained_layout=True)
    lim = (0.0, float(max(px3.max(), oy3.max())) * 1.05)  # from zero: the lowest deciles must stay in view
    ax.plot(lim, lim, color=GRAY, lw=LW, ls=(0, (4, 2.5)), label="Identity")
    ax.errorbar(
        px,
        oy,
        xerr=[px - px1, px3 - px],
        yerr=[oy - oy1, oy3 - oy],
        fmt="o",
        ms=3,
        color=BLACK,
        ecolor=BLACK,
        elinewidth=LW,
        capsize=1.5,
        capthick=LW,
        label="Decile of $\\hat{e}(v)$",
    )
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted error, mean within decile (mm)")
    ax.set_ylabel("Measured error, mean within decile (mm)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    L.save(fig, "fig_calibration")
    return {"deciles": len(n), "n_feet_deciles_1_3": n[:3].tolist()}


# ----------------------------------------------------------------------------- Fig. registration
def fig_registration():
    t = L.read_table("fig4_vertices.csv")
    corr = L.read_table("fig4_correspondences_per_strategy.csv")
    ens = L.read_table("fig4_strategies_one_trial.csv")
    c300 = L.read_table("fig4_contrast_300_trials.csv")
    u, v, p = t["proj_u"], t["proj_v"], t["depth"]
    e_hat, rot = t["e_hat_mm"], t["kmeans_region"].astype(int)
    vmax = float(np.percentile(e_hat, 99))
    locs = {
        s: corr["vertex_index"][corr["strategy"] == s].astype(int)
        for s in np.unique(corr["strategy"])
    }
    reg_of = {
        s: int(r) for s, r in zip(ens["strategy"], ens["kmeans_region"], strict=True)
    }
    target = {
        s: int(a) for s, a in zip(ens["strategy"], ens["target_index"], strict=True)
    }
    d_target = {
        s: float(x) for s, x in zip(ens["strategy"], ens["d_target_mm"], strict=True)
    }
    order = [
        ("field", "Lowest predicted error $\\hat{e}$"),
        ("d_shape", "Lowest shape disagreement $d_{\\mathrm{shape}}$"),
        ("random", "Random region"),
        ("oracle", "Oracle (lowest measured $e$)"),
        ("global", "Whole surface"),
    ]
    fig = plt.figure(figsize=(L.WIDTH_2COL, 4.0))
    gs = fig.add_gridspec(
        2, 3, hspace=0.08, wspace=0.03, left=0.005, right=0.995, top=0.94, bottom=0.085
    )
    ax = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    foot = L.foot_method()
    sc = None
    for a, letter, (s, tit) in zip(ax[:5], "abcde", order, strict=True):
        mask = [(rot == reg_of[s], L.GRAY_REGION)] if reg_of[s] >= 0 else []
        sc = L.mesh(a, u, v, p, e_hat, 0, vmax, foot=foot, masks=mask)
        a.scatter(u[locs[s]], v[locs[s]], s=3.0, color=RED, linewidths=0, zorder=4)
        a.scatter(
            [u[target[s]]],
            [v[target[s]]],
            s=30,
            marker="*",
            color="white",
            edgecolors=BLACK,
            linewidths=0.5,
            zorder=5,
        )
        L.title(
            a,
            letter,
            f"{tit}\n$d_{{\\mathrm{{target}}}}$ = {d_target[s]:.2f} mm (one trial)",
            fs=7,
        )
        a.set_anchor("N")
    # (f) distribution of the target displacement in 300 trials: region of smaller x of larger ê
    a = ax[5]
    box = a.get_position()
    a.set_position(
        [box.x0 + 0.085, box.y0 + 0.055, box.width - 0.21, box.height - 0.10]
    )
    rng = np.random.default_rng(0)  # jitter only for readability
    labels = []
    for i, (key, label) in enumerate(
        (
            ("lowest_e_hat", "Lowest\n$\\hat{e}$ region"),
            ("highest_e_hat", "Highest\n$\\hat{e}$ region"),
        )
    ):
        y = c300["d_target_mm"][c300["region"] == key]
        a.scatter(
            i + rng.uniform(-0.16, 0.16, len(y)),
            y,
            s=3,
            color=GRAY,
            alpha=0.5,
            linewidths=0,
            rasterized=True,
        )
        q1, q2, q3 = np.percentile(y, [25, 50, 75])
        a.hlines([q1, q3], i - 0.27, i + 0.27, color=BLACK, lw=LW)
        a.hlines(q2, i - 0.33, i + 0.33, color=BLACK, lw=1.4)
        labels.append(label)
    a.set_xticks([0, 1])
    a.set_xticklabels(labels, fontsize=6.5)
    a.set_xlim(-0.6, 1.6)
    a.set_ylim(0, None)
    a.set_ylabel("$d_{\\mathrm{target}}$ (mm), 300 trials")
    a.spines[["top", "right"]].set_visible(False)
    L.title(a, "f", "Region with lowest vs. highest $\\hat{e}$", fs=7)
    cax = fig.add_axes(
        [box.x0 + box.width - 0.075, box.y0 + 0.055, 0.012, box.height - 0.10]
    )
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("$\\hat{e}(v)$ in (a)–(e) (mm)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, width=LW, length=2.2)
    cb.outline.set_linewidth(LW)
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                ms=3,
                color=L.GRAY_REGION,
                label="selected region",
            ),
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                ms=2.5,
                color=RED,
                label="$K$ = 200 sampled correspondences",
            ),
            Line2D(
                [],
                [],
                ls="",
                marker="*",
                ms=6,
                color="white",
                markeredgecolor=BLACK,
                label="remote target",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.36, 0.0),
        ncol=3,
        frameon=False,
        fontsize=6.5,
        handletextpad=0.3,
        columnspacing=1.4,
    )
    L.save(fig, "fig_registration")
    med = {
        k: float(np.median(c300["d_target_mm"][c300["region"] == k]))
        for k in ("lowest_e_hat", "highest_e_hat")
    }
    return {
        "vmax_mm_p99": vmax,
        "d_target_one_trial_mm": d_target,
        "median_contrast_mm": med,
    }


# ----------------------------------------------------------------------------- Fig. thorax
def fig_thorax():
    t = L.read_table("fig7_vertices.csv")
    rc = L.read_table("fig7_rho_per_case.csv")
    thk, ext_e = L.read_matrix("fig7_thick_ct_coronal_HU.csv")
    sr, ext_f = L.read_matrix("fig7_ct_sr_coronal_HU.csv")
    ref, _ = L.read_matrix("fig7_reference_ct_coronal_HU.csv")
    with open(
        os.path.join(L.FIGS, "figure_data_provenance.json"), encoding="utf-8"
    ) as f:
        case = json.load(f)["thorax"]["case"]
    # mesh with faces (re-extracted by a4_export_figure_data.py); the thorax cache does not store normals
    m = np.load(os.path.join(RENDER.MESHES, case + ".npz"), allow_pickle=False)
    Vc = np.c_[t["x_mm"], t["y_mm"], t["z_mm"]]
    RENDER._CACHE[case] = {
        "V": Vc,
        "N": np.zeros_like(Vc),
        "V_regen": m["V"].astype(float),
        "faces": m["faces"].astype(int),
        "idx_cache": m["idx_cache"].astype(int),
        "verification": json.loads(str(m["verification"])),
    }
    u, v, p = t["proj_u"], t["proj_v"], t["depth"]
    # view rotated 90 degrees in the figure plane so that the spine is vertical, head up (z grows toward the feet)
    u, v = (-v, u) if np.corrcoef(u, t["z_mm"])[0, 1] < 0 else (v, -u)
    rho_case = float(rc["rho_sr"][rc["case"] == case][0])
    ct = {"cmap": "gray", "vmin": -500, "vmax": 1500}
    win = ((ext_f[0], ext_f[1]), (ext_f[3], ext_f[2]))

    fig = plt.figure(figsize=(L.WIDTH_2COL, 4.6))
    gs = fig.add_gridspec(
        2,
        3,
        left=0.005,
        right=0.985,
        top=0.955,
        bottom=0.075,
        wspace=0.16,
        hspace=0.16,
        height_ratios=[0.8, 1.0],
    )
    a = [fig.add_subplot(gs[0, j]) for j in range(3)]
    slice_2d(a[0], thk, ext_e, win, **ct)
    L.title(a[0], "a", "Thick-slice CT (5 mm), coronal section")
    slice_2d(a[1], sr, ext_f, win, **ct)
    L.title(a[1], "b", "Super-resolved CT")
    slice_2d(a[2], ref, ext_f, win, **ct)
    L.title(a[2], "c", "Thin-slice CT (1 mm, reference)")
    b = [fig.add_subplot(gs[1, j]) for j in range(3)]
    L.mesh(b[0], u, v, p, t["e_measured_mm"], 0, 3.0, foot=case)
    L.title(b[0], "d", "Measured surface error $e(v)$")
    sc = L.mesh(b[1], u, v, p, t["e_hat_mm"], 0, 3.0, foot=case)
    L.title(
        b[1],
        "e",
        f"Predicted field $\\hat{{e}}(v)$ ($\\rho$ = {rho_case:.3f})",
    )
    for ax in (*a, *b[:2]):
        ax.set_anchor("N")
    bar(fig, b[1], sc, "mm", width=0.03)
    ax = b[2]
    box = ax.get_position()
    ax.set_position(
        [box.x0 + 0.115, box.y0 + 0.03, box.width - 0.135, box.height - 0.06]
    )
    rng = np.random.default_rng(0)
    for i, vals in enumerate((rc["rho_sr"], rc["rho_tri"])):
        ax.scatter(
            i + rng.uniform(-0.14, 0.14, len(vals)),
            vals,
            s=5,
            color=GRAY,
            alpha=0.6,
            linewidths=0,
        )
        q1, q2, q3 = np.percentile(vals, [25, 50, 75])
        ax.hlines([q1, q3], i - 0.26, i + 0.26, color=BLACK, lw=LW)
        ax.hlines(q2, i - 0.32, i + 0.32, color=BLACK, lw=1.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Super-\nresolution", "Trilinear"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, None)
    ax.set_ylabel("Vertex-level Spearman $\\rho$ per case")
    ax.spines[["top", "right"]].set_visible(False)
    L.title(ax, "f", "Test cases ($n$ = 100)")
    L.save(fig, "fig_thorax")
    return {
        "case": case,
        "rho_case": rho_case,
        "vmax_mm": 3.0,
        "verification": RENDER.verification(case),
    }


FIGURES = {
    "pipeline": fig_pipeline,
    "field": None,
    "calibration": fig_calibration,
    "uncertainty": fig_uncertainty,
    "shape": fig_shape,
    "thorax": fig_thorax,
    "registration": fig_registration,
}


def fig_field():
    """Fig. field x error: the clean version unchanged, only with the final name."""
    out = L.fig5()
    for ext in ("pdf", "png"):
        os.replace(
            os.path.join(OUTPUT, f"fig5_field_vs_error.{ext}"),
            os.path.join(OUTPUT, f"fig_field_vs_error.{ext}"),
        )
    return out


FIGURES["field"] = fig_field


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(FIGURES))
    args = ap.parse_args()
    os.makedirs(OUTPUT, exist_ok=True)
    prov = {
        "script": os.path.relpath(__file__, L.ROOT),
        "data": dt.datetime.now(dt.UTC).astimezone().isoformat(timespec="seconds"),
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "mesh_mode": L.MODE,
        "figures": {},
    }
    for k in args.only.split(","):
        prov["figures"][k] = FIGURES[k]()
    prov["csv_sources_sha256"] = dict(sorted(L.SOURCES_USED.items()))
    p = os.path.join(OUTPUT, "final_figures_provenance.json")
    if os.path.exists(p) and set(args.only.split(",")) != set(FIGURES):
        with open(p, encoding="utf-8") as f:
            prev = json.load(f)
        prev["figures"].update(prov["figures"])
        prev["csv_sources_sha256"].update(prov["csv_sources_sha256"])
        prev["data"] = prov["data"]
        prov = prev
    with open(p, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=1, ensure_ascii=False)
    os.makedirs(MANUSCRIPT, exist_ok=True)
    for fname in sorted(os.listdir(OUTPUT)):
        if fname.endswith(".pdf"):
            shutil.copy2(os.path.join(OUTPUT, fname), os.path.join(MANUSCRIPT, fname))
    print(json.dumps({k: "ok" for k in args.only.split(",")}))


if __name__ == "__main__":
    main()
