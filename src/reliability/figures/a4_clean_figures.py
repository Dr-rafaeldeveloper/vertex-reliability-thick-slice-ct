"""Clean version (journal style) of the six figures of the paper, drawn ONLY from the CSVs already exported by the
generator scripts (`figs/data/*.csv` and `results_figure_data/*.csv`). Recomputes nothing: every number that
appears in the figure comes from a CSV, whose SHA-256 is written to `figs/clean/clean_figures_provenance.json`.

Style: white background, rectangular geometry, single line weight (0.6 pt), Arial 8 pt (ticks 7 pt), no overall title
(the LaTeX caption describes the figure), panel labels (a), (b),... in bold in the panel title, color only where it
encodes a quantity (maps) or distinguishes overlapping elements. Widths: 190 mm (two columns) and 90 mm (one column).
Output vector PDF (meshes rasterized at 600 dpi) and 600 dpi PNG in `output/figures/figs/clean/`.

Usage: python src/reliability/figures/a4_clean_figures.py [--only 1,2,3,4,5,6]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from itertools import pairwise

import matplotlib

matplotlib.use("Agg")
import a4_mesh_render as RENDER
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # repository root
OUTPUT_DIR = os.path.join(ROOT, "output", "figures")
FIGS = os.path.join(OUTPUT_DIR, "figs")
FIGS_OUT = os.path.join(FIGS, "clean")
DATA = os.path.join(FIGS, "data")
DATA_RES = os.path.join(OUTPUT_DIR, "results_figure_data")

DPI = 600
WIDTH_2COL = 190 / 25.4  # in
WIDTH_1COL = 90 / 25.4
CMAP = "viridis"
GRAY = "#4d4d4d"
BLACK = "#000000"
BLUE = "#1f4e79"  # elements linked to the reference
RED = "#c0392b"  # highlighted element (matches, SR vertices)
LIGHT_BLUE = "#e8eef5"
LIGHT_GRAY = "#f0f0f0"
GRAY_REGION = "#c8c8c8"
LW = 0.6

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 7.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": LW,
        "xtick.major.width": LW,
        "ytick.major.width": LW,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": DPI,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.cal": "Arial:italic",
    }
)

SOURCES_USED: dict[str, str] = {}


# ----------------------------------------------------------------------------- utilities
def _sha(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_table(name, folder=DATA):
    """CSV with header -> dict column -> ndarray (float when possible, otherwise str)."""
    filepath = os.path.join(folder, name)
    SOURCES_USED[os.path.relpath(filepath, OUTPUT_DIR)] = _sha(filepath)
    with open(filepath, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    out = {}
    for j, c in enumerate(header):
        col = [r[j] for r in body]
        try:
            out[c] = np.array([float(v) for v in col])
        except ValueError:
            out[c] = np.array(col)
    return out


def read_matrix(name):
    """2D image CSV: 1st line '# extent_mm (y0, y1, z_bottom, z_top) = [...]', then rows = z."""
    filepath = os.path.join(DATA, name)
    SOURCES_USED[os.path.relpath(filepath, OUTPUT_DIR)] = _sha(filepath)
    with open(filepath, encoding="utf-8") as f:
        header = f.readline()
    ext = json.loads(header.split("=", 1)[1].strip())
    M = np.loadtxt(filepath, delimiter=",", skiprows=1)
    return M, ext


def save(fig, name):
    os.makedirs(FIGS_OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(FIGS_OUT, f"{name}.{ext}"),
            dpi=DPI,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(fig)
    print("figure saved:", name, flush=True)


def title(ax, letter, text, fs=7.5):
    """Panel title aligned to the left, letter in bold."""
    ax.set_title(f"$\\bf{{({letter})}}$ {text}", loc="left", fontsize=fs, pad=3)


def no_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def box(ax, xy, w, h, text, fc="white", ec=BLACK, lw=LW, fs=6.5):
    ax.add_patch(Rectangle(xy, w, h, fc=fc, ec=ec, lw=lw, joinstyle="miter"))
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        linespacing=1.25,
    )


def arrow(ax, p0, p1, color=BLACK, ls="-", lw=LW):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=7,
            color=color,
            lw=lw,
            linestyle=ls,
            shrinkA=0,
            shrinkB=0,
            joinstyle="miter",
            capstyle="butt",
        )
    )


def elbow(ax, points, color=BLACK, ls="-", lw=LW):
    """Orthogonal arrow: straight segments through `points`, head only on the last one."""
    for a, b in pairwise(points[:-1]):
        ax.add_line(
            Line2D(
                [a[0], b[0]],
                [a[1], b[1]],
                color=color,
                lw=lw,
                ls=ls,
                solid_capstyle="butt",
            )
        )
    arrow(ax, points[-2], points[-1], color=color, ls=ls, lw=lw)


MODE = "points"  # "points" | "shaded" (route A) | "surface" (route B); see a4_mesh_render


def mesh(ax, u, v, depth, val, vmin, vmax, s=0.5, cmap=CMAP, foot=None, masks=()):
    """Draws the mesh colored by `val` in the global mode MODE. `masks` = [(bool per vertex, color),...] highlights
 regions (faces in surface mode; larger points in the other modes)."""
    if MODE == "points":
        o = np.argsort(depth)  # paints first what is behind
        sc = ax.scatter(
            u[o],
            v[o],
            c=val[o],
            s=s,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
            linewidths=0,
        )
        ax.set_aspect("equal")
    elif MODE == "shaded":
        sc = RENDER.shaded_points(
            ax, foot, u, v, depth, val, vmin, vmax, cmap=cmap, s=max(2.5, 6 * s)
        )
    elif MODE == "surface":
        sc = RENDER.surface(
            ax, foot, u, v, depth, val, vmin, vmax, cmap=cmap, masks=masks
        )
    else:
        raise ValueError(MODE)
    if MODE != "surface":
        for mask, color in masks:
            k = np.asarray(mask, bool)
            ax.scatter(
                u[k], v[k], s=max(1.2, 3 * s), color=color, linewidths=0, rasterized=True
            )
    no_axes(ax)
    return sc


def foot_method():
    """Representative foot of Figs. 2–4 (the one that exported figs/data/fig2*–fig4*.csv)."""
    with open(os.path.join(FIGS, "figures_provenance.json"), encoding="utf-8") as f:
        return json.load(f)["foot"]


def foot_results():
    """Representative foot of Fig. 5 (the one that exported figs/data/fig5ab_vertices.csv)."""
    with open(
        os.path.join(FIGS, "results_figures_provenance.json"), encoding="utf-8"
    ) as f:
        return json.load(f)["foot"]


def colorbar_(fig, sc, axs, label, **kw):
    cb = fig.colorbar(sc, ax=axs, **kw)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=7, width=LW, length=2.5)
    cb.outline.set_linewidth(LW)
    return cb


def largest_component(X, radius_mm=3.0):
    """Mask of the largest connected component of the neighborhood graph (pairs closer than radius_mm); same criterion as
 a4_results_figures.largest_component (the cache stores vertices, not faces)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pairs = cKDTree(X).query_pairs(r=radius_mm, output_type="ndarray")
    g = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(X), len(X))
    )
    _n, rot = connected_components(g, directed=False)
    return rot == np.bincount(rot).argmax()


# ----------------------------------------------------------------------------- Fig. 1 (scheme)
def fig1():
    fig, ax = plt.subplots(figsize=(WIDTH_2COL, 3.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
    w, h = 17.5, 9.0
    xs = [0.5, 20.9, 41.3, 61.7, 82.1]
    xc = [x + w / 2 for x in xs]
    ya, yb, yc = 42.0, 27.0, 7.0  # top of the upper band, 2nd row, lower band
    y_lane = 23.0

    ax.text(
        0.5,
        53.0,
        "Training and evaluation (uses the higher-resolution reference)",
        fontsize=8,
        weight="bold",
    )
    ax.text(
        0.5,
        19.2,
        "Target-case inference (no reference available)",
        fontsize=8,
        weight="bold",
    )
    ax.add_line(
        Line2D([0, 100], [y_lane, y_lane], color=GRAY, lw=LW, ls=(0, (4, 2.5)))
    )

    row_a = [
        "Higher-resolution CT\n(0.5 mm; 48 feet)",
        "Controlled thick-slice\ngeneration\n(k = 6; Eqs. 1–2; §2.3)",
        "Through-plane\nsuper-resolution\n(§2.4)",
        "Segmentation and\nsurface extraction\n(§2.5)",
        "Vertex-wise features\nx(v), nine descriptors\n(§2.7.1)",
    ]
    for x, t in zip(xs, row_a, strict=True):
        box(ax, (x, ya), w, h, t)
    for a, b in pairwise(xs):
        arrow(ax, (a + w, ya + h / 2), (b, ya + h / 2))

    box(
        ax,
        (xs[1], yb),
        w,
        h,
        "Reference surface\n(same segmentation\nand meshing)",
        fc=LIGHT_BLUE,
        ec=BLUE,
    )
    box(
        ax,
        (xs[2], yb),
        w,
        h,
        "Vertex-wise geometric\nerror $e(v)$\n(Eq. 5; §2.6)",
        fc=LIGHT_BLUE,
        ec=BLUE,
    )
    box(
        ax,
        (xs[3], yb),
        w,
        h,
        "Random-forest\nregression (§2.7.3);\nheld-out prediction\n(§2.7.4)",
    )
    box(
        ax,
        (xs[4], yb),
        w,
        h,
        "Evaluation (§2.9–2.11):\nbaselines, real paired\nCT, rigid registration",
    )
    # reference enters only in the upper band
    elbow(ax, [(xc[0], ya), (xc[0], 39.0), (xc[1], 39.0), (xc[1], yb + h)], color=BLUE)
    arrow(ax, (xs[1] + w, yb + h / 2), (xs[2], yb + h / 2), color=BLUE)
    arrow(ax, (xs[2] + w, yb + h / 2), (xs[3], yb + h / 2), color=BLUE)
    elbow(ax, [(xc[4], ya), (xc[4], 39.0), (xc[3], 39.0), (xc[3], yb + h)])
    arrow(ax, (xs[3] + w, yb + h / 2), (xs[4], yb + h / 2))

    row_c = [
        "Thick-slice CT\n(target case)",
        "Through-plane\nsuper-resolution",
        "Segmentation and\nsurface extraction",
        "Vertex-wise features\nx(v), nine descriptors",
        "Predicted reliability\nfield $\\hat{e}(v)$ in mm\n(Eq. 19)",
    ]
    for i, (x, t) in enumerate(zip(xs, row_c, strict=True)):
        box(
            ax,
            (x, yc),
            w,
            h,
            t,
            fc=LIGHT_GRAY if i == 4 else "white",
            lw=1.0 if i == 4 else LW,
        )
    for a, b in pairwise(xs):
        arrow(ax, (a + w, yc + h / 2), (b, yc + h / 2))
    # trained model crosses the line of the bands
    elbow(
        ax, [(xc[3], yb), (xc[3], 19.5), (xc[4], 19.5), (xc[4], yc + h)], ls=(0, (3, 2))
    )
    ax.text(
        (xc[3] + xc[4]) / 2,
        20.2,
        "trained model",
        fontsize=6.5,
        ha="center",
        va="bottom",
    )
    ax.text(
        0.5,
        1.5,
        "Reference data enter only the upper lane; the target case is processed without any reference.",
        fontsize=7,
        style="italic",
    )
    save(fig, "fig1_framework")
    return {"kind": "schematic (no data)"}


# ----------------------------------------------------------------------------- Fig. 2
def fig2():
    ct, ext = read_matrix("fig2a_thick_ct_sagittal_HU.csv")
    m, _ = read_matrix("fig2b_thick_mask.csv")
    phi, _ = read_matrix("fig2c_thick_sdf_mm.csv")
    phif, extf = read_matrix("fig2d_interpolated_sdf_mm.csv")
    sec = read_table("fig2e_sinterp_section.csv")
    vp = read_table("fig2e_sr_vertices_on_plane.csv")
    vf = read_table("fig2f_vertices_dshape.csv")
    m = m > 0.5
    sy = (ext[1] - ext[0]) / ct.shape[1]
    thick_sz = (ext[2] - ext[3]) / ct.shape[0]
    # window: union (mask, S_interp section, SR vertices in the plane) +- 10 mm, limited to the extent
    zz, yy = np.nonzero(m)
    y_all = np.r_[yy * sy, sec["y_mm"], vp["y_mm"]]
    z_all = np.r_[zz * thick_sz, sec["z_mm"], vp["z_mm"]]
    ymm = (max(ext[0], y_all.min() - 10), min(ext[1], y_all.max() + 10))
    zmm = (max(ext[3], z_all.min() - 10), min(ext[2], z_all.max() + 10))
    reg = m | (phi < 8)
    lim = float(np.percentile(np.abs(phi[reg]), 99)) if reg.any() else 8.0

    fig, axs = plt.subplots(2, 3, figsize=(WIDTH_2COL, 4.6), constrained_layout=True)
    ax = axs.ravel()
    kw = {"extent": ext, "aspect": "equal", "interpolation": "nearest"}
    ax[0].imshow(ct, cmap="gray", vmin=-500, vmax=1500, **kw)
    title(ax[0], "a", "Thick-slice CT, sagittal section")
    ax[1].imshow(m, cmap="gray", vmin=0, vmax=1, **kw)
    title(ax[1], "b", "Bone mask on the thick grid (§2.5)")
    im = ax[2].imshow(phi, cmap="RdBu_r", vmin=-lim, vmax=lim, **kw)
    title(ax[2], "c", "Signed distance field $\\phi$ (Eq. 15)")
    im2 = ax[3].imshow(
        phif,
        cmap="RdBu_r",
        vmin=-lim,
        vmax=lim,
        extent=extf,
        aspect="equal",
        interpolation="nearest",
    )
    title(ax[3], "d", "Interpolated field $\\tilde{\\phi}$ (Eq. 16)")
    ax[4].imshow(ct, cmap="gray", vmin=-500, vmax=1500, alpha=0.55, **kw)
    for s in np.unique(sec["segment"]):
        k = sec["segment"] == s
        ax[4].plot(sec["y_mm"][k], sec["z_mm"][k], color=BLUE, lw=0.7)
    ax[4].scatter(
        vp["y_mm"], vp["z_mm"], s=1.0, color=RED, linewidths=0, rasterized=True
    )
    title(ax[4], "e", "$\\mathcal{S}_{\\mathrm{interp}}$ (Eq. 17) and SR surface")
    for a in ax[:5]:
        a.set_xlim(ymm)
        a.set_ylim(zmm[1], zmm[0])
        no_axes(a)
    # scale bar (20 mm) in panel (a)
    x0, y0 = ymm[0] + 4, zmm[1] - 5
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
                label="SR vertices, |Δx| ≤ 0.5 mm",
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
    sc = mesh(
        ax[5],
        vf["proj_u"],
        vf["proj_v"],
        vf["depth"],
        np.clip(vf["d_shape_mm"], 0, 2),
        0,
        2,
        s=0.35,
        foot=foot_method(),
    )
    title(ax[5], "f", "$d_{\\mathrm{shape}}(v)$ on the SR surface (Eq. 18)")
    colorbar_(fig, im, [ax[2]], "$\\phi$ (mm)", fraction=0.05, pad=0.02, shrink=0.85)
    colorbar_(
        fig, im2, [ax[3]], "$\\tilde{\\phi}$ (mm)", fraction=0.05, pad=0.02, shrink=0.85
    )
    colorbar_(
        fig,
        sc,
        [ax[5]],
        "$d_{\\mathrm{shape}}$ (mm; clipped at 2)",
        fraction=0.05,
        pad=0.02,
        shrink=0.85,
    )
    save(fig, "fig2_shape_disagreement")
    return {
        "window_y_mm": [float(ymm[0]), float(ymm[1])],
        "window_z_mm": [float(zmm[0]), float(zmm[1])],
        "limit_sdf_mm": lim,
        "vertices_on_plane": len(vp["y_mm"]),
        "d_shape_median_mm": float(np.median(vf["d_shape_mm"])),
    }


# ----------------------------------------------------------------------------- Fig. 3
def fig3():
    t = read_table("fig3_vertices_uncertainties_field.csv")
    fig = plt.figure(figsize=(WIDTH_2COL, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05])
    axs = fig.add_subplot(gs[0, :])
    axs.set_xlim(0, 100)
    axs.set_ylim(0, 31)
    axs.axis("off")
    routes = [
        (
            "a",
            "Intensity-space uncertainty (§2.8.1–2.8.2)",
            [
                "Repeated reconstructions\n(deep ensemble, MC dropout,\nDropsembles)",
                "Voxel-wise variability\n(standard deviation)",
                "Uncertainty in HU sampled\nat the vertices (Eq. 27)",
            ],
            "white",
        ),
        (
            "b",
            "Surface-space uncertainty (§2.8.3)",
            [
                "Repeated reconstructions",
                "Segmentation and signed\ndistance fields (Eq. 31)",
                "Surface uncertainty in mm\n($u_{\\mathrm{geo,std}}$, $u_{\\mathrm{geo,abs}}$, $u_{\\mathrm{geo,MC}}$;\nEqs. 32–34)",
            ],
            "white",
        ),
        (
            "c",
            "Proposed geometric reliability field (§2.7)",
            [
                "Single reconstructed\ntarget surface",
                "Nine vertex-wise\ndescriptors x(v) (Eq. 7)",
                "Predicted geometric error\n$\\hat{e}(v)$ in mm (Eq. 19)",
            ],
            LIGHT_GRAY,
        ),
    ]
    wb, hb, gap = 30.0, 6.6, 5.0
    for i, (letter, tit, steps, fc) in enumerate(routes):
        y = 22.0 - i * 10.5
        axs.text(
            0,
            y + hb + 0.6,
            f"({letter}) {tit}",
            fontsize=7.5,
            weight="bold",
            va="bottom",
        )
        for j, text in enumerate(steps):
            x = j * (wb + gap)
            box(axs, (x, y), wb, hb, text, fc=fc)
            if j < 2:
                arrow(axs, (x + wb, y + hb / 2), (x + wb + gap, y + hb / 2))
    vmax_hu = float(np.percentile(t["u_ens_HU"], 99))
    vmax_mm = float(
        max(np.percentile(t["u_geo_std_mm"], 99), np.percentile(t["e_hat_mm"], 99))
    )
    u, v, p = t["proj_u"], t["proj_v"], t["depth"]
    a1 = fig.add_subplot(gs[1, 0])
    a2 = fig.add_subplot(gs[1, 1])
    a3 = fig.add_subplot(gs[1, 2])
    foot = foot_method()
    s1 = mesh(a1, u, v, p, t["u_ens_HU"], 0, vmax_hu, foot=foot)
    mesh(a2, u, v, p, t["u_geo_std_mm"], 0, vmax_mm, foot=foot)
    s3 = mesh(a3, u, v, p, t["e_hat_mm"], 0, vmax_mm, foot=foot)
    title(a1, "d", "$u_{\\mathrm{ens}}(v)$, intensity space")
    title(a2, "e", "$u_{\\mathrm{geo,std}}(v)$, surface space")
    title(a3, "f", "$\\hat{e}(v)$, predicted geometric error")
    colorbar_(fig, s1, [a1], "HU", fraction=0.05, pad=0.01, shrink=0.75)
    colorbar_(
        fig, s3, [a2, a3], "mm (same scale)", fraction=0.025, pad=0.01, shrink=0.75
    )
    save(fig, "fig3_uncertainty_spaces")
    return {
        "vmax_HU_p99": vmax_hu,
        "vmax_mm_p99": vmax_mm,
        "u_ens_median_HU": float(np.median(t["u_ens_HU"])),
        "u_geo_std_median_mm": float(np.median(t["u_geo_std_mm"])),
        "e_hat_median_mm": float(np.median(t["e_hat_mm"])),
    }


# ----------------------------------------------------------------------------- Fig. 4
def fig4():
    t = read_table("fig4_vertices.csv")
    corr = read_table("fig4_correspondences_per_strategy.csv")
    ens = read_table("fig4_strategies_one_trial.csv")
    contr = read_table("fig4_contrast_regions.csv")
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
        s: float(d) for s, d in zip(ens["strategy"], ens["d_target_mm"], strict=True)
    }
    reg_contr = {
        s: int(r) for s, r in zip(contr["region"], contr["kmeans_region"], strict=True)
    }
    d_contr = {
        s: float(d) for s, d in zip(contr["region"], contr["median_d_mm"], strict=True)
    }
    order = [
        ("field", "Reliability-guided (lowest $\\hat{e}$)"),
        ("d_shape", "Shape disagreement (lowest $d_{\\mathrm{shape}}$)"),
        ("random", "Random surface selection"),
        ("oracle", "Oracle (lowest measured $e$)"),
        ("global", "Global surface (K = 200)"),
    ]
    fig = plt.figure(figsize=(WIDTH_2COL, 4.4))
    gs = fig.add_gridspec(
        2, 3, hspace=0.25, wspace=0.03, left=0.005, right=0.93, top=0.93, bottom=0.08
    )
    ax = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    sc = None
    foot = foot_method()
    s_corr = 1.8 if MODE == "points" else 3.0
    for a, letter, (s, tit) in zip(ax[:5], "abcde", order, strict=True):
        mask = [(rot == reg_of[s], GRAY_REGION)] if reg_of[s] >= 0 else []
        sc = mesh(a, u, v, p, e_hat, 0, vmax, s=0.3, foot=foot, masks=mask)
        a.scatter(
            u[locs[s]], v[locs[s]], s=s_corr, color=RED, linewidths=0, zorder=4
        )
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
        title(
            a,
            letter,
            f"{tit}\n$d_{{\\mathrm{{target}}}}$ = {d_target[s]:.2f} mm (one trial)",
            fs=7,
        )
    a = ax[5]
    mask = [
        (rot == reg_contr[s], color)
        for s, color in (("lowest_e_hat", RED), ("highest_e_hat", BLACK))
    ]
    mesh(a, u, v, p, e_hat, 0, vmax, s=0.6, foot=foot, masks=mask)
    title(
        a,
        "f",
        "Lowest- vs. highest-$\\hat{e}$ region (300 trials each)\n"
        f"median $d_{{\\mathrm{{target}}}}$ = {d_contr['lowest_e_hat']:.2f} mm vs. {d_contr['highest_e_hat']:.2f} mm",
        fs=7,
    )
    colorbar_(
        fig,
        sc,
        ax,
        "$\\hat{e}(v)$, predicted geometric error (mm)",
        fraction=0.02,
        pad=0.01,
        shrink=0.6,
    )
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                ms=3,
                color=GRAY_REGION,
                label="selected k-means region (§2.9.2)",
            ),
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                ms=2.5,
                color=RED,
                label="K = 200 sampled correspondences",
            ),
            Line2D(
                [],
                [],
                ls="",
                marker="*",
                ms=6,
                color="white",
                markeredgecolor=BLACK,
                label="remote target (> 80th percentile)",
            ),
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                ms=3,
                color=RED,
                label="(f) lowest-$\\hat{e}$ region",
            ),
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                ms=3,
                color=BLACK,
                label="(f) highest-$\\hat{e}$ region",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.47, 0.0),
        ncol=5,
        frameon=False,
        fontsize=6.5,
        handletextpad=0.3,
        columnspacing=1.2,
    )
    save(fig, "fig4_registration")
    return {
        "vmax_mm_p99": vmax,
        "d_target_one_trial_mm": d_target,
        "median_contrast_mm": d_contr,
    }


# ----------------------------------------------------------------------------- Fig. 5
def fig5():
    t = read_table("fig5ab_vertices.csv")
    per_foot = read_table("foot_per_case.csv", DATA_RES)
    rho_sr, rho_tri = per_foot["rho_vertex_field_sr"], per_foot["rho_vertex_field_tri"]
    e, e_hat = t["e_mm"], t["e_hat_mm"]
    px, py, pz = t["px"], t["py"], t["depth"]
    # framing by the largest connected component (distant residual fragments are still plotted, but outside the
    # window) — same criterion as a4_results_figures; (px, py, depth) is the vertex in the PCA frame, distances preserved
    principal = largest_component(np.c_[px, py, pz])
    xlim = (px[principal].min() - 2, px[principal].max() + 2)
    ylim = (py[principal].min() - 2, py[principal].max() + 2)
    vmax = 2.0  # mm; same scale in both panels; above this it saturates (state in the caption)
    foot = foot_results()
    rho_foot = float(rho_sr[per_foot["foot"] == foot][0])  # r32, not recomputed

    fig = plt.figure(figsize=(WIDTH_2COL, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.7, 0.9])
    a1, a2, a3 = (
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[:, 1]),
    )
    mesh(a1, px, py, pz, e, 0, vmax, s=1.0, foot=foot)
    sc = mesh(a2, px, py, pz, e_hat, 0, vmax, s=1.0, foot=foot)
    for a in (a1, a2):
        a.set_xlim(xlim)
        a.set_ylim(ylim)
    title(a1, "a", "Measured surface error $e(v)$")
    title(
        a2,
        "b",
        f"Predicted reliability field $\\hat{{e}}(v)$ (Spearman $\\rho$ = {rho_foot:.3f})",
    )
    colorbar_(
        fig,
        sc,
        [a1, a2],
        "mm (same scale; values above 2 mm saturated)",
        fraction=0.04,
        pad=0.01,
        shrink=0.7,
    )
    rng = np.random.default_rng(
        0
    )  # horizontal jitter only for readability; the CSV has no jitter
    for i, vals in enumerate((rho_sr, rho_tri)):
        x = i + rng.uniform(-0.13, 0.13, len(vals))
        a3.scatter(x, vals, s=7, color=GRAY, alpha=0.7, linewidths=0)
        q1, q2, q3 = np.percentile(vals, [25, 50, 75])
        a3.hlines([q1, q3], i - 0.25, i + 0.25, color=BLACK, lw=LW)
        a3.hlines(q2, i - 0.3, i + 0.3, color=BLACK, lw=1.4)
    a3.set_xticks([0, 1])
    a3.set_xticklabels(["Super-\nresolution", "Trilinear"])
    a3.set_xlim(-0.6, 1.6)
    a3.set_ylim(0, None)
    a3.set_ylabel("Vertex-level Spearman $\\rho$ per foot")
    title(a3, "c", "Cohort ($n$ = 48)")
    a3.spines[["top", "right"]].set_visible(False)
    save(fig, "fig5_field_vs_error")
    return {
        "foot": foot,
        "rho_spearman_foot": rho_foot,
        "vmax_mm": vmax,
        "vertices_outside_largest_component": int((~principal).sum()),
        "rho_sr_median": float(np.median(rho_sr)),
        "rho_tri_median": float(np.median(rho_tri)),
    }


# ----------------------------------------------------------------------------- Fig. 6
def fig6():
    d = read_table("foot_calibration_deciles.csv", DATA_RES)
    px, px1, px3 = d["predicted_median_mm"], d["predicted_q1"], d["predicted_q3"]
    oy, oy1, oy3 = d["observed_median_mm"], d["observed_q1"], d["observed_q3"]
    n, dec = d["n_feet"].astype(int), d["decile"].astype(int)
    fig, ax = plt.subplots(figsize=(WIDTH_1COL, 3.3), constrained_layout=True)
    lim = (0.3, float(max(px3.max(), oy3.max())) * 1.05)
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
        label="Decile of $\\hat{e}(v)$: median across feet, IQR",
    )
    partial = [(int(dec[k]), int(n[k])) for k in range(len(n)) if n[k] < 48]
    if partial:
        decs = ", ".join(str(a) for a, _ in partial)
        ns = ", ".join(str(b) for _, b in partial)
        ax.text(
            0.98,
            0.03,
            f"Deciles {decs}: $n$ = {ns} feet (tied predictions)",
            transform=ax.transAxes,
            fontsize=6.5,
            ha="right",
            va="bottom",
        )
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted error, decile median (mm)")
    ax.set_ylabel("Measured error, decile median (mm)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig6_calibration_deciles")
    return {"deciles": len(n), "n_feet_deciles_1_3": n[:3].tolist()}


def main():
    global MODE, FIGS_OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="1,2,3,4,5,6")
    ap.add_argument(
        "--mode",
        default="surface",
        choices=("points", "shaded", "surface"),
        help="how the meshes are drawn",
    )
    ap.add_argument("--output", default="", help="subfolder of figs/ (default: clean)")
    args = ap.parse_args()
    MODE = args.mode
    if args.output:
        FIGS_OUT = os.path.join(FIGS, args.output)
    which = set(args.only.split(",")) if args.only else {"1", "2", "3", "4", "5", "6"}
    fns = {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6}
    prov = {
        "script": os.path.relpath(__file__, ROOT),
        "data": dt.datetime.now(dt.UTC).astimezone().isoformat(timespec="seconds"),
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "style": "journal: white background, rectangles, 0.6 pt, Arial 8/7 pt, 190 mm or 90 mm, vector PDF + PNG 600 dpi",
        "mesh_mode": MODE,
        "figures": {},
    }
    for k in sorted(which):
        prov["figures"][f"fig{k}"] = fns[k]()
    prov["csv_sources_sha256"] = dict(sorted(SOURCES_USED.items()))
    if MODE == "surface":
        prov["reextracted_meshes"] = {
            foot: RENDER.verification(foot) for foot in {foot_method(), foot_results()}
        }
    os.makedirs(FIGS_OUT, exist_ok=True)
    with open(
        os.path.join(FIGS_OUT, "clean_figures_provenance.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(prov, f, indent=1, ensure_ascii=False)
    print(json.dumps(prov["figures"], ensure_ascii=False))


if __name__ == "__main__":
    main()
