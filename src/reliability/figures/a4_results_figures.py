"""Results figures (Section 3), generated only from the caches/JSONs/CSVs of the current round: predicted field x
measured error on the representative foot + rho per foot, and calibration by deciles (foot). The internal names
fig5/fig6 are export labels (manuscript: field vs error = Fig. 2, calibration = Fig. 3).

Outside the package hash (subfolder `figures/`). Reuses the utilities of a4_method_figures (load_foot,
loo_prediction, projection_2d, export, save). Labels in English with the terms of the manuscript; vector PDF
(meshes rasterized at 600 dpi) + 600 dpi PNG; serif font (STIX, compatible with elsarticle). Output in
`output/figures/figs/`, plotted data in `figs/data/`, provenance in
`figs/results_figures_provenance.json`.

Representative foot: the one with per-vertex rho closest to the cohort median (z001, 0.415 vs 0.414;
foot_per_case.csv).

Usage: python src/reliability/figures/a4_results_figures.py [--foot z001_foot.nii.gz]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import a4_config as C
import a4_statistics as E
from a4_method_figures import (
    CMAP,
    COLOR,
    FIGS,
    load_foot,
    export,
    loo_prediction,
    projection_2d,
)

DATA_RES = os.path.join(os.path.dirname(FIGS), "results_figure_data")
DPI = 600
plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": DPI,
        "pdf.fonttype": 42,
    }
)


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print("figure saved:", name)


def read_csv(name):
    with open(os.path.join(DATA_RES, name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mesh_panel(ax, P, depth, val, title, vmax):
    o = np.argsort(depth)  # paints first what is behind
    sc = ax.scatter(
        P[o, 0],
        P[o, 1],
        c=val[o],
        s=2.4,
        cmap=CMAP,
        vmin=0.0,
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
    return sc


def largest_component(V, radius_mm=3.0):
    """Boolean mask of the vertices of the largest connected component of the neighborhood graph (pairs closer than
 radius_mm). The cache stores vertices, not faces; the radius (3 mm = k*Δz) links vertices of the same decimated
 surface."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pairs = cKDTree(V).query_pairs(r=radius_mm, output_type="ndarray")
    g = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(V), len(V))
    )
    _n, rot = connected_components(g, directed=False)
    return rot == np.bincount(rot).argmax()


# ----------------------------------------------------------------------------- Fig. 5
def fig5(ident, cache, e_hat):
    """(a) measured e(v) and (b) predicted ê(v) on the representative foot, same scale; (c) rho per foot (SR and
 trilinear)."""
    V, e = cache["V"], cache["e"]
    # projection and framing by the largest connected component of the mesh: small, distant fragments (residual
    # components of the segmentation) remain in the exported data, but do not define the view
    principal = largest_component(V)
    P, depth = projection_2d(V[principal])
    c = V[principal].mean(axis=0)
    _u, _s, vt = np.linalg.svd(V[principal] - c, full_matrices=False)
    P = (V - c) @ vt[:2].T
    depth = (V - c) @ vt[2]
    lim = [(float(P[principal, k].min()), float(P[principal, k].max())) for k in (0, 1)]
    vmax = 2.0  # mm; same scale in both panels, values above are saturated (stated in the caption)
    per_foot = read_csv("foot_per_case.csv")
    rho_sr = np.array([float(r["rho_vertex_field_sr"]) for r in per_foot])
    rho_tri = np.array([float(r["rho_vertex_field_tri"]) for r in per_foot])
    rho_foot = E.spearman(e_hat, e)

    # (a) and (b) stacked on the left (the foot is elongated: each panel uses the full width); (c) on the right across
    # both rows
    fig = plt.figure(figsize=(7.0, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.6, 0.9])
    axs = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[:, 1]),
    ]
    mesh_panel(axs[0], P, depth, e, "(a) Measured surface error $e(v)$", vmax)
    sc = mesh_panel(
        axs[1],
        P,
        depth,
        e_hat,
        rf"(b) Predicted field $\hat{{e}}(v)$ ($\rho$ = {rho_foot:.3f})",
        vmax,
    )
    for ax in axs[:2]:
        m = 0.03 * max(lim[0][1] - lim[0][0], lim[1][1] - lim[1][0])
        ax.set_xlim(lim[0][0] - m, lim[0][1] + m)
        ax.set_ylim(lim[1][0] - m, lim[1][1] + m)
    cb = fig.colorbar(sc, ax=axs[:2], fraction=0.04, pad=0.01, shrink=0.8)
    cb.set_label("Surface error (mm)")
    cb.ax.tick_params(labelsize=7)
    ax = axs[2]
    rng = np.random.default_rng(0)  # jitter only for readability; CSV without jitter
    for i, (vals, color) in enumerate(((rho_sr, COLOR["sr"]), (rho_tri, COLOR["ref"]))):
        x = i + rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(x, vals, s=8, color=color, alpha=0.75, linewidths=0)
        q1, q2, q3 = np.percentile(vals, [25, 50, 75])
        ax.hlines([q1, q3], i - 0.25, i + 0.25, color="k", lw=0.7)
        ax.hlines(q2, i - 0.3, i + 0.3, color="k", lw=1.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Super-\nresolution", "Trilinear"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(r"Vertex-level Spearman $\rho$ (per foot)")
    ax.set_title("(c) Cohort ($n$ = 48)")
    ax.axhline(0, color="#999999", lw=0.5, ls=":")
    save(fig, "fig5_field_vs_error")
    export(
        "fig5ab_vertices", px=P[:, 0], py=P[:, 1], depth=depth, e_mm=e, e_hat_mm=e_hat
    )
    export(
        "fig5c_rho_per_foot", foot=[r["foot"] for r in per_foot], rho_sr=rho_sr, rho_tri=rho_tri
    )
    return {
        "foot": ident,
        "rho_spearman_foot": rho_foot,
        "vmax_mm": vmax,
        "vertices_outside_largest_component": int((~principal).sum()),
        "rho_sr_median": float(np.median(rho_sr)),
        "rho_tri_median": float(np.median(rho_tri)),
        "source_c": "results_figure_data/foot_per_case.csv (r32)",
    }


# ----------------------------------------------------------------------------- Fig. 6
def fig6():
    """Calibration by deciles of the predicted error (foot): median across feet of the observed x predicted median per
 decile, IQR across feet on both axes, identity. Source: results_figure_data/foot_calibration_deciles.csv
 (r33.sr_deciles)."""
    d = read_csv("foot_calibration_deciles.csv")
    px = np.array([float(r["predicted_median_mm"]) for r in d])
    px1 = np.array([float(r["predicted_q1"]) for r in d])
    px3 = np.array([float(r["predicted_q3"]) for r in d])
    oy = np.array([float(r["observed_median_mm"]) for r in d])
    oy1 = np.array([float(r["observed_q1"]) for r in d])
    oy3 = np.array([float(r["observed_q3"]) for r in d])
    n = np.array([int(r["n_feet"]) for r in d])
    dec = np.array([int(r["decile"]) for r in d])

    fig, ax = plt.subplots(figsize=(3.5, 3.4), constrained_layout=True)
    lim = (0.3, float(max(px3.max(), oy3.max())) * 1.05)
    ax.plot(lim, lim, color="#999999", lw=0.8, ls="--", label="Identity")
    ax.errorbar(
        px,
        oy,
        xerr=[px - px1, px3 - px],
        yerr=[oy - oy1, oy3 - oy],
        fmt="o",
        ms=3.5,
        color=COLOR["field"],
        ecolor=COLOR["field"],
        elinewidth=0.7,
        capsize=1.5,
        label=r"Decile of $\hat{e}(v)$ (median across feet; IQR)",
    )
    partial = [(int(dec[k]), int(n[k])) for k in range(len(d)) if n[k] < 48]
    if partial:
        ax.text(
            0.98,
            0.03,
            "Deciles %s: $n$ = %s feet\n(tied predictions)"
            % (
                ", ".join(str(a) for a, _ in partial),
                ", ".join(str(b) for _, b in partial),
            ),
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
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig6_calibration_deciles")
    export(
        "fig6_deciles",
        decile=dec,
        predicted_median=px,
        predicted_q1=px1,
        predicted_q3=px3,
        observed_median=oy,
        observed_q1=oy1,
        observed_q3=oy3,
        n_feet=n,
    )
    return {
        "deciles": len(d),
        "n_feet_deciles_1_3": n[:3].tolist(),
        "source": "foot_calibration_deciles.csv (r33.sr_deciles)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--foot",
        default="z001_foot.nii.gz",
        help="representative foot (rho closest to the median)",
    )
    args = ap.parse_args()
    _filepath, cache = load_foot(args.foot)
    e_hat = loo_prediction(args.foot, cache)
    prov = {
        "foot": args.foot,
        "foot_criterion": "per-vertex rho closest to the cohort median (foot_per_case.csv)",
        "cache_hash": cache["meta"]["environment"]["code_hash_reliability"],
        "environment": C.environment_record(with_torch=False),
        "dpi": DPI,
        "figures": {"fig5": fig5(args.foot, cache, e_hat), "fig6": fig6()},
        "note": "figures generated from the exported data",
    }
    os.makedirs(FIGS, exist_ok=True)
    with open(
        os.path.join(FIGS, "results_figures_provenance.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(prov, f, indent=1, ensure_ascii=False)
    print(json.dumps(prov["figures"], ensure_ascii=False))


if __name__ == "__main__":
    main()
