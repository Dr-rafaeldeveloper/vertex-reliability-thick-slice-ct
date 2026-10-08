"""Mechanism of the negative association between intensity-space uncertainty and the surface error (Section 3.4):
per foot, the ensemble uncertainty u_ens(v), the measured error e(v), the local Hounsfield value and the gradient
magnitude of the thick-slice volume at every vertex, with Spearman correlations and the partial Spearman correlation
of u_ens and e given the local intensity and gradient. Uses the thick-slice volume of the main chain (prepare_foot,
Eq. 1) and the caches. CPU only. Outside the package hash (subfolder).

Output: results/rS_intensity_mechanism.json
Usage: python src/reliability/supplement/intensity_uncertainty_mechanism.py [--n-feet N]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from reliability import a4_config as C  # noqa: E402
from reliability import a4_statistics as E  # noqa: E402
from reliability.a4_run_foot import prepare_foot  # noqa: E402


def sp(a, b):
    return float(spearmanr(a, b)[0])


def partial_spearman(x, y, Z):
    """Spearman correlation of x and y after removing the rank-linear dependence of both on the columns of Z."""
    rx, ry = rankdata(x), rankdata(y)
    R = np.column_stack([rankdata(Z[:, j]) for j in range(Z.shape[1])] + [np.ones(len(x))])
    bx = np.linalg.lstsq(R, rx, rcond=None)[0]
    by = np.linalg.lstsq(R, ry, rcond=None)[0]
    return float(np.corrcoef(rx - R @ bx, ry - R @ by)[0, 1])


def sample(volume_zyx, idx_zyx):
    return map_coordinates(volume_zyx, idx_zyx, order=1, mode="nearest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-feet", type=int, default=0)
    args = ap.parse_args()
    listing = json.load(open(C.FOOT_LIST_N48, encoding="utf-8"))
    if args.n_feet:
        listing = listing[: args.n_feet]
    per = {}
    for filepath in listing:
        ident = os.path.basename(filepath)
        z = np.load(os.path.join(C.A4_CACHE_FOOT, ident + ".npz"), allow_pickle=False)
        V, e, u_ens, u_geo = z["V"].astype(float), z["e"].astype(float), z["u_ens"].astype(float), z["u_geo_std"].astype(float)
        F = z["F"].astype(float)
        hu_t, thick, grid, spacing, thick_spacing, _roi, _tr = prepare_foot(filepath, C.K_FOOT)
        g = grid.as_dict()
        org, sp_ = np.asarray(g["origin_mm"], float), np.asarray(g["spacing_mm"], float)  # xyz
        idx_xyz = (V - org) / sp_
        idx_thin = idx_xyz[:, ::-1].T  # zyx
        idx_thick = idx_thin.copy()
        idx_thick[0] = (idx_thin[0] - (C.K_FOOT - 1) / 2.0) / C.K_FOOT
        hu_thin = sample(hu_t, idx_thin)
        hu_thick = sample(thick, idx_thick)
        dz, dy, dx = float(thick_spacing[2]), float(spacing[1]), float(spacing[0])
        gz, gy, gx = np.gradient(thick, dz, dy, dx)
        gmag = sample(np.sqrt(gx**2 + gy**2 + gz**2), idx_thick)
        per[ident] = {
            "n": int(len(e)),
            "rho_uens_e": sp(u_ens, e), "rho_uens_hu_thick": sp(u_ens, hu_thick), "rho_uens_hu_thin": sp(u_ens, hu_thin), "rho_uens_gradient": sp(u_ens, gmag),
            "rho_e_hu_thick": sp(e, hu_thick), "rho_e_gradient": sp(e, gmag), "rho_ugeo_std_hu_thick": sp(u_geo, hu_thick),
            "partial_uens_e_given_hu_thick": partial_spearman(u_ens, e, hu_thick[:, None]),
            "partial_uens_e_given_gradient": partial_spearman(u_ens, e, gmag[:, None]),
            "partial_uens_e_given_hu_and_gradient": partial_spearman(u_ens, e, np.column_stack([hu_thick, gmag])),
            "partial_uens_e_given_descriptors": partial_spearman(u_ens, e, F[:, :6]),
        }
        q = np.quantile(hu_thick, [0.25, 0.5, 0.75])
        s = np.digitize(hu_thick, q)
        per[ident]["uens_median_by_hu_quartile"] = [float(np.median(u_ens[s == i])) for i in range(4)]
        per[ident]["e_median_by_hu_quartile"] = [float(np.median(e[s == i])) for i in range(4)]
        print(ident[:4], {k: round(v, 3) for k, v in per[ident].items() if isinstance(v, float)}, flush=True)
        del hu_t, thick, gx, gy, gz
    keys = [k for k in next(iter(per.values())) if isinstance(next(iter(per.values()))[k], float)]
    out = {
        "description": "u_ens against e, local HU (thin and thick volume) and gradient magnitude of the thick volume at the vertices; Spearman and partial Spearman (rank-linear adjustment)",
        "n_feet": len(per),
        "summary": {k: E.summary([per[h][k] for h in per]) for k in keys},
        "uens_median_by_hu_quartile": [float(np.median([per[h]["uens_median_by_hu_quartile"][i] for h in per])) for i in range(4)],
        "e_median_by_hu_quartile": [float(np.median([per[h]["e_median_by_hu_quartile"][i] for h in per])) for i in range(4)],
        "per_case": per,
        "environment": C.environment_record(with_torch=False),
    }
    p = os.path.join(C.A4_RESULTS, "rS_intensity_mechanism.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    for k in keys:
        s = out["summary"][k]
        print("%s: %.3f (%.3f-%.3f)" % (k, s["median"], *s["iqr"]))
    print("u_ens by HU quartile:", np.round(out["uens_median_by_hu_quartile"], 2), "| e by HU quartile:", np.round(out["e_median_by_hu_quartile"], 3))
    print("->", p)


if __name__ == "__main__":
    main()
