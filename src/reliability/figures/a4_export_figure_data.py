"""Exports the data of the new figures of the paper: foot CT slices for the
pipeline infographic (Fig. 1) and the thorax mesh, slices and correlations (Fig. 6; the export files keep the
internal prefix fig7_). Outside the package
hash (subfolder `figures/`); uses the SAME chain as a4_run_foot / a4_run_thorax (like a4_mesh_faces.py) and checks
the regenerated vertices against the cache. Writes nothing to foot_cache / thorax_cache nor to results/.

Output: output/figures/figs/data/ (CSV) and figs/data/meshes/<case>.npz.
Usage: python src/reliability/figures/a4_export_figure_data.py [--only foot,thorax]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import a4_config as C
import a4_data as D
import a4_statistics as E
import a4_sr as SR
import a4_surface as S
from a4_run_foot import prepare_foot
from a4_run_thorax import prepare_thorax

OUTPUT_DIR = os.path.join(C.ROOT, "output", "figures")
FIGS = os.path.join(OUTPUT_DIR, "figs")
DATA = os.path.join(FIGS, "data")
MESHES = os.path.join(DATA, "meshes")
DATA_RES = os.path.join(OUTPUT_DIR, "results_figure_data")


def export_matrix(name, M, extent_mm):
    """2D image as CSV (rows = vertical axis) + physical extent in the header (same format as figs/data/fig2*)."""
    with open(os.path.join(DATA, name + ".csv"), "w", encoding="utf-8") as f:
        f.write(
            "# extent_mm (h0, h1, v_bottom, v_top) = %s\n"
            % (list(map(float, extent_mm)),)
        )
        np.savetxt(f, np.asarray(M, float), delimiter=",", fmt="%.6g")


def export(name, **columns):
    cols = {k: np.asarray(v).reshape(-1) for k, v in columns.items()}
    n = {len(v) for v in cols.values()}
    assert len(n) == 1, (name, {k: len(v) for k, v in cols.items()})
    with open(
        os.path.join(DATA, name + ".csv"), "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(list(cols))
        for i in range(n.pop()):
            w.writerow([cols[k][i] for k in cols])


def pca_projection(V, base=None):
    """Orthographic projection onto the two principal directions; `base` = mask of the vertices that define the axes."""
    b = V if base is None else V[base]
    c = b.mean(axis=0)
    _u, _s, vt = np.linalg.svd(b - c, full_matrices=False)
    P = (V - c) @ vt.T
    return P[:, 0], P[:, 1], P[:, 2]


def save_mesh(ident, V, F, Vc, extra):
    d_rc, idx = cKDTree(Vc).query(V)
    d_cr, _ = cKDTree(V).query(Vc)
    conf = {
        "case": ident,
        "n_vertices_regen": len(V),
        "n_vertices_cache": len(Vc),
        "n_faces": len(F),
        "identical_vertices_in_same_order": bool(
            len(V) == len(Vc) and np.allclose(V, Vc, atol=1e-3)
        ),
        "dist_regen_to_cache_mm_max": float(d_rc.max()),
        "dist_cache_to_regen_mm_max": float(d_cr.max()),
        "sr_iters": int(C.SR_ITERS),
        **extra,
    }
    os.makedirs(MESHES, exist_ok=True)
    np.savez_compressed(
        os.path.join(MESHES, ident + ".npz"),
        V=V.astype(np.float32),
        faces=F.astype(np.int32),
        idx_cache=idx.astype(np.int32),
        d_regen_to_cache=d_rc.astype(np.float32),
        verification=json.dumps(conf, ensure_ascii=False),
    )
    with open(
        os.path.join(MESHES, ident + ".verification.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(conf, f, indent=1, ensure_ascii=False)
    print(json.dumps(conf, ensure_ascii=False), flush=True)
    return conf


# ----------------------------------------------------------------------------- foot (infographic)
def export_foot(dev):
    with open(os.path.join(FIGS, "figures_provenance.json"), encoding="utf-8") as f:
        prov = json.load(f)
    ident, ix = prov["foot"], int(prov["figures"]["fig2"]["sagittal_plane_index_x"])
    listing = json.load(open(C.FOOT_LIST_N48, encoding="utf-8"))
    filepath = next(p for p in listing if os.path.basename(p) == ident)
    t0 = time.time()
    hu_t, thick, grid, spacing, thick_spacing, roi, _thick_roi = prepare_foot(filepath, C.K_FOOT)
    tri = D.interpolate_trilinear_z(thick, C.K_FOOT)
    tn, lo, esc = SR.normalize(thick)
    network, _info = SR.train(
        SR.in_plane_slices(tn),
        C.K_FOOT,
        C.SR_ITERS,
        dev,
        seed=C.SR_SEED_EVALUATED,
        verbose=False,
    )
    rec = 0.5 * (
        SR.apply(network, tn, C.K_FOOT, orientations=("yz",))
        + SR.apply(network, tn, C.K_FOOT, orientations=("xz",))
    )
    rec = SR.denormalize(rec, lo, esc)
    del network
    m_sr = S.segment_bone(rec, roi, spacing)
    # check: the mesh of this reconstruction reproduces the cache vertices
    mesh = S.mask_to_mesh(m_sr, grid)
    z = np.load(os.path.join(C.A4_CACHE_FOOT, ident + ".npz"), allow_pickle=False)
    Vc = z["V"].astype(float)
    dmax = float(cKDTree(Vc).query(np.asarray(mesh.vertices, float))[0].max())
    assert dmax < 1e-2, ("regenerated SR differs from the cache", dmax)
    sy, sz, thick_sz = spacing[1], spacing[2], thick_spacing[2]
    ext_f = [0, hu_t.shape[1] * sy, hu_t.shape[0] * sz, 0]
    ext_e = [0, thick.shape[1] * sy, thick.shape[0] * thick_sz, 0]
    export_matrix("fig1_thin_ct_sagittal_HU", hu_t[:, :, ix], ext_f)
    export_matrix("fig1_thick_ct_sagittal_HU", thick[:, :, ix], ext_e)
    export_matrix("fig1_ct_trilinear_sagittal_HU", tri[:, :, ix], ext_f)
    export_matrix("fig1_ct_sr_sagittal_HU", rec[:, :, ix], ext_f)
    export_matrix("fig1_mask_sr_sagittal", m_sr[:, :, ix].astype(np.uint8), ext_f)
    # per-vertex descriptors (cache), in the order of the cache vertices
    F = z["F"].astype(float)
    export(
        "fig1_vertices_descriptors",
        **{nm: F[:, j] for j, nm in enumerate(C.FEATURE_NAMES)},
        e_measured_mm=z["e"].astype(float),
    )
    print(
        "foot ok",
        ident,
        "ix",
        ix,
        "dmax_cache_mm",
        dmax,
        "%.0fs" % (time.time() - t0),
        flush=True,
    )
    return {"foot": ident, "ix": ix, "dmax_cache_mm": dmax}


# ----------------------------------------------------------------------------- thorax
def export_thorax(dev):
    with open(
        os.path.join(DATA_RES, "thorax_per_case.csv"), encoding="utf-8", newline=""
    ) as f:
        rows = list(csv.DictReader(f))
    rho = np.array([float(r["rho_vertex"]) for r in rows])
    cases = [r["case"] for r in rows]
    ident = cases[
        int(np.argmin(np.abs(rho - np.median(rho))))
    ]  # rho closest to the cohort median
    thin, thk, _id, part = next(p for p in D.rplhr_pairs() if p[2] == ident)
    assert part == "test", part
    t0 = time.time()
    hu_ref, thick, grid, spacing_f, _thick_spacing, roi, _thick_roi, k, _info = prepare_thorax(
        thin, thk
    )
    tn, lo, esc = SR.normalize(thick)
    network, _info_sr = SR.train(
        SR.in_plane_slices(tn),
        k,
        C.SR_ITERS,
        dev,
        seed=C.SR_SEED_EVALUATED,
        verbose=False,
    )
    rec = SR.denormalize(SR.apply(network, tn, k), lo, esc)
    del network
    mesh = S.mask_to_mesh(S.segment_bone(rec, roi, spacing_f), grid)
    V, Fc = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)
    z = np.load(os.path.join(C.A4_CACHE_THORAX, ident + ".npz"), allow_pickle=False)
    Vc, e = z["V"].astype(float), z["e"].astype(float)
    conf = save_mesh(
        ident, V, Fc, Vc, {"time_s": round(time.time() - t0, 1), "partition": part}
    )

    # ê(v) of the held-out case: LOFO memo of the full thorax model (the same as r36)
    def memo(label):
        j = json.load(
            open(
                os.path.join(
                    C.A4_RESULTS,
                    "rf_selection_%s_%s.json" % (label, C.RF_SEARCH_SAMPLER),
                ),
                encoding="utf-8",
            )
        )
        return np.load(
            os.path.join(C.A4_RESULTS, "_lofo_%s_%s.npz" % (label, j["key"])),
            allow_pickle=False,
        )

    pred_sr, pred_tri = memo("thorax_sr"), memo("thorax_tri")
    e_hat = pred_sr["pred_" + ident]
    assert len(e_hat) == len(Vc)
    u, v, p = pca_projection(Vc)
    export(
        "fig7_vertices",
        x_mm=Vc[:, 0],
        y_mm=Vc[:, 1],
        z_mm=Vc[:, 2],
        proj_u=u,
        proj_v=v,
        depth=p,
        e_measured_mm=e,
        e_hat_mm=e_hat,
    )
    # coronal slices (y fixed at the median of the vertices): thick 5 mm, SR and 1 mm reference
    idx = grid.physical_to_index(Vc)
    iy = int(np.round(np.median(idx[:, 1])))
    sx, sz = spacing_f[0], spacing_f[2]
    ext_f = [0, hu_ref.shape[2] * sx, hu_ref.shape[0] * sz, 0]
    ext_e = [0, thick.shape[2] * sx, thick.shape[0] * sz * k, 0]
    export_matrix("fig7_thick_ct_coronal_HU", thick[:, iy, :], ext_e)
    export_matrix("fig7_ct_sr_coronal_HU", rec[:, iy, :], ext_f)
    export_matrix("fig7_reference_ct_coronal_HU", hu_ref[:, iy, :], ext_f)
    # rho per case: SR (r36, from the CSV) and trilinear (recomputed from the memo and checked against the r36 median)
    rho_tri = []
    for c in cases:
        zc = np.load(os.path.join(C.A4_CACHE_THORAX, c + ".npz"), allow_pickle=False)
        rho_tri.append(E.spearman(pred_tri["pred_" + c], zc["e_tri"].astype(float)))
    rho_tri = np.array(rho_tri)
    r36 = json.load(
        open(os.path.join(C.A4_RESULTS, "r36_thorax.json"), encoding="utf-8")
    )
    med_r36 = r36["trilinear"]["rho_vertex"]["median"]
    assert abs(float(np.median(rho_tri)) - med_r36) < 1e-9, (
        float(np.median(rho_tri)),
        med_r36,
    )
    export("fig7_rho_per_case", case=cases, rho_sr=rho, rho_tri=rho_tri)
    out = {
        "case": ident,
        "criterion": "per-vertex rho closest to the cohort median (thorax_per_case.csv)",
        "rho_case": float(rho[cases.index(ident)]),
        "rho_spearman_recomputed": E.spearman(e_hat, e),
        "iy": iy,
        "verification": conf,
        "rho_tri_median_matches_r36": True,
    }
    print(
        "thorax ok",
        json.dumps({k: v for k, v in out.items() if k != "verification"}),
        flush=True,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="foot,thorax")
    args = ap.parse_args()
    dev = SR.pick_device(require_cuda=True)
    prov = {"environment": C.environment_record(with_torch=False)}
    if "foot" in args.only:
        prov["foot"] = export_foot(dev)
    if "thorax" in args.only:
        prov["thorax"] = export_thorax(dev)
    with open(
        os.path.join(FIGS, "figure_data_provenance.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(prov, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
