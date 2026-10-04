"""Bone segmentation, surface extraction and per-vertex error (§2.5 and §2.6).

§2.5, in the order of the text:
 1. fixed threshold of 200 HU;
 2. "morphological cleaning to remove isolated foreground components and small discontinuities
 produced by thresholding" -> binary closing ×2 (replicated border) and removal of
 components < 50 mm3. Cavity filling REMOVED in the endosteal surfaces enter all meshes equally;
 3. "Connected-component analysis... excluding disconnected objects unrelated to the target
 anatomy" -> in the VOXEL grid: components < 50 mm3 removed + ROI of the thick volume (a4_roi); no mesh
 component filter (the CCA of the text precedes "before mesh extraction");
 4. "smoothed in physical space using an isotropic Gaussian kernel with a characteristic scale of
 0.8 mm" (sigma per axis = 0.8 / spacing);
 5. marching cubes [28] Lorensen & Cline -> method="lorensen";
 6. "12 iterations of Taubin smoothing" (λ = ν = 0.5 declared);
 7. "quadric-error mesh decimation... to approximately 60,000 triangular faces" (explicit error if
 the backend does not decimate);
 8. "expressed in physical coordinates": coordinates = origin + D·(index ⊙ spacing). Without caps: the surfaces stay
 open where the volume cuts the bone, equally
 in the reference and in the reconstructions; the fraction of border vertices is written to the cache (open_border).
§2.6: 120 000 points sampled uniformly from the reference (seed of the identifier);
e(v) = Euclidean distance to the nearest sampled point (Eq. 5), unsigned, in mm.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage as ndi
from skimage import measure

from reliability import a4_config as C
from reliability.a4_grid import Grid


# ----------------------------------------------------------------------------- §2.5 (1)–(3): mask
def segment_bone(
    hu: np.ndarray,
    roi: np.ndarray,
    spacing_mm,
    threshold: float = C.BONE_THRESHOLD_HU,
    stats: dict | None = None,
) -> np.ndarray:
    """hu (Z,Y,X) in HU; boolean roi on the same grid; spacing_mm = (sx, sy, sz). If `stats` is a dict, it records the
 effect of each cleaning step: voxels added by the closing, components and volume
 removed by the 50 mm3 rule, voxels filled, voxels cut by the ROI."""
    vox_mm3 = float(np.prod(spacing_mm))
    m0 = hu > threshold  # §2.5 threshold 200 HU
    # closing with the replicated border: the scipy default (border_value=0) ERODES by `iterations` voxels the bone that
    # touches a face of the volume (6 mm on the thick grid of the foot, 10 mm in the thorax)
    it = C.CLEANING_CLOSING_ITER
    m1 = ndi.binary_closing(np.pad(m0, it, mode="edge"), iterations=it)[
        it:-it, it:-it, it:-it
    ]
    m2 = _remove_small(m1, C.CLEANING_MIN_MM3, spacing_mm)  # isolated components
    m3 = (
        ndi.binary_fill_holes(m2) if C.CLEANING_FILL_CAVITIES else m2
    )  # cavities: off (True branch = previous reading)
    m = m3 & roi  # target anatomy (ROI of the thick volume)
    if stats is not None:
        _, n1 = ndi.label(m1)
        _, n2 = ndi.label(m2)
        stats.update(
            {
                "threshold_voxels": int(m0.sum()),
                "voxels_added_by_closing": int(m1.sum() - m0.sum()),
                "components_before_50mm3": int(n1),
                "removed_components_50mm3": int(n1 - n2),
                "volume_removed_50mm3_mm3": float((m1.sum() - m2.sum()) * vox_mm3),
                "voxels_filled_in_cavities": int(m3.sum() - m2.sum()),
                "voxels_cut_by_roi": int(m3.sum() - m.sum()),
                "final_voxels": int(m.sum()),
                "fraction_volume_removed_50mm3": float(
                    (m1.sum() - m2.sum()) / max(1, m1.sum())
                ),
            }
        )
    return m


def _remove_small(mask, min_mm3, spacing_mm):
    vox_mm3 = float(np.prod(spacing_mm))
    min_vox = max(1, int(min_mm3 / vox_mm3))
    lbl, n = ndi.label(mask)
    if n == 0:
        return mask
    size = ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
    return np.isin(lbl, np.where(size >= min_vox)[0] + 1)


# ----------------------------------------------------------------------------- §2.5 (4)–(8): mesh
def mask_to_mesh(
    mask: np.ndarray,
    grid: Grid,
    target_faces: int = C.TARGET_FACES,
    taubin_iter: int = C.TAUBIN_ITER,
    gauss_mm: float = C.GAUSS_MM,
) -> trimesh.Trimesh:
    if mask.sum() == 0:
        raise ValueError("empty mask")
    sz, sy, sx = grid.spacing[2], grid.spacing[1], grid.spacing[0]
    pad = C.PADDING_VOXELS
    m = np.pad(
        mask.astype(np.float32), pad
    )  # pad = 0 (no caps); > 0 = previous branch
    if gauss_mm and gauss_mm > 0:
        m = ndi.gaussian_filter(
            m, sigma=(gauss_mm / sz, gauss_mm / sy, gauss_mm / sx)
        )  # §2.5 isotropic Gaussian in mm
    verts_zyx, faces, _, _ = measure.marching_cubes(
        m,
        level=C.MARCHING_CUBES_LEVEL,
        spacing=(1.0, 1.0, 1.0),
        method=C.MARCHING_CUBES_METHOD,
    )  # §2.5 [28]
    idx_xyz = verts_zyx[:, ::-1] - pad  # grid indices, without the padding
    V = grid.index_to_physical(idx_xyz)  # physical coordinates (§2.5)
    mesh = trimesh.Trimesh(
        vertices=V, faces=faces, process=True
    )  # merging of duplicate vertices
    # (precondition of Eq. 10)
    n_comp_before = len(mesh.split(only_watertight=False))
    if C.MESH_MIN_COMPONENT_FACES > 0:  # off (0); the CCA is done in the voxel grid
        mesh = _keep_components(mesh, C.MESH_MIN_COMPONENT_FACES)
    n_comp = len(mesh.split(only_watertight=False))
    faces_before = len(mesh.faces)
    if taubin_iter > 0:
        # §2.5 "12 iterations of Taubin smoothing": in trimesh each `iteration` is a half step (even = λ, odd = ν),
        # so 12 full iterations = 24 half steps
        trimesh.smoothing.filter_taubin(
            mesh, lamb=C.TAUBIN_LAMBDA, nu=C.TAUBIN_NU, iterations=2 * taubin_iter
        )
    if target_faces and len(mesh.faces) > target_faces:
        mesh = _decimate(mesh, target_faces)  # §2.5 Garland–Heckbert to ~60 000
    mesh.remove_unreferenced_vertices()  # hygiene
    mesh.metadata["faces_before_decimation"] = faces_before  # evidence
    mesh.metadata["components"] = {
        "before_filter": int(n_comp_before),
        "after": int(n_comp),
    }
    return mesh


def _keep_components(mesh: trimesh.Trimesh, min_faces: int) -> trimesh.Trimesh:
    comps = mesh.split(only_watertight=False)
    if len(comps) <= 1:
        return mesh
    comps = [c for c in comps if len(c.faces) >= min_faces]
    if not comps:
        return mesh
    return trimesh.util.concatenate(comps)


def _decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """explicit error if the decimation does not happen (no silently returning the intact mesh)."""
    try:
        d = mesh.simplify_quadric_decimation(face_count=target_faces)
    except TypeError:
        d = mesh.simplify_quadric_decimation(target_faces)
    if not isinstance(d, trimesh.Trimesh) or len(d.faces) >= len(mesh.faces):
        raise RuntimeError(
            "quadric decimation did not reduce the mesh (%d -> %s faces); backend missing?"
            % (
                len(mesh.faces),
                None if not isinstance(d, trimesh.Trimesh) else len(d.faces),
            )
        )
    if not (
        (1 - C.DECIMATION_TOLERANCE) * target_faces
        <= len(d.faces)
        <= (1 + C.DECIMATION_TOLERANCE) * target_faces
    ):
        raise RuntimeError(
            "decimation off target: %d faces (target %d)" % (len(d.faces), target_faces)
        )
    return d


# ----------------------------------------------------------------------------- §2.6: per-vertex error
def open_border(mesh: trimesh.Trimesh) -> np.ndarray:
    """Border vertices (edges with a single face) of an open mesh. Evidence
 for Eq. 10/11: at these vertices the angular defect does not close 2π."""
    from trimesh import grouping

    edges = mesh.edges_sorted
    unique = grouping.group_rows(edges, require_count=1)
    b = np.zeros(len(mesh.vertices), bool)
    if len(unique):
        b[np.unique(edges[unique])] = True
    return b


def sample_reference(
    mesh_ref: trimesh.Trimesh, identifier: str, n: int = C.N_REFERENCE_POINTS
) -> np.ndarray:
    """§2.6: "120,000 points were uniformly sampled from the corresponding higher-resolution reference
 surface" (area sampling = uniform on the surface). Seed derived from the identifier."""
    import zlib

    pts, _ = trimesh.sample.sample_surface(
        mesh_ref, n, seed=zlib.crc32(identifier.encode("utf-8"))
    )
    return np.asarray(pts, float)


def vertex_error(V: np.ndarray, ref_points: np.ndarray) -> np.ndarray:
    """Eq. 5: e(v) = min_{p ∈ S_ref} ||v − p||_2, unsigned, in mm (nearest neighbor)."""
    from scipy.spatial import cKDTree

    return cKDTree(ref_points).query(V)[0]


def reference_correspondence(V: np.ndarray, mesh_ref: trimesh.Trimesh) -> np.ndarray:
    """§2.11: correspondences "treated as noiseless" -> EXACT closest point on the reference
 surface (point-to-triangle), used as q(v) in the registration. declared (the text does not define
 how the correspondences are obtained)."""
    from reliability.a4_proximity import nearest_exact

    return nearest_exact(V, mesh_ref)[1]
