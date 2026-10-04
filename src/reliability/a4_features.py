"""The nine per-vertex features (§2.7.1, Eq. 7–14, Table 1) and the shape-disagreement
descriptor (§2.7.2, Eq. 15–18).

x(v) = [x1... x9] (Eq. 7), in the order:
 x1 = |n_z(v)| Eq. 8 component of the unit normal along the acquisition axis
 x2 = d_f(v) = min_m |z_v − z_m| Eq. 9 distance to the center of the nearest acquired slice (mm)
 x3 = 2π − Σ_f θ_f(v) Eq. 10 SIGNED angular defect (rad)
 x4 = ||v − L^(3)(v)||_2, λ = 0.5 Eq. 11 roughness: 3 pure Laplacian smoothings (no volume constraint)
 x5 = |n_z(v)| · d_f(v) Eq. 12
 x6 = d_shape(v) = d(v, S_interp) Eq. 13/18 EXACT distance to the interpolated surface
 x7 = Δz, x8 = t_slice, x9 = Δxy Eq. 14 case constants
S_interp (§2.7.2): φ = signed SDF of the bone mask of the THICK volume (Eq. 15, negative inside),
interpolated along z by k (Eq. 16, trilinear), surface = zero level of φ̃ (Eq. 17) extracted
by marching cubes on the field itself, without Gaussian, Taubin or decimation; d_shape = min over the
surface (Eq. 18), computed exactly (point-to-triangle).
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage as ndi
from skimage import measure

from reliability import a4_config as C
from reliability import (
    a4_surface as S,
)  # only segment_bone; a4_surface does not import this module
from reliability.a4_data import interpolate_trilinear_z
from reliability.a4_grid import Grid
from reliability.a4_proximity import nearest_exact, without_degenerate


# ----------------------------------------------------------------------------- §2.7.2
def sdf_mm(mask: np.ndarray, spacing_mm) -> np.ndarray:
    """Eq. 15: φ(x) = −d(x, Γ) inside Ω, +d(x, Γ) outside; Euclidean distance in mm (anisotropic
 sampling). Also used in Eq. 31 (surface uncertainty)."""
    sx, sy, sz = spacing_mm
    outside = ndi.distance_transform_edt(~mask, sampling=(sz, sy, sx))
    inside = ndi.distance_transform_edt(mask, sampling=(sz, sy, sx))
    return (outside - inside).astype(np.float32)


def interpolated_surface(
    thick_mask: np.ndarray, k: int, thin_grid: Grid
) -> trimesh.Trimesh:
    """Eq. 15–17: S_interp = { x: φ̃(x) = 0 }, φ̃ = U_k[φ]. No post-processing."""
    thick_spacing = (thin_grid.spacing[0], thin_grid.spacing[1], thin_grid.spacing[2] * k)
    # no cap (SINTERP_CAP = False), consistent with the thin meshes without padding; the True branch (1 thick voxel
    # of background on each face) is the previous reading
    pad = (
        1 if C.SINTERP_CAP else 0
    )  # no cap (consistent with PADDING_VOXELS = 0)
    phi = sdf_mm(
        np.pad(thick_mask, pad, mode="constant", constant_values=False), thick_spacing
    )  # Eq. 15 (thick grid)
    phi_t = interpolate_trilinear_z(phi, k)  # Eq. 16 (trilinear along z)
    verts_zyx, faces, _, _ = measure.marching_cubes(
        phi_t, level=0.0, spacing=(1.0, 1.0, 1.0), method=C.MARCHING_CUBES_METHOD
    )  # Eq. 17
    verts_zyx = verts_zyx - np.array(
        [pad * k, pad, pad]
    )  # removes the padding offset
    V = thin_grid.index_to_physical(verts_zyx[:, ::-1])
    return without_degenerate(
        trimesh.Trimesh(vertices=V, faces=faces, process=True)
    )  # zero-area faces are not surface


def thick_mask_sinterp(
    thick: np.ndarray,
    thick_roi: np.ndarray,
    thick_spacing,
    mode: str | None = None,
    stats: dict | None = None,
) -> np.ndarray:
    """§2.7.2 "the bone mask obtained from the original thick-slice volume" (Eq. 15), on the thick grid.
 mode "cleaning": the mask of §2.5 (threshold 200 HU, closing, < 50 mm3, soft-tissue ROI;
 no cavities, per CLEANING_FILL_CAVITIES) = a4_surface.segment_bone on the thick volume;
 mode "threshold": pure threshold, the same comparator as segment_bone. Only place in the code
 that decides the mask of S_interp (a4_run_foot, a4_run_thorax and the figures call here).
 Declared consequence: the closing is counted in voxels of the grid on which the mask is made
 (CLEANING_CLOSING_ITER = 2), so on the thick grid it reaches 2·k·Δz in z (foot: 6 mm; thorax: 10 mm) versus 1 mm on
 the
 thin grid — "the same thresholding and post-processing sequence" (§2.5), the text does not give the scale in mm.
 `stats`
 (dict) receives the per-step cleaning statistics, as in the other masks (meta["cleaning"]["s_interp"])."""
    mode = C.SINTERP_MASK if mode is None else mode
    if mode == "threshold":
        return thick > C.BONE_THRESHOLD_HU
    if mode == "cleaning":
        return S.segment_bone(thick, thick_roi, thick_spacing, stats=stats)
    raise ValueError("unknown SINTERP_MASK: %r" % (mode,))


def exact_distance(P: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    """Eq. 18: d_shape(v) = min_{q ∈ S_interp} ||v − q||_2, EXACT (point-to-triangle; a4_proximity)."""
    return nearest_exact(P, mesh)[0]


# ----------------------------------------------------------------------------- §2.7.1
def d_f_mm(V: np.ndarray, thin_grid: Grid, k: int) -> np.ndarray:
    """Eq. 9: distance to the center of the nearest ACQUIRED slice. The centers of the thick slices
 j are at fine index j·k + (k−1)/2 (Eq. 1 and the refined grid); the distance is measured along
 the acquisition axis in fine index and converted to mm by Δz (independent of origin/direction)."""
    zi = thin_grid.z_index(V)
    t = zi / k - (k - 1) / (2.0 * k)
    return np.abs(t - np.round(t)) * k * float(thin_grid.spacing[2])


def roughness(mesh: trimesh.Trimesh) -> np.ndarray:
    """Eq. 11: ||v − L^(3)(v)||_2 with λ = 0.5 — three pure Laplacian smoothings (volume_constraint=False; Eq. 11 does
not describe a volume constraint)."""
    m = mesh.copy()
    trimesh.smoothing.filter_laplacian(
        m,
        lamb=C.ROUGHNESS_LAMBDA,
        iterations=C.ROUGHNESS_ITER,
        implicit_time_integration=False,
        volume_constraint=False,
    )
    return np.linalg.norm(np.asarray(mesh.vertices) - np.asarray(m.vertices), axis=1)


def features(
    mesh: trimesh.Trimesh, thin_grid: Grid, k: int, s_interp: trimesh.Trimesh
):
    """Returns F (n,9) in the order of Eq. 7–14."""
    V = np.asarray(mesh.vertices, float)
    N = np.asarray(mesh.vertex_normals, float)
    z_axis = thin_grid.D[:, 2] / np.linalg.norm(
        thin_grid.D[:, 2]
    )  # acquisition axis in physical coordinates
    nz = np.abs(N @ z_axis)  # Eq. 8
    df = d_f_mm(V, thin_grid, k)  # Eq. 9
    defect = np.asarray(mesh.vertex_defects, float)  # Eq. 10 (signed)
    rug = roughness(mesh)  # Eq. 11
    inter = nz * df  # Eq. 12
    dshape = exact_distance(V, s_interp)  # Eq. 13/18
    dz = float(thin_grid.spacing[2])
    t_slice = k * dz
    dxy = float(thin_grid.spacing[0])  # Eq. 14
    one = np.ones(len(V))
    F = np.column_stack(
        [nz, df, defect, rug, inter, dshape, dz * one, t_slice * one, dxy * one]
    ).astype(np.float32)
    assert F.shape[1] == C.N_FEATURES
    return F
