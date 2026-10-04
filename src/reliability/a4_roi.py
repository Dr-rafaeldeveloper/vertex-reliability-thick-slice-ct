"""Soft-tissue ROI derived from the THICK VOLUME (§2.3 and §2.5).

§2.3: "The resulting 3-mm volume constituted the only image input used by the subsequent through-plane
reconstruction and surface-generation pipeline. The original 0.5-mm volume was retained exclusively for
construction of the higher-resolution reference surface and for calculation of geometric reconstruction
error". §2.5: "Connected-component analysis was then applied to retain the anatomically relevant bone
structure while excluding disconnected objects unrelated to the target anatomy. The same thresholding and
post-processing sequence was applied to the higher-resolution reference volumes and to all reconstructed
volumes".

(declared): the "target anatomy" is delimited by a soft-tissue ROI = largest connected component
of voxels > -300 HU of the THICK volume, after a binary opening (1 it.); the ROI is taken to the thin grid by
repeating each thick slice k times (nearest neighbor, no interpolation) and dilated by a sphere of
3 mm (isotropic in mm), so as not to cut the bone at the border. The SAME ROI is used for the reference, the
trilinear,
the SR and the ensemble members. S_interp uses the mask of §2.5 on the thick grid, WITH this ROI. Nothing of the
thin volume enters here.
`roi_on_thick_grid` returns the thin ROI reduced to the thick grid (block = any marked sub-slice); it is the ROI
that the
thick mask of S_interp uses (a4_features.thick_mask_sinterp).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from reliability import a4_config as C


def thick_roi(
    thick: np.ndarray,
    k: int,
    thin_spacing_mm=None,
    ar_hu: float = C.ROI_AR_HU,
    dilate_mm: float = C.ROI_DILATE_MM,
) -> np.ndarray:
    """thick (Zt, Y, X) in HU -> boolean ROI on the thin grid (Zt·k, Y, X). thin_spacing_mm = (sx, sy, sz) of the thin
 grid, for the isotropic dilation in mm."""
    body = thick > ar_hu
    body = ndi.binary_opening(body, iterations=C.ROI_OPENING_ITER)
    lbl, n = ndi.label(body)
    if n > 1:
        size = ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
        body = lbl == (int(np.argmax(size)) + 1)
    thin = np.repeat(body, k, axis=0)  # each thick slice covers k sub-slices
    if dilate_mm > 0 and thin_spacing_mm is not None:
        sx, sy, sz = thin_spacing_mm
        rz, ry, rx = [max(1, int(round(dilate_mm / p))) for p in (sz, sy, sx)]
        zz, yy, xx = np.ogrid[-rz : rz + 1, -ry : ry + 1, -rx : rx + 1]
        elem = (zz * sz) ** 2 + (yy * sy) ** 2 + (
            xx * sx
        ) ** 2 <= dilate_mm**2  # ellipsoid = sphere of dilate_mm in mm
        thin = ndi.binary_dilation(thin, structure=elem)
    return thin


def roi_on_thick_grid(thin_roi: np.ndarray, k: int) -> np.ndarray:
    """Thin ROI (Zt·k, Y, X) -> thick ROI (Zt, Y, X): block marked if any of the k sub-slices is."""
    Zf = thin_roi.shape[0]
    return thin_roi.reshape(Zf // k, k, thin_roi.shape[1], thin_roi.shape[2]).any(
        axis=1
    )


def plane_box(thick: np.ndarray, margin: int = 8, ar_hu: float = C.ROI_AR_HU):
    """§2.2.2 (thorax): box (y0, y1, x0, x1) of the body in the thick volume, with margin, for the in-plane
 crop "to satisfy memory constraints". Computed only on the thick one; applied to both volumes."""
    body = thick > ar_hu
    body = ndi.binary_opening(body, iterations=1)
    lbl, n = ndi.label(body)
    if n > 1:
        size = ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
        body = lbl == (int(np.argmax(size)) + 1)
    body = ndi.binary_dilation(body, iterations=2)
    ys, xs = np.where(body.any(axis=0))
    Y, X = thick.shape[1], thick.shape[2]
    return (
        max(0, int(ys.min()) - margin),
        min(Y, int(ys.max()) + 1 + margin),
        max(0, int(xs.min()) - margin),
        min(X, int(xs.max()) + 1 + margin),
    )
