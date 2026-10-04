"""Voxel grid and physical coordinates (§2.5: "All reconstructed and reference surfaces were
expressed in physical coordinates").

Convention: numpy arrays in (z, y, x) order; continuous indices (i_x, i_y, i_z); physical point
p = origin + D · (index ⊙ spacing), with D the 3×3 direction matrix of the exam (SimpleITK). All the
surfaces of a case (reference, SR, trilinear, S_interp) use the SAME thin grid, so that
distances between them are invariant to origin/direction; d_f (Eq. 9) is computed along the
acquisition axis z in fine INDEX and converted to mm by Δz, which makes it independent of the origin.
"""

from __future__ import annotations

import numpy as np


class Grid:
    def __init__(self, spacing, origin=(0.0, 0.0, 0.0), direction=None, shape=None):
        """spacing = (sx, sy, sz) mm; origin = (ox, oy, oz) mm; direction = 9 values (row by row) or None
 (identity); shape = (Z, Y, X) of the array."""
        self.spacing = np.asarray(spacing, float)
        self.origin = np.asarray(origin, float)
        self.D = (
            np.eye(3) if direction is None else np.asarray(direction, float).reshape(3, 3)
        )
        self.Dinv = np.linalg.inv(self.D)
        self.shape = None if shape is None else tuple(int(v) for v in shape)

    @classmethod
    def from_sitk(cls, img, shape=None):
        return cls(
            img.GetSpacing(),
            img.GetOrigin(),
            img.GetDirection(),
            shape if shape is not None else tuple(reversed(img.GetSize())),
        )

    def refined_z(self, k: int) -> Grid:
        """Thin grid of a thick volume (Eq. 1/Eq. 2 inverted): z spacing / k and origin shifted
 by -(k-1)/(2k) of the thick index, so that the center of block j (fine index j k + (k-1)/2)
 coincides with the center of thick slice j. It is the grid of the 0.5 mm volume in the foot (by construction) and
 the regridded reference grid in the thorax (a4_data.load_rplhr_pair)."""
        spacing = self.spacing.copy()
        spacing[2] /= k
        origin = self.index_to_physical(np.array([[0.0, 0.0, -(k - 1) / (2.0 * k)]]))[
            0
        ]
        shape = (
            None
            if self.shape is None
            else (self.shape[0] * k, self.shape[1], self.shape[2])
        )
        return Grid(spacing, origin, self.D.reshape(-1), shape)

    def index_to_physical(self, idx_xyz: np.ndarray) -> np.ndarray:
        """idx (n,3) in (i_x, i_y, i_z) -> points (n,3) in mm."""
        return self.origin + (np.asarray(idx_xyz, float) * self.spacing) @ self.D.T

    def physical_to_index(self, p_xyz: np.ndarray) -> np.ndarray:
        """points (n,3) mm -> continuous indices (n,3) in (i_x, i_y, i_z)."""
        return ((np.asarray(p_xyz, float) - self.origin) @ self.Dinv.T) / self.spacing

    def sample(self, vol: np.ndarray, p_xyz: np.ndarray, order: int = 1) -> np.ndarray:
        """Interpolation (order 1 = trilinear, Eq. 27 and Eq. 31) of a (z,y,x) volume at the physical
 points p_xyz. Outside the volume: border value (mode='nearest')."""
        from scipy import ndimage as ndi

        idx = self.physical_to_index(p_xyz)
        coords = np.vstack([idx[:, 2], idx[:, 1], idx[:, 0]])
        return ndi.map_coordinates(
            np.asarray(vol, np.float32), coords, order=order, mode="nearest"
        )

    def z_index(self, p_xyz: np.ndarray) -> np.ndarray:
        """Continuous coordinate along the acquisition axis (fine index in z)."""
        return self.physical_to_index(p_xyz)[:, 2]

    def as_dict(self) -> dict:
        return {
            "spacing_mm": self.spacing.tolist(),
            "origin_mm": self.origin.tolist(),
            "direction": self.D.reshape(-1).tolist(),
            "shape_zyx": self.shape,
        }
