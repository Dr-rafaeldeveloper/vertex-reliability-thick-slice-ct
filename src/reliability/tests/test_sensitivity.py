"""Variants fixed a priori and swap/restore of the a4_config constants."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from reliability import a4_config as C  # noqa: E402
from reliability.supplement.sensitivity import a4c_sensitivity as SEN  # noqa: E402


def test_variants_contain_base_and_bracket_published_value():
    v = SEN.VARIANTS
    assert v["base"] == {} and len(v) == 15
    assert v["gauss_0.5"]["GAUSS_MM"] < C.GAUSS_MM < v["gauss_1.2"]["GAUSS_MM"]
    assert v["faces_30k"]["TARGET_FACES"] < C.TARGET_FACES < v["faces_120k"]["TARGET_FACES"]
    assert (
        v["closing_1"]["CLEANING_CLOSING_ITER"]
        < C.CLEANING_CLOSING_ITER
        < v["closing_3"]["CLEANING_CLOSING_ITER"]
    )
    assert (
        v["min_25mm3"]["CLEANING_MIN_MM3"]
        < C.CLEANING_MIN_MM3
        < v["min_100mm3"]["CLEANING_MIN_MM3"]
    )
    assert v["roi_ar_-400"]["ROI_AR_HU"] < C.ROI_AR_HU < v["roi_ar_-200"]["ROI_AR_HU"]
    assert (
        v["roi_dilate_1.5"]["ROI_DILATE_MM"]
        < C.ROI_DILATE_MM
        < v["roi_dilate_6"]["ROI_DILATE_MM"]
    )
    assert {
        v["sinterp_closing_0"]["SINTERP_CLOSING"],
        v["sinterp_closing_1"]["SINTERP_CLOSING"],
    } == {0, 1}


def test_override_restores_even_on_exception():
    before = C.CLEANING_CLOSING_ITER
    with SEN.override(CLEANING_CLOSING_ITER=7):
        assert C.CLEANING_CLOSING_ITER == 7
    assert C.CLEANING_CLOSING_ITER == before
    with pytest.raises(RuntimeError):
        with SEN.override(CLEANING_CLOSING_ITER=9):
            raise RuntimeError("x")
    assert C.CLEANING_CLOSING_ITER == before


def test_mask_without_closing_equals_sequence_with_ideal_zero_closing():
    """Without fragmented bone, closing with 1 iteration changes the mask; without closing, the clean threshold
remains."""
    import numpy as np

    thick = np.zeros((6, 20, 20), np.float32)
    thick[1:5, 4:16, 4:16] = 1000.0
    thick[2, 9:11, 9:11] = 0.0  # small internal hole
    roi = np.ones_like(thick, bool)
    m = SEN.thick_mask_without_closing(thick, roi, (0.7, 0.7, 3.0))
    assert m.shape == thick.shape and m.sum() > 0
    assert not m[
        2, 9, 9
    ]  # without closing and without filling cavities, the hole remains


def test_without_closing_equals_segment_bone_when_closing_changes_nothing():
    """in a solid block the closing of 1 iteration does not change the mask; the two routes must coincide."""
    import numpy as np

    from reliability import a4_surface as S

    thick = np.zeros((8, 24, 24), np.float32)
    thick[2:6, 6:18, 6:18] = 1000.0
    roi = np.ones_like(thick, bool)
    spacing = (0.7, 0.7, 3.0)
    with SEN.override(CLEANING_CLOSING_ITER=1):
        ref = S.segment_bone(thick, roi, spacing)
    assert np.array_equal(SEN.thick_mask_without_closing(thick, roi, spacing), ref)
