"""Input data (§2.2, §2.3): controlled thick-slice generation, trilinear baseline,
real RPLHR-CT pairs, split validation and pair consistency check.

Functions:
 thick_slice(hu, k) §2.3, Eq. 1–2 mean per block of k slices
 interpolate_trilinear_z(vol, k) §2.4 baseline: trilinear resampling to the target grid (and SR input)
 load_rplhr_pair(...) §2.2.2 regrid of the 1 mm onto the 5 mm grid, HU, 0.7 mm pixel, crop in Z
 crop_plane(...) §2.2.2 in-plane crop "to satisfy memory constraints" (same box in both)
 validate_rplhr_split 150 = 50 val + 100 test
 pair_consistency(...) r between the thick slice and the mean of the 5 thin ones (≈ 0.999)
CLI: python src/reliability/a4_data.py --check-rplhr -> results/rplhr_pairs_consistency.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reliability import a4_config as C


# ----------------------------------------------------------------------------- §2.3
def thick_slice(hu: np.ndarray, k: int):
    """Eq. 1: I_thick(x, y, j) = (1/k) Σ_{r=0}^{k-1} I_HR(x, y, k j + r); Eq. 2: Δz_thick = k Δz_HR.
 Returns (hu_truncated, thick). Z is truncated to a multiple of k because Eq. 1
 only defines complete blocks; the truncation also applies to the reference (same volume)."""
    Z = hu.shape[0]
    Zc = (Z // k) * k
    if Zc == 0:
        raise ValueError("volume with fewer than k slices")
    hu_t = np.ascontiguousarray(hu[:Zc])
    thick = (
        hu_t.reshape(Zc // k, k, hu.shape[1], hu.shape[2])
        .astype(np.float64)
        .mean(axis=1)
        .astype(np.float32)
    )
    return hu_t, thick


def interpolate_trilinear_z(vol: np.ndarray, k: int) -> np.ndarray:
    """§2.4: "The original thick-slice volume was resampled to the same target voxel grid... without
 learned intensity correction" — trilinear resampling along z, factor k, voxel centers
 aligned (align_corners=False: the center of thick voxel j falls on thin index j k + (k-1)/2)."""
    import torch

    t = torch.from_numpy(np.ascontiguousarray(vol)[None, None]).float()
    Zt, Y, X = vol.shape
    up = torch.nn.functional.interpolate(
        t, size=(Zt * k, Y, X), mode="trilinear", align_corners=False
    )
    return up[0, 0].numpy()


# ----------------------------------------------------------------------------- §2.2.2
def load_rplhr_pair(
    thin_path: str,
    thick_path: str,
    pixel_mm: float = C.PIXEL_THORAX_MM,
    thin_dz: float = C.DZ_THIN_THORAX_MM,
    thick_dz: float = C.DZ_THICK_THORAX_MM,
    hu_a: float = C.HU_RPLHR_A,
    hu_b: float = C.HU_RPLHR_B,
):
    """§2.2.2: "the 1-mm volume was regridded to the spatial frame of the corresponding 5-mm
 reconstruction"; intensities reverted to HU by the original interval; 0.7 mm pixel assumed;
 "Volumes were cropped when required... while the same spatial extent and transformation were
 preserved for each paired 1-mm and 5-mm reconstruction".
 Returns hu_ref (Zt·k, Y, X), thick_hu (Zt, Y, X), thin_spacing=(pixel, pixel, thick_dz/k), k, info.
 The reference grid has k sub-slices per thick slice, with sub-slice 0 centered at
 -(k-1)/(2k) of the thick index, so that the center of block j coincides with the center of thick
 slice j (same convention as Eq. 1 and interpolate_trilinear_z)."""
    thin = sitk.ReadImage(thin_path)
    thk = sitk.ReadImage(thick_path)
    raw = {
        "thin_min": float(sitk.GetArrayViewFromImage(thin).min()),
        "thin_max": float(sitk.GetArrayViewFromImage(thin).max()),
        "thick_min": float(sitk.GetArrayViewFromImage(thk).min()),
        "thick_max": float(sitk.GetArrayViewFromImage(thk).max()),
        "thin_header_spacing": list(thin.GetSpacing()),
        "thick_header_spacing": list(thk.GetSpacing()),
    }
    thin.SetSpacing((pixel_mm, pixel_mm, thin_dz))
    thk.SetSpacing((pixel_mm, pixel_mm, thick_dz))
    k = int(round(thick_dz / thin_dz))
    X, Y, Zt = thk.GetSize()
    dz = thick_dz / k
    ref = sitk.Image(X, Y, Zt * k, sitk.sitkFloat32)
    ref.SetSpacing((pixel_mm, pixel_mm, dz))
    ref.SetDirection(thk.GetDirection())
    ref.SetOrigin(
        thk.TransformContinuousIndexToPhysicalPoint((0.0, 0.0, -(k - 1) / (2.0 * k)))
    )
    outside_value = (-1024.0 - hu_b) / hu_a  # air, in file units
    thin_r = sitk.Resample(
        sitk.Cast(thin, sitk.sitkFloat32),
        ref,
        sitk.Transform(),
        sitk.sitkLinear,
        float(outside_value),
    )
    hu_ref = hu_a * sitk.GetArrayFromImage(thin_r).astype(np.float32) + hu_b
    thick_hu = hu_a * sitk.GetArrayFromImage(thk).astype(np.float32) + hu_b
    # coverage: sub-slices whose center falls inside the thin volume (crop in Z to complete blocks)
    z_lo = thin.TransformContinuousIndexToPhysicalPoint((0, 0, 0))[2]
    z_hi = thin.TransformContinuousIndexToPhysicalPoint((0, 0, thin.GetSize()[2] - 1))[
        2
    ]
    z_lo, z_hi = min(z_lo, z_hi), max(z_lo, z_hi)
    zs = np.array(
        [ref.TransformIndexToPhysicalPoint((0, 0, int(z)))[2] for z in range(Zt * k)]
    )
    inside = (zs >= z_lo) & (
        zs <= z_hi
    )  # sub-slices with center INSIDE the thin volume (no tolerance)
    zi = np.where(inside)[0]
    z0 = -(-int(zi[0]) // k) * k
    z1 = ((int(zi[-1]) + 1) // k) * k
    hu_ref = np.ascontiguousarray(hu_ref[z0:z1])
    thick_hu = np.ascontiguousarray(thick_hu[z0 // k : z1 // k])
    info = {
        "k": k,
        "thin_dz_mm": thin_dz,
        "thick_dz_mm": thick_dz,
        "pixel_mm": pixel_mm,
        "Z_thick_file": int(Zt),
        "Z_thick_used": int(z1 // k - z0 // k),
        "crop_z_blocks": [int(z0 // k), int(z1 // k)],
        "thin_coverage": float(inside.mean()),
        "coverage_after_crop": float(inside[z0:z1].mean()),
        "hu_conversion": [hu_a, hu_b],
        "xy_size": [int(Y), int(X)],
        "raw_file_values": raw,
    }
    return hu_ref, thick_hu, (pixel_mm, pixel_mm, dz), k, info


def crop_plane(hu_ref: np.ndarray, thick_hu: np.ndarray, box):
    """§2.2.2: in-plane crop, the same box (y0, y1, x0, x1) in both volumes."""
    y0, y1, x0, x1 = box
    return np.ascontiguousarray(hu_ref[:, y0:y1, x0:x1]), np.ascontiguousarray(
        thick_hu[:, y0:y1, x0:x1]
    )


def rplhr_pairs():
    pairs = json.load(open(C.RPLHR_PAIRS, encoding="utf-8"))

    def partition(a):
        a2 = a.replace("\\", "/")
        return "val" if "/val/" in a2 else ("test" if "/test/" in a2 else "?")

    return [(thin, thk, name, partition(thin)) for thin, thk, name in pairs]


def validate_rplhr_split() -> dict:
    """§2.2.2: "150 thoracic cases... comprising 50 validation cases and 100 test cases"."""
    pairs = rplhr_pairs()
    val = [p[2] for p in pairs if p[3] == "val"]
    test = [p[2] for p in pairs if p[3] == "test"]
    ok = (
        len(pairs) == C.N_THORAX
        and len(val) == C.N_THORAX_VAL
        and len(test) == C.N_THORAX_TEST
        and len({p[2] for p in pairs}) == C.N_THORAX
        and all(os.path.exists(p[0]) and os.path.exists(p[1]) for p in pairs)
    )
    return {
        "n_pairs": len(pairs),
        "n_val": len(val),
        "n_test": len(test),
        "unique_names": len({p[2] for p in pairs}),
        "files_exist": all(
            os.path.exists(p[0]) and os.path.exists(p[1]) for p in pairs
        ),
        "val": val,
        "test": test,
        "compliant_2_2_2": bool(ok),
    }


def pair_consistency(pairs=None, verbose=True) -> dict:
    """§2.2.2: "Consistency of the paired volumes was verified by comparing each thick slice
 with the mean of the five corresponding thin slices, producing an intensity correlation of
 approximately r = 0.999". After the regrid of load_rplhr_pair, the k sub-slices of block j are the
 "five corresponding thin slices" of thick slice j. Pearson r per slice; per case: mean and
 minimum; in the cohort: median and minimum across cases."""
    pairs = pairs or rplhr_pairs()
    per_case = {}
    t0 = time.time()
    for i, (thin, thk, name, part) in enumerate(pairs, 1):
        hu_ref, thick_hu, sp, k, info = load_rplhr_pair(thin, thk)
        mean5 = hu_ref.reshape(
            thick_hu.shape[0], k, hu_ref.shape[1], hu_ref.shape[2]
        ).mean(axis=1)
        rs = []
        discarded = 0
        for j in range(thick_hu.shape[0]):
            a = thick_hu[j].ravel().astype(np.float64)
            b = mean5[j].ravel().astype(np.float64)
            if a.std() > 0 and b.std() > 0:
                rs.append(float(np.corrcoef(a, b)[0, 1]))
            else:
                discarded += 1  # r undefined on a constant slice (declared)
        rs = np.array(rs)
        per_case[name] = {
            "partition": part,
            "k": k,
            "n_slices_total": int(thick_hu.shape[0]),
            "n_slices_evaluated": len(rs),
            "n_constant_slices_discarded": int(discarded),
            "r_mean": float(rs.mean()),
            "r_median": float(np.median(rs)),
            "r_min": float(rs.min()),
            "thin_coverage": info["thin_coverage"],
            "raw_file_values": info["raw_file_values"],
        }
        if verbose and (i % 10 == 0 or i == len(pairs)):
            print(
                "  [%d/%d] %s r_mean %.4f r_min %.4f (%.0fs)"
                % (i, len(pairs), name, rs.mean(), rs.min(), time.time() - t0),
                flush=True,
            )
    med = np.array([v["r_mean"] for v in per_case.values()])
    br = [v["raw_file_values"] for v in per_case.values()]
    return {
        "n_cases": len(per_case),
        "r_mean_per_case_median": float(np.median(med)),
        "r_mean_per_case_min": float(med.min()),
        "discarded_slices_total": int(
            sum(v["n_constant_slices_discarded"] for v in per_case.values())
        ),
        "slices_total": int(sum(v["n_slices_total"] for v in per_case.values())),
        "observed_normalized_range": {
            "thin_min": min(b["thin_min"] for b in br),
            "thin_max": max(b["thin_max"] for b in br),
            "thick_min": min(b["thick_min"] for b in br),
            "thick_max": max(b["thick_max"] for b in br),
            "note": "files in [0,1]; HU = %g*v + %g (§2.2.2, original interval [-1024, 2048] HU)"
            % (C.HU_RPLHR_A, C.HU_RPLHR_B),
        },
        "r_mean_per_case_max": float(med.max()),
        "r_min_absolute": float(min(v["r_min"] for v in per_case.values())),
        "r_expected_paper": C.R_PAIRS_EXPECTED,
        "per_case": per_case,
    }


def window_hu_rplhr(n_cases: int = 12, seed: int = 0) -> dict:
    """Internal evidence for the HU window (§2.2.2 "original intensity interval"): soft-tissue mode in the
 RAW [0,1] values of the 5 mm volume (FOV core, range 0.2-0.5). With HU = a*v - 1024: a = 3072 (window
 [-1024, 2048]) puts soft tissue (~50 HU) at v = 0.350; a = 4095 (window [-1024, 3071]) at v = 0.262."""
    pairs = rplhr_pairs()
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(pairs), min(n_cases, len(pairs)), replace=False)
    modes = {}
    for i in sel:
        v = sitk.GetArrayFromImage(sitk.ReadImage(pairs[i][1])).astype(np.float32)
        core = v[:, 100:400, 100:400]
        h, _ = np.histogram(
            core[(core > 0.2) & (core < 0.5)], bins=np.linspace(0.2, 0.5, 151)
        )
        modes[pairs[i][2]] = float(0.2 + (np.argmax(h) + 0.5) * 0.002)
    med = float(np.median(list(modes.values())))
    hyp = {
        "[-1024, 2048]": {
            "a": 3072.0,
            "v_soft_tissue_50HU": 1074 / 3072,
            "mode_HU": 3072 * med - 1024,
        },
        "[-1024, 3071]": {
            "a": 4095.0,
            "v_soft_tissue_50HU": 1074 / 4095,
            "mode_HU": 4095 * med - 1024,
        },
    }
    return {
        "n_cases": len(modes),
        "soft_tissue_mode_v": modes,
        "median_mode_v": med,
        "assumptions": hyp,
        "conclusion": "window [-1024, 2048]: the mode falls at %.0f HU (soft tissue); the alternative would give %.0f HU (impossible)"
        % (hyp["[-1024, 2048]"]["mode_HU"], hyp["[-1024, 3071]"]["mode_HU"]),
        "adopted": [C.HU_RPLHR_B, C.HU_RPLHR_B + C.HU_RPLHR_A],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check-rplhr",
        action="store_true",
        help="results/rplhr_pairs_consistency.json",
    )
    ap.add_argument(
        "--window-hu",
        action="store_true",
        help="internal evidence of the HU window -> results/window_hu_rplhr.json",
    )
    ap.add_argument(
        "--n-cases", type=int, default=0, help="0 = all (only for a quick test)"
    )
    args = ap.parse_args()
    if args.window_hu:
        C.ensure_folders()
        j = window_hu_rplhr()
        j["environment"] = C.environment_record(with_torch=False)
        p = os.path.join(C.A4_RESULTS, "window_hu_rplhr.json")
        json.dump(j, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(j["conclusion"], "| saved", p)
    if args.check_rplhr:
        C.ensure_folders()
        split = validate_rplhr_split()
        print(
            "split: n=%d val=%d test=%d unique=%d exist=%s compliant=%s"
            % (
                split["n_pairs"],
                split["n_val"],
                split["n_test"],
                split["unique_names"],
                split["files_exist"],
                split["compliant_2_2_2"],
            ),
            flush=True,
        )
        pairs = rplhr_pairs()
        if args.n_cases > 0:
            pairs = pairs[: args.n_cases]
        cons = pair_consistency(pairs)
        print(
            "consistency: median r_mean %.4f | min %.4f | max %.4f | absolute r_min %.4f (expected approx. %.3f)"
            % (
                cons["r_mean_per_case_median"],
                cons["r_mean_per_case_min"],
                cons["r_mean_per_case_max"],
                cons["r_min_absolute"],
                C.R_PAIRS_EXPECTED,
            )
        )
        out = {
            "description": __doc__,
            "split_2_2_2": split,
            "consistency_2_2_2": cons,
            "environment": C.environment_record(with_torch=False),
        }
        p = os.path.join(C.A4_RESULTS, "rplhr_pairs_consistency.json")
        json.dump(out, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("saved", p)


if __name__ == "__main__":
    main()
