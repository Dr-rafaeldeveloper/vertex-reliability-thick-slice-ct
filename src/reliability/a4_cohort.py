"""Provenance of the foot cohort (§2.2.1).

The paper states: "The available foot examinations were screened before inclusion. Duplicate acquisitions and
cases that did not provide a complete or technically usable representation of the target anatomy were
excluded. After this procedure, 48 distinct feet were retained for analysis." and "The retained CT volumes
had an original through-plane slice thickness of 0.5 mm and an in-plane pixel spacing ranging from
approximately 0.88 to 1.37 mm."

This script reconstructs that screening from `metadata.csv` and from the files themselves, and WRITES the
evidence:
 1. duplicates: hash of the voxel-by-voxel CONTENT (md5 of the array + shape), not of the file (the .nii.gz files
 with identical content differ in bytes); crops with the same pixel hash = same acquisition
 ("Duplicate acquisitions"); the first identifier (alphabetical order) of each group is kept.
 The md5 of the file is written only as provenance;
 2. incomplete/unusable representation: (a) entries without a file (read failure of the source NIfTI);
 (b) crops classified in the QC as without the foot (qc_anatomy = WRONG_ANATOMY);
 3. assertions of §2.2.1 on the retained ones: n = 48; sz = 0.5 mm in all (Eq. 2 depends on it);
 sx = sy in all (validity of x9 = Δxy); pixel within [0.88, 1.37] mm.
Outputs: output/validation/reliability/results/foot_cohort_provenance.json and
output/validation/reliability/foot_list_n48.json.
Also checks that the list matches the old list (output/validation/list_n48.json).
Usage: python src/reliability/a4_cohort.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reliability import a4_config as C


def file_md5(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pixel_hash(filepath: str) -> str:
    """md5 of the voxel-by-voxel content (array in C-order, native dtype) + shape."""
    a = sitk.GetArrayFromImage(sitk.ReadImage(filepath))
    h = hashlib.md5()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def header(filepath: str) -> dict:
    rd = sitk.ImageFileReader()
    rd.SetFileName(filepath)
    rd.ReadImageInformation()
    sx, sy, sz = rd.GetSpacing()
    X, Y, Z = rd.GetSize()
    return {
        "sx": float(sx),
        "sy": float(sy),
        "sz": float(sz),
        "dim": [int(X), int(Y), int(Z)],
        "origin": [float(v) for v in rd.GetOrigin()],
        "direction": [float(v) for v in rd.GetDirection()],
    }


def provenance() -> dict:
    rows = list(csv.DictReader(open(C.FOOT_METADATA, encoding="utf-8")))
    cases = []
    for r in rows:
        c = {
            "id": r["id"],
            "file": r["foot_file"] or None,
            "status_csv": r["status"][:60],
            "qc_anatomy_csv": r["qc_anatomy"],
            "dup_of_csv": r["dup_of"] or None,
            "hash_csv": r["foot_hash"] or None,
            "sex": r["meta_sex"],
            "age": r["meta_age_years"],
        }
        if c["file"]:
            p = os.path.join(C.FOOT_FOLDER, c["file"])
            c["file_md5"] = file_md5(p)
            c["pixel_hash"] = pixel_hash(p)
            c.update(header(p))
        cases.append(c)
    # screening universe: subjects listed by the download script (log.txt), z001..z066 minus those absent from the list
    universe = sorted({"z%03d" % i for i in range(1, 67)})
    ids_csv = {c["id"] for c in cases}
    missing_from_csv = [u for u in universe if u not in ids_csv]
    # 2a. no file
    no_file = [c["id"] for c in cases if not c["file"]]
    # 1. duplicates by pixel hash (voxel-by-voxel identical content)
    groups = {}
    for c in cases:
        if c.get("pixel_hash"):
            groups.setdefault(c["pixel_hash"], []).append(c["id"])
    duplicates = {}
    for ids in groups.values():
        ids = sorted(ids)
        for d in ids[1:]:
            duplicates[d] = ids[0]
    # 2b. wrong anatomy in the QC
    wrong_anatomy = [
        c["id"] for c in cases if c["qc_anatomy_csv"].startswith("WRONG_ANATOMY")
    ]
    excluded = set(no_file) | set(duplicates) | set(wrong_anatomy)
    retained = [c for c in cases if c["id"] not in excluded]
    # 3. assertions of §2.2.1
    assert len(retained) == C.N_FEET, "expected %d retained feet, got %d" % (
        C.N_FEET,
        len(retained),
    )
    for c in retained:
        assert abs(c["sz"] - C.SZ_FOOT_MM) < 1e-6, (
            "%s: sz = %.4f != %.2f (§2.2.1 / Eq. 2)" % (c["id"], c["sz"], C.SZ_FOOT_MM)
        )
        assert abs(c["sx"] - c["sy"]) < 1e-6, (
            "%s: sx != sy (x9 = Δxy requires square pixels)" % c["id"]
        )
    pix = [c["sx"] for c in retained]
    assert (
        C.PIXEL_FOOT_MM[0] - 0.01 <= min(pix) and max(pix) <= C.PIXEL_FOOT_MM[1] + 0.01
    ), "pixel outside [%.2f, %.2f]: %.4f–%.4f" % (*C.PIXEL_FOOT_MM, min(pix), max(pix))
    # check against the csv (dup_of) and against the old list
    dup_csv = {c["id"]: c["dup_of_csv"] for c in cases if c["dup_of_csv"]}
    # near-duplicates among retained: pairs with the same dimension and the same pixel -> direct and mirrored correlation
    near = []
    for i in range(len(retained)):
        for j in range(i + 1, len(retained)):
            a, b = retained[i], retained[j]
            if a["dim"] == b["dim"] and abs(a["sx"] - b["sx"]) < 1e-6:
                A = (
                    sitk.GetArrayFromImage(
                        sitk.ReadImage(os.path.join(C.FOOT_FOLDER, a["file"]))
                    )
                    .astype(np.float64)
                    .ravel()
                )
                B3 = sitk.GetArrayFromImage(
                    sitk.ReadImage(os.path.join(C.FOOT_FOLDER, b["file"]))
                ).astype(np.float64)
                rs = {
                    "direct": float(np.corrcoef(A, B3.ravel())[0, 1]),
                    "mirror_x": float(np.corrcoef(A, B3[:, :, ::-1].ravel())[0, 1]),
                    "mirror_y": float(np.corrcoef(A, B3[:, ::-1, :].ravel())[0, 1]),
                }
                near.append(
                    {
                        "pair": [a["id"], b["id"]],
                        "r": rs,
                        "same_acquisition": bool(max(rs.values()) > 0.99),
                    }
                )
    assert not any(q["same_acquisition"] for q in near), (
        "near-duplicate among the retained: %s" % near
    )
    listing = [os.path.join(C.FOOT_FOLDER, c["file"]).replace("\\", "/") for c in retained]
    old = (
        [x.replace("\\", "/") for x in json.load(open(C.FOOT_LIST_N48_OLD))]
        if os.path.exists(C.FOOT_LIST_N48_OLD)
        else None
    )
    flags = [
        {"id": c["id"], "qc_anatomy_csv": c["qc_anatomy_csv"]}
        for c in retained
        if c["qc_anatomy_csv"] not in ("OK", "")
    ]
    return {
        "description": __doc__,
        "metadata_source": C.FOOT_METADATA,
        "folder": C.FOOT_FOLDER,
        "screening_universe": {
            "description": "subjects z001..z066 of the VSDFullBody collection; the download script (foot_ankle_dataset/full/log.txt) listed 63",
            "n_universe": len(universe),
            "missing_from_csv": missing_from_csv,
            "absence_reason": "not found in the local files; to be checked with the author (subject list of the download of 2026-07-08)",
        },
        "n_entries_csv": len(cases),
        "n_with_file": sum(1 for c in cases if c["file"]),
        "excluded": {
            "no_file_or_read_failed": no_file,
            "pixel_hash_duplicates (excluded -> kept)": duplicates,
            "wrong_anatomy_qc": sorted(wrong_anatomy),
        },
        "csv_duplicates_match_pixel_hash": dup_csv == duplicates,
        "pixel_hash_matches_foot_hash_csv": all(
            c.get("pixel_hash") == c["hash_csv"] for c in cases if c.get("pixel_hash")
        ),
        "csv_hash_note": "foot_hash in the CSV was computed by another procedure (script of 2026-07); the GROUPINGS match (csv_duplicates_match_pixel_hash)",
        "n_retained": len(retained),
        "retained": [
            {
                k: c.get(k)
                for k in (
                    "id",
                    "file",
                    "pixel_hash",
                    "file_md5",
                    "sx",
                    "sy",
                    "sz",
                    "dim",
                    "origin",
                    "direction",
                    "qc_anatomy_csv",
                    "sex",
                    "age",
                )
            }
            for c in retained
        ],
        "near_duplicates_checked": near,
        "demographics_warning": "the exact pixel duplicates have diverging sex/age in the CSV (e.g. z013 vs z024); the sex/age fields are NOT reliable to characterize the cohort",
        "assertions_2_2_1": {
            "n_equals_48": True,
            "sz_0_5_mm_in_all": True,
            "sx_equals_sy_in_all": True,
            "pixel_min_mm": min(pix),
            "pixel_max_mm": max(pix),
            "pixel_range_paper_mm": list(C.PIXEL_FOOT_MM),
        },
        "retained_with_non_excluding_qc_flag": flags,
        "list_matches_old_n48_list": (sorted(listing) == sorted(old))
        if old is not None
        else None,
        "environment": C.environment_record(with_torch=False),
    }, listing


def main():
    C.ensure_folders()
    proc, listing = provenance()
    p = os.path.join(C.A4_RESULTS, "foot_cohort_provenance.json")
    json.dump(proc, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(listing, open(C.FOOT_LIST_N48, "w", encoding="utf-8"), indent=1)
    print(
        "csv entries %d | with file %d | without file %s | duplicates %d | wrong anatomy %d | retained %d"
        % (
            proc["n_entries_csv"],
            proc["n_with_file"],
            proc["excluded"]["no_file_or_read_failed"],
            len(proc["excluded"]["pixel_hash_duplicates (excluded -> kept)"]),
            len(proc["excluded"]["wrong_anatomy_qc"]),
            proc["n_retained"],
        )
    )
    print(
        "csv duplicates == pixel hash:",
        proc["csv_duplicates_match_pixel_hash"],
        "| pixel hash == foot_hash of the csv:",
        proc["pixel_hash_matches_foot_hash_csv"],
        "| pixel %.3f–%.3f mm | non-excluding QC flags: %s"
        % (
            proc["assertions_2_2_1"]["pixel_min_mm"],
            proc["assertions_2_2_1"]["pixel_max_mm"],
            proc["retained_with_non_excluding_qc_flag"],
        ),
    )
    print(
        "list matches the old n48 list:",
        proc["list_matches_old_n48_list"],
    )
    print("saved", p, "and", C.FOOT_LIST_N48)


if __name__ == "__main__":
    main()
