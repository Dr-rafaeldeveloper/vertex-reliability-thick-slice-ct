# Data

Two public datasets are used. Neither was acquired by the authors.

## Foot CT (primary cohort)

Crops of the foot from the **VSDFullBody** collection of post-mortem whole-body CT examinations
(M. Kistler, *VSDFullBody: The Virtual Skeleton Database Full Body CT Collection*, Zenodo, 2013,
<https://doi.org/10.5281/zenodo.8270365>), licence **CC BY-NC 3.0** (attribution, non-commercial).

| File | Content |
|---|---|
| `foot/metadata.csv` | One row per screened examination: identifier, screening flags, quality-control label, file name and MD5 of the voxel content. The age, sex, height and weight columns come from the public VSD metadata. |
| `foot/SHA256SUMS` | SHA-256 of each of the 62 volumes. |
| `foot/foot/*.nii.gz` | The 62 volumes (0.5-mm slices). **Not in git**: download with `python scripts/download_foot_data.py`. |
| `sample/z002_foot.nii.gz` | One of the volumes, kept in git for `scripts/quick_check.py`. |

`src/reliability/a4_cohort.py` reproduces the screening described in the paper (duplicates by voxel content,
examinations without the target anatomy) and keeps 48 feet. It writes the list used by every later step.

## Thoracic CT (real paired thick and thin slices)

**RPLHR-CT** (P. Yu et al., *RPLHR-CT dataset and transformer baseline for volumetric super-resolution from CT
scans*, MICCAI 2022): 5-mm and 1-mm reconstructions of the same acquisition. The volumes are **not redistributed
here**; request them from the dataset authors (<https://github.com/smilenaxx/RPLHR-CT>).

| File | Content |
|---|---|
| `rplhr/cases.json` | The 150 case identifiers used and their partition: 50 validation cases (selection of the super-resolution hyperparameters) and 100 test cases (evaluation). |
| `rplhr/pairs_val_test.json` | Local paths of the pairs. **Not in git**: created by `python scripts/prepare_rplhr.py --root <RPLHR-CT folder>`. |

The data folder can be placed elsewhere by setting the environment variable `A4_DATA_DIR`.
