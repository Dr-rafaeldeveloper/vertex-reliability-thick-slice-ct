"""Build the list of RPLHR-CT pairs used in the paper from a local copy of the dataset.

RPLHR-CT is not redistributed here. Obtain it from its authors (see data/README.md) and point this script to the
folder that contains `val/` and `test/`, each with `1mm/` and `5mm/` sub-folders:

    <root>/val/1mm/CT00000100.nii.gz    <root>/val/5mm/CT00000100.nii.gz
    <root>/test/1mm/CT00000150.nii.gz   <root>/test/5mm/CT00000150.nii.gz

The 150 cases and their partition (50 validation, 100 test) are listed in data/rplhr/cases.json.

Usage: python scripts/prepare_rplhr.py --root /path/to/RPLHR-CT
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER = os.path.join(ROOT, "data", "rplhr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    with open(os.path.join(FOLDER, "cases.json"), encoding="utf-8") as f:
        cases = json.load(f)
    pairs, missing = [], []
    for c in cases:
        thin = os.path.join(root, c["partition"], "1mm", c["case"] + ".nii.gz")
        thk = os.path.join(root, c["partition"], "5mm", c["case"] + ".nii.gz")
        for p in (thin, thk):
            if not os.path.exists(p):
                missing.append(p)
        pairs.append([thin, thk, c["case"]])
    if missing:
        print("%d files not found, e.g. %s" % (len(missing), missing[0]))
        return 1
    with open(os.path.join(FOLDER, "pairs_val_test.json"), "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=1)
    print("%d pairs written to data/rplhr/pairs_val_test.json" % len(pairs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
