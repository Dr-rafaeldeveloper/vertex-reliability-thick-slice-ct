"""Download the foot CT volumes used in the paper and verify them.

The 62 screened foot volumes (48 retained after screening) are distributed as a release asset of this repository.
They are crops of the public VSDFullBody collection (CC BY-NC 3.0; see data/README.md).

Usage: python scripts/download_foot_data.py [--zip path/to/foot_ct_volumes.zip]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESTINATION = os.path.join(ROOT, "data", "foot")
URL = "https://github.com/Dr-rafaeldeveloper/vertex-reliability-thick-slice-ct/releases/download/v1.0.0/foot_ct_volumes.zip"


def sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def matches():
    failures = 0
    with open(os.path.join(DESTINATION, "SHA256SUMS"), encoding="utf-8") as f:
        rows = [l.split() for l in f if l.strip()]
    for sum, rel in rows:
        p = os.path.join(DESTINATION, rel)
        if not os.path.exists(p):
            print("missing:", rel)
            failures += 1
        elif sha256(p) != sum:
            print("checksum mismatch:", rel)
            failures += 1
    print("%d files checked, %d problems" % (len(rows), failures))
    return failures == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--zip",
        default="",
        help="use a zip already downloaded instead of fetching the release asset",
    )
    ap.add_argument("--only-verify", action="store_true")
    args = ap.parse_args()
    if not args.only_verify:
        z = args.zip
        if not z:
            z = os.path.join(ROOT, "data", "foot_ct_volumes.zip")
            if not os.path.exists(z):
                print("downloading", URL)
                urllib.request.urlretrieve(URL, z)
        with zipfile.ZipFile(z) as fname:
            fname.extractall(DESTINATION)  # the archive contains the folder foot/
    return 0 if matches() else 1


if __name__ == "__main__":
    sys.exit(main())
