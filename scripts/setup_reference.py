"""Place the record of the super-resolution hyperparameter selection where the pipeline expects it.

The learning rate and number of iterations used in the paper were selected on the 50 RPLHR-CT validation cases
(src/reliability/sr_selection/). That search retrains the network for every validation case in each of its 21
evaluations, so its record is shipped in reference_results/ and the adopted values are fixed in a4_config.py.
This script copies the record to the results folder, where a4_config and the tests read it. Re-running the search
is optional (see README).

Usage: python scripts/setup_reference.py
"""

from __future__ import annotations

import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from reliability import a4_config as C


def main():
    os.makedirs(C.A4_RESULTS, exist_ok=True)
    dst = os.path.join(C.A4_RESULTS, "sr_selection_hebo.json")
    if os.path.exists(dst):
        print("already present:", dst)
        return
    shutil.copy2(os.path.join(ROOT, "reference_results", "sr_selection_hebo.json"), dst)
    print("copied to", dst)


if __name__ == "__main__":
    main()
