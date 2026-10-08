#!/bin/bash
# Full reproduction, in the order used for the paper. Run from the repository root, after the data steps of the README.
# Each step writes to output/validation/reliability/ and skips cases whose cache already exists, so the script can be
# stopped and restarted. One heavy process at a time.
set -eu
export PYTHONIOENCODING=utf-8
P=src/reliability
L=output/validation/reliability/logs
mkdir -p "$L"

python scripts/setup_reference.py
python $P/a4_cohort.py                                         > "$L/cohort.log" 2>&1           # foot screening: 62 -> 48 feet
python $P/a4_run_foot.py                                       > "$L/run_foot.log" 2>&1         # 48 feet: SR, surfaces, features, uncertainty
python $P/a4_run_thorax.py --partition test                    > "$L/run_thorax.log" 2>&1       # 100 thoracic test cases
python $P/a4_data.py --check-rplhr --window-hu                 > "$L/data.log" 2>&1             # consistency of the thoracic pairs
python $P/a4_analyses.py --only 31,32,33,341,342,35,37         > "$L/analyses_foot.log" 2>&1    # foot results (Sections 3.1-3.5, 3.7)
python $P/a4_analyses.py --only 36                             > "$L/analyses_thorax.log" 2>&1  # thoracic results (Section 3.6)
python $P/supplement/sensitivity/a4c_sensitivity.py --process  > "$L/sens_process.log" 2>&1
python $P/supplement/sensitivity/a4c_sensitivity.py --analyze  > "$L/sens_analyze.log" 2>&1
python $P/supplement/exact_error.py                            > "$L/exact_error.log" 2>&1   # no-op unless ERROR_DEFINITION = "sampled"
python $P/supplement/error_tails.py                            > "$L/error_tails.log" 2>&1   # mean, P90, fraction > 2 mm (Section 3.1)
python $P/supplement/median_constant_baseline.py               > "$L/median_baseline.log" 2>&1
python $P/supplement/calibration_and_descriptor_checks.py      > "$L/calibration_checks.log" 2>&1
python $P/supplement/ewc_calibration_check.py                  > "$L/ewc_calibration.log" 2>&1
# The exact-error caches of the paper were rebuilt from the sampled run (supplement/rebuild_caches_exact.py); a fresh run
# of a4_run_foot.py produces the same target (distance to the stored nearest point for the super-resolved mesh, bounded
# search for the trilinear mesh and for the thorax).
python scripts/compare_with_reference.py
