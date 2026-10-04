"""Selection of the learning rate and the number of SR iterations (§2.4) by Bayesian optimization
(HEBO) on the RPLHR-CT validation partition (50 cases), criterion = median surface error against the
1 mm reference.

Subfolder OUTSIDE the package hash (`a4_config.code_hash` covers only `src/reliability/*.py`): the selection does not
change any result by itself; only the adoption of the selected values in `a4_config` (SR_LR, SR_ITERS) changes the
hash and forces regenerating caches and analyses. Each output records the package hash AND the hash of this subfolder
(`sr_config.sr_selection_hash`).

Two environments: `evaluate_sr.py` runs in the main environment (torch + GPU); `optimize_sr_hebo.py` runs in the
conda environment `hebo310` (HEBO 0.3.6 requires numpy < 1.25 and does not install with Python 3.13) and calls the
evaluator by subprocess. `sr_state.py` and `sr_config.py` do not import the package and serve both environments."""
