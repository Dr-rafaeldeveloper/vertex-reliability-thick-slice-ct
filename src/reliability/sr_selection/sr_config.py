"""Constants of the SR selection. Without importing the reliability package (also used in the hebo310
environment).

Text decides: HEBO; 20 evaluations; RPLHR-CT validation partition (50 cases);
criterion = median surface error against the 1 mm reference; values applied without new tuning to the 48 feet and
to the 100 test cases.
Where the text is silent, declared assumptions: bounds of the space; aggregation
across cases; seeds (Sobol; GP/ES per evaluation; network); HEBO defaults; published configuration evaluated as a
reference point outside the 20; tie-break; adoption rule; rule for an evaluation/case that fails."""

from __future__ import annotations

import hashlib
import os

# LITERAL value of §2.4 ("initial learning rate of 1e-3", "2500 iterations"): reference point; does not change when
# the selected one is adopted in a4_config (the test compares a4_config with the published one OR with the adopted
# one recorded)
PUBLISHED_SR = (1e-3, 2500)
SR_SEARCH_LR = (
    1e-4,
    1e-2,
)  # log-uniform, one decade on each side of the published value (HEBO 'pow', base 10)
SR_SEARCH_ITERS = (
    500,
    5000,
)  # integer; 2 500 at the center; ceiling = 2x the published (cost: ~2x per case)
SR_SEARCH_EVALUATIONS = 20  # text
SR_SEARCH_SEED = 0  # scramble_seed of the initial Sobol of HEBO
SR_SEARCH_GP_SEED = "SR_SEARCH_SEED + index of the evaluation, in torch.manual_seed and np.random.seed before each suggest"
SR_SEARCH_PARTITION = "val"  # text: RPLHR-CT validation partition (50 cases)
SR_SEARCH_AGGREGATION = "median across cases of the per-case median of e(v)"  # (same aggregation as r31/§2.12)
HEBO_DEFAULTS = {  # HEBO 0.3.6 defaults (GP model, MACE acquisition, 1 + n_parameters = 3 Sobol points)
    "model_name": "gp",
    "acq": "MACE",
    "rand_sample": 3,
    "es": "nsga2",
}
HEBO_VERSION = "0.3.6"
HEBO_ENVIRONMENT = "hebo310"
TIEBREAK = "lowest objective; tie -> fewer iterations; then lower rate"
ADOPTION = (  # the published one competes in the adoption, but does not enter HEBO (the 20 suggestions stay intact)
    "the configuration with the lowest objective among the 20 evaluated AND the published one is adopted, with the same tie-break"
)
FAILURE_RULE = (  #, fixed a priori: failure = 'not better than the baseline'
    "a failing case (diverging SR, empty mask, invalid mesh) receives e_median = e_tri_median of the same case; "
    "an evaluation whose subprocess fails entirely receives objective = trilinear objective of the reference evaluation"
)
CACHE_VAL_MB_PER_CASE = (
    16  # measured in the smoke test (CT00000100: 16.3 MB) -> ~0.8 GB over the 50 cases
)


def sr_selection_hash() -> str:
    """Concatenated SHA-256 of the .py files of this subfolder (traceability; the subfolder is outside the package
hash)."""
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for name in sorted(os.listdir(here)):
        if name.endswith(".py"):
            h.update(name.encode())
            with open(os.path.join(here, name), "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:16]


def current_design(n_evaluations: int, extra: list, hebo_version: str) -> dict:
    """Design recorded in the state and checked on resume."""
    return {
        "lr": list(SR_SEARCH_LR),
        "iters": list(SR_SEARCH_ITERS),
        "n_evaluations": int(n_evaluations),
        "sobol_seed": SR_SEARCH_SEED,
        "gp_seed": SR_SEARCH_GP_SEED,
        "partition": SR_SEARCH_PARTITION,
        "aggregation": SR_SEARCH_AGGREGATION,
        "hebo": dict(HEBO_DEFAULTS, version=hebo_version),
        "tiebreak": TIEBREAK,
        "adoption": ADOPTION,
        "failure_rule": FAILURE_RULE,
        "extra": list(extra),
        "sr_selection_hash": sr_selection_hash(),
    }
