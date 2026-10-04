"""NESTED selection of the forest hyperparameters (§2.7.3) inside the
leave-one-case-out (§2.7.4, Eq. 20). For each held-out case h:
 1. the training cases (all but h) are divided into C.RF_SEARCH_FOLDS inner folds by case, deterministically
 (order of the identifiers, circular distribution i mod k);
 2. each attempt of the optimizer proposes a configuration in the C.RF_SEARCH space; it is evaluated by fitting the
 forest on the
 inner training folds with C.RF_SEARCH_N_VERTICES vertices per case (seed of the identifier) and measuring, on the
 cases
 of the inner validation fold, the Spearman rho between prediction and e(v) on the N_TRAIN_PER_CASE subsampled
 vertices
 of those cases (seed of the identifier, the same in all outer folds — declared); objective = median of those
 rho over all the inner validation cases (the primary metric, §2.9.1);
 3. the configuration with the highest objective is refit with N_TRAIN_PER_CASE vertices per case on ALL the
 training cases and
 predicts all vertices of h (Eq. 20). min_samples_leaf is a FRACTION of the training samples, converted to a count
 at each fit (floor 1), so that the configuration has the same meaning in the search and in the refit; max_features is
 clipped to the number of columns of the model (ablation).
Optimizer: Optuna TPE (Bergstra et al. 2011; Akiba et al. 2019), C.RF_SEARCH_STARTUP initial random attempts,
seed = C.RF_SEED + index of the outer fold; or random search (Bergstra & Bengio 2012) with the same budget, as a
control for sensitivity to the optimizer (C.RF_SEARCH_SAMPLER = "random"). No vertex of the held-out case enters the
search
or the refit (Cawley & Talbot 2010). Each attempt is written (parameters, objective). If no attempt produces a
finite objective (degenerate case), the fold uses C.RF_PARAMS converted to a fraction, with a record ("fallback").
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reliability import a4_config as C
from reliability import a4_statistics as E

optuna.logging.set_verbosity(optuna.logging.WARNING)


def n_trials_for(label: str) -> int:
    """Budget per cohort: labels 'thorax*' use RF_SEARCH_TRIALS_THORAX, the others RF_SEARCH_TRIALS_FOOT."""
    return (
        C.RF_SEARCH_TRIALS_THORAX if label.startswith("thorax") else C.RF_SEARCH_TRIALS_FOOT
    )


def inner_folds(training_ids, k: int | None = None):
    """Deterministic partition of the training cases into k folds by case: the i-th (sorted) identifier goes to the
 fold i mod k. Returns a list of k lists of identifiers (inner validation of each fold)."""
    k = C.RF_SEARCH_FOLDS if k is None else k
    ids = sorted(training_ids)
    return [ids[i::k] for i in range(k)]


def fixed_params_as_fraction(n_samples: int) -> dict:
    """C.RF_PARAMS (text: 200/8/50/3) expressed in the search space (leaf as a fraction)."""
    p = C.RF_PARAMS
    return {
        "n_estimators": int(p["n_estimators"]),
        "max_depth": int(p["max_depth"]),
        "leaf_frac": float(p["min_samples_leaf"]) / float(n_samples),
        "max_features": int(p["max_features"]),
    }


def forest_params(params: dict, n_samples: int) -> RandomForestRegressor:
    """Instantiates the forest from a configuration of the RF_SEARCH space (leaf_frac -> count, floor 1)."""
    leaf = max(1, int(round(params["leaf_frac"] * n_samples)))
    return RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_samples_leaf=leaf,
        max_features=int(params["max_features"]),
        n_jobs=-1,
        random_state=C.RF_SEED,
    )


def _sample(trial, p_max: int) -> dict:
    lo, hi, spacing = C.RF_SEARCH["n_estimators"]
    return {
        "n_estimators": trial.suggest_int("n_estimators", lo, hi, step=spacing),
        "max_depth": trial.suggest_int("max_depth", *C.RF_SEARCH["max_depth"]),
        "leaf_frac": trial.suggest_float(
            "leaf_frac", *C.RF_SEARCH["leaf_frac"], log=True
        ),
        "max_features": trial.suggest_int(
            "max_features",
            C.RF_SEARCH["max_features"][0],
            min(C.RF_SEARCH["max_features"][1], p_max),
        ),
    }


def inner_objective(
    params, F, e, training_ids, cols, sub_search, sub_val, folds
) -> float:
    """Median of the Spearman rho per inner validation case, over the inner folds."""
    rhos = []
    for val in folds:
        excl = set(val)
        tr = [j for j in training_ids if j not in excl]
        X = np.vstack([F[j][sub_search[j]][:, cols] for j in tr])
        y = np.concatenate([e[j][sub_search[j]] for j in tr])
        mdl = forest_params(params, len(y)).fit(X, y)
        for v in val:
            pred = mdl.predict(F[v][sub_val[v]][:, cols])
            rhos.append(E.spearman(pred, e[v][sub_val[v]]))
    return float(np.nanmedian(rhos)) if np.isfinite(rhos).any() else float("nan")


def select_fold(F, e, h, cols, n_trials, fold_index, sampler=None):
    """Nested search for the held-out case h. Returns (best configuration, list of attempts, seconds, seed,
 index of the winning attempt or None if fallback)."""
    t0 = time.time()
    sampler = C.RF_SEARCH_SAMPLER if sampler is None else sampler
    training_ids = [j for j in sorted(F) if j != h]
    folds = inner_folds(training_ids)
    sub_search = {
        j: E.training_subsample(len(e[j]), j, C.RF_SEARCH_N_VERTICES) for j in training_ids
    }
    sub_val = {
        j: E.training_subsample(len(e[j]), j, C.N_TRAIN_PER_CASE) for j in training_ids
    }
    seed = C.RF_SEED + fold_index
    if sampler == "tpe":
        optuna_sampler = optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=C.RF_SEARCH_STARTUP
        )
    elif sampler == "random":
        optuna_sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        raise ValueError("unknown RF_SEARCH_SAMPLER: %r" % (sampler,))
    study = optuna.create_study(direction="maximize", sampler=optuna_sampler)
    attempts = []

    def target(trial):
        params = _sample(trial, len(cols))
        val = inner_objective(
            params, F, e, training_ids, cols, sub_search, sub_val, folds
        )
        attempts.append({"number": trial.number, "params": params, "objective": val})
        return val

    study.optimize(target, n_trials=n_trials, catch=(ValueError,))
    finite = [t for t in attempts if np.isfinite(t["objective"])]
    if finite:
        winner = max(finite, key=lambda t: t["objective"])
        best, winning = dict(winner["params"]), int(winner["number"])
    else:  # degenerate case: no attempt with a finite objective
        n_search = sum(len(sub_search[j]) for j in training_ids)
        best, winning = fixed_params_as_fraction(n_search), None
    return best, attempts, round(time.time() - t0, 1), seed, winning


def nested_lofo(
    F, e, columns=None, n_trials=None, sampler=None, fixed_params=None, verbose=True
):
    """LOFO (Eq. 20) with nested selection per outer fold. If fixed_params (dict {id: params}) is given, there is no
search:
 each fold uses the given configuration (ablation, §2.10, C.RF_ABLATION_REUSE_SELECTION). Returns
 ({id: (pred, const)}, {id: params}, {id: [attempts]}, {id: details}) — details: seconds, seed, winning
 attempt (None = fallback), min_samples_leaf as a count in the refit, mean number of leaves per tree, outer rho."""
    ids = sorted(F)
    cols = list(range(C.N_FEATURES)) if columns is None else list(columns)
    n_trials = C.RF_SEARCH_TRIALS_FOOT if n_trials is None else n_trials
    sub = {i: E.training_subsample(len(e[i]), i, C.N_TRAIN_PER_CASE) for i in ids}
    pred, params_out, trials_out, det = {}, {}, {}, {}
    for k, h in enumerate(ids):
        if fixed_params is not None:
            params = dict(fixed_params[h])
            params["max_features"] = min(int(params["max_features"]), len(cols))
            attempts, s, seed, winner = [], 0.0, None, None
        else:
            params, attempts, s, seed, winner = select_fold(
                F, e, h, cols, n_trials, k, sampler
            )
        tr = [j for j in ids if j != h]
        X = np.vstack([F[j][sub[j]][:, cols] for j in tr])
        y = np.concatenate([e[j][sub[j]] for j in tr])
        mdl = forest_params(params, len(y)).fit(X, y)
        const = E.constant_prediction(e, tr, y)  # §2.9.4
        p = mdl.predict(F[h][:, cols])
        pred[h] = (p, const)  # Eq. 20
        params_out[h], trials_out[h] = params, attempts
        det[h] = {
            "search_seconds": s,
            "seed": seed,
            "winning_attempt": winner,
            "fallback": fixed_params is None and winner is None,
            "min_samples_leaf_refit": int(mdl.min_samples_leaf),
            "n_samples_refit": int(len(y)),
            "mean_leaves_per_tree": float(
                np.mean([t.get_n_leaves() for t in mdl.estimators_])
            ),
            "external_rho": E.spearman(p, e[h]),
        }
        if verbose:
            print(
                "  [%d/%d] %s | %s | %d attempts | rho ext %.3f | %.0fs"
                % (
                    k + 1,
                    len(ids),
                    h,
                    {
                        a: (round(b, 5) if isinstance(b, float) else b)
                        for a, b in params.items()
                    },
                    len(attempts),
                    det[h]["external_rho"],
                    s,
                ),
                flush=True,
            )
    return pred, params_out, trials_out, det


def sampler_control(label="rf9_sr"):
    """Control for sensitivity to the optimizer: repeats the nested selection of the full foot model with RANDOM
 search and the same budget, and compares it with the TPE selection recorded in rf_selection_<label>_tpe.json (LOFO
 memo key included): per-fold configurations and outer rho per case. Writes
 results/rf_sampler_control_<label>.json."""
    import glob
    import json

    F, e = {}, {}
    for f in sorted(glob.glob(os.path.join(C.A4_CACHE_FOOT, "*.npz"))):
        z = np.load(f, allow_pickle=False)
        h = os.path.basename(f)[:-4]
        F[h], e[h] = z["F"], z["e"].astype(np.float32)
    p_tpe = os.path.join(C.A4_RESULTS, "rf_selection_%s_tpe.json" % label)
    tpe = json.load(open(p_tpe, encoding="utf-8"))
    from reliability.a4_analyses import memo_key

    assert tpe["key"] == memo_key(F, e, None, label), (
        "rf_selection_%s_tpe.json does not correspond to the current caches" % label
    )
    z = np.load(
        os.path.join(C.A4_RESULTS, "_lofo_%s_%s.npz" % (label, tpe["key"])),
        allow_pickle=False,
    )
    rho_tpe = {h: E.spearman(z["pred_" + h], e[h]) for h in F}
    pred, params, attempts_rand, det = nested_lofo(F, e, sampler="random")
    rho_ale = {h: det[h]["external_rho"] for h in F}
    out = {
        "description": sampler_control.__doc__,
        "label": label,
        "key_tpe": tpe["key"],
        "tpe": {
            "params_per_case": tpe["params_per_case"],
            "external_rho_per_case": rho_tpe,
        },
        "random": {
            "params_per_case": params,
            "attempts_per_case": attempts_rand,
            "details_per_case": det,
            "external_rho_per_case": rho_ale,
        },
        "external_rho_tpe": E.summary(list(rho_tpe.values())),
        "external_rho_random": E.summary(list(rho_ale.values())),
        "paired_comparison_tpe_minus_random": E.paired_comparison(
            rho_tpe, rho_ale
        ),
        "environment": C.environment_record(with_torch=False),
    }
    p = os.path.join(C.A4_RESULTS, "rf_sampler_control_%s.json" % label)
    json.dump(
        out, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str
    )
    print("sampler control saved:", p)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler-control", action="store_true")
    ap.add_argument("--label", default="rf9_sr")
    a = ap.parse_args()
    if a.sampler_control:
        sampler_control(a.label)
