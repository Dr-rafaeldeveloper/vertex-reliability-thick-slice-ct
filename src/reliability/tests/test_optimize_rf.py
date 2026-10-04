"""Tests of the nested selection of the forest: deterministic inner partition, conversion of the leaf
fraction, absence of leakage of the held-out case, determinism per seed (and variation with the fold index), reuse of
the configuration in the ablation, budget per cohort, fallback in a degenerate case, memo key sensitive to the design
and validation of the selection JSON by the key."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from reliability import a4_analyses as A  # noqa: E402
from reliability import a4_config as C  # noqa: E402
from reliability import a4_optimize_rf as O  # noqa: E402


def _data(n_cases=6, n=300, seed=0):
    rng = np.random.default_rng(seed)
    F, e = {}, {}
    for i in range(n_cases):
        X = rng.normal(size=(n, C.N_FEATURES)).astype(np.float32)
        F["c%d" % i] = X
        e["c%d" % i] = (0.5 + 0.8 * X[:, 5] + 0.1 * rng.normal(size=n)).astype(
            np.float32
        )
    return F, e


def _small(monkeypatch):
    monkeypatch.setattr(C, "RF_SEARCH_N_VERTICES", 100)
    monkeypatch.setattr(C, "N_TRAIN_PER_CASE", 150)
    monkeypatch.setattr(C, "RF_SEARCH_STARTUP", 2)


def test_inner_folds_deterministic_disjoint_and_complete(monkeypatch):
    ids = ["c%d" % i for i in range(7)]
    d = O.inner_folds(ids, k=3)
    assert d == [["c0", "c3", "c6"], ["c1", "c4"], ["c2", "c5"]]
    assert sorted(sum(d, [])) == sorted(ids)
    assert O.inner_folds(list(reversed(ids)), k=3) == d  # independent of the order
    monkeypatch.setattr(C, "RF_SEARCH_FOLDS", 2)
    assert (
        len(O.inner_folds(ids)) == 2
    )  # default resolved at call time, not at import


def test_forest_params_leaf_fraction_becomes_count():
    p = {"n_estimators": 150, "max_depth": 6, "leaf_frac": 2.66e-4, "max_features": 3}
    m = O.forest_params(p, 188000)
    assert m.min_samples_leaf == 50 and m.n_estimators == 150 and m.max_depth == 6
    assert (
        O.forest_params(p, 47000).min_samples_leaf == 13
    )  # same fraction, smaller base
    assert O.forest_params({**p, "leaf_frac": 1e-9}, 100).min_samples_leaf == 1
    fx = O.fixed_params_as_fraction(188000)
    assert (
        O.forest_params(fx, 188000).min_samples_leaf
        == C.RF_PARAMS["min_samples_leaf"]
    )


def test_budget_per_cohort():
    assert O.n_trials_for("thorax_sr") == C.RF_SEARCH_TRIALS_THORAX
    assert O.n_trials_for("thorax_tri") == C.RF_SEARCH_TRIALS_THORAX
    assert O.n_trials_for("rf9_sr") == C.RF_SEARCH_TRIALS_FOOT
    assert O.n_trials_for("without5") == C.RF_SEARCH_TRIALS_FOOT


def test_selection_without_leakage_deterministic_and_seed_per_fold(monkeypatch):
    """The search of fold h only sees the other cases: changing e[h] does not change the chosen configuration; same seed,
 same configuration; different fold indices give different trajectories; TPE and random search work."""
    _small(monkeypatch)
    F, e = _data()
    cols = list(range(9))
    p1, t1, _, s1, v1 = O.select_fold(F, e, "c0", cols, n_trials=4, fold_index=0)
    p2, t2, _, s2, _ = O.select_fold(F, e, "c0", cols, n_trials=4, fold_index=0)
    assert p1 == p2 and s1 == s2 == C.RF_SEED
    assert [t["objective"] for t in t1] == [t["objective"] for t in t2]
    assert v1 is not None and t1[v1]["params"] == p1
    e2 = dict(e)
    e2["c0"] = np.zeros_like(e["c0"])  # the held-out one changes; the search must not notice
    p3, _, _, _, _ = O.select_fold(F, e2, "c0", cols, n_trials=4, fold_index=0)
    assert p3 == p1
    _, t5, _, s5, _ = O.select_fold(F, e, "c0", cols, n_trials=4, fold_index=5)
    assert s5 == C.RF_SEED + 5
    assert [t["params"] for t in t5] != [t["params"] for t in t1]  # seed per fold
    p4, _, _, _, _ = O.select_fold(
        F, e, "c0", cols, n_trials=4, fold_index=0, sampler="random"
    )
    assert set(p4) == {"n_estimators", "max_depth", "leaf_frac", "max_features"}
    assert len(t1) == 4 and all(-1 <= t["objective"] <= 1 for t in t1)


def test_fallback_when_no_attempt_is_finite(monkeypatch):
    """e constant in all training cases -> rho undefined in all attempts -> configuration of the text."""
    _small(monkeypatch)
    F, e = _data(n_cases=4)
    for h in F:
        e[h] = np.full_like(e[h], 1.0)
    p, t, _, _, winner = O.select_fold(
        F, e, "c0", list(range(9)), n_trials=3, fold_index=0
    )
    assert winner is None and all(not np.isfinite(x["objective"]) for x in t)
    assert (
        p["n_estimators"] == C.RF_PARAMS["n_estimators"]
        and p["max_depth"] == C.RF_PARAMS["max_depth"]
    )


def test_nested_lofo_details_and_reuse_in_ablation(monkeypatch):
    _small(monkeypatch)
    F, e = _data(n_cases=4)
    pred, params, trials, det = O.nested_lofo(F, e, n_trials=3, verbose=False)
    assert set(pred) == set(F) and all(pred[h][0].shape == (300,) for h in F)
    assert all(len(trials[h]) == 3 for h in F)
    for h in F:
        d = det[h]
        assert d["seed"] == C.RF_SEED + sorted(F).index(h)
        assert (
            d["min_samples_leaf_refit"] >= 1 and d["mean_leaves_per_tree"] >= 1
        )
        assert -1 <= d["external_rho"] <= 1 and d["fallback"] is False
    pred_iso, params_iso, trials_iso, det_iso = O.nested_lofo(
        F, e, columns=[5], fixed_params=params, verbose=False
    )
    assert all(trials_iso[h] == [] and det_iso[h]["seed"] is None for h in F)
    assert all(
        params_iso[h]["max_features"] == 1 for h in F
    )  # cropped to the number of columns
    assert all(
        params_iso[h]["n_estimators"] == params[h]["n_estimators"]
        and params_iso[h]["leaf_frac"] == params[h]["leaf_frac"]
        for h in F
    )


def test_memo_key_changes_with_design_and_data(monkeypatch):
    F, e = _data(n_cases=3, n=50)
    monkeypatch.setattr(C, "RF_SELECTION", "nested")
    k0 = A.memo_key(F, e, None, "rf9_sr")
    assert k0 == A.memo_key(F, e, None, "rf9_sr")  # stable
    monkeypatch.setattr(C, "RF_SEARCH_FOLDS", C.RF_SEARCH_FOLDS + 1)
    assert A.memo_key(F, e, None, "rf9_sr") != k0
    monkeypatch.undo()
    monkeypatch.setattr(C, "RF_SELECTION", "nested")
    monkeypatch.setattr(C, "RF_SEARCH_SAMPLER", "random")
    assert A.memo_key(F, e, None, "rf9_sr") != k0
    monkeypatch.setattr(C, "RF_SEARCH_SAMPLER", "tpe")
    assert A.memo_key(F, e, None, "thorax_sr") != k0  # different budget
    assert (
        A.memo_key(F, e, None, "rf9_sr", "rf9_sr", "abc") != k0
    )  # ablation with params
    e2 = dict(e)
    e2["c0"] = e["c0"] + 1
    assert A.memo_key(F, e2, None, "rf9_sr") != k0  # content of the data
    monkeypatch.setattr(C, "RF_SELECTION", "fixed")
    assert A.memo_key(F, e, None, "rf9_sr") != k0  # fixed vs nested design


def test_selected_params_requires_key_of_current_design(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "A4_RESULTS", str(tmp_path))
    monkeypatch.setattr(C, "RF_SELECTION", "nested")
    A._SELECTION.clear()
    with pytest.raises(RuntimeError, match="not found"):
        A._selected_params("rf9_sr", "abc123")
    p = tmp_path / ("rf_selection_rf9_sr_%s.json" % C.RF_SEARCH_SAMPLER)
    p.write_text(
        json.dumps({"key": "other", "params_per_case": {"c0": {}}}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="key"):
        A._selected_params("rf9_sr", "abc123")
    p.write_text(
        json.dumps({"key": "abc123", "params_per_case": {"c0": {"x": 1}}}),
        encoding="utf-8",
    )
    assert A._selected_params("rf9_sr", "abc123") == {"c0": {"x": 1}}
    A._SELECTION.clear()
