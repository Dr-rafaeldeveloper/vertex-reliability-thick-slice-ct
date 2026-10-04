"""Tests of the SR selection, main environment (without HEBO): validation partition, search space
consistent with the published/adopted value, objective aggregation, persistent state (adoption, tie-break, design,
Sobol), resume with a HEBO test double, fidelity of the evaluator arguments to process_case, end-to-end evaluator on
CPU over a synthetic case (determinism), failure rule, validation of the cache hash and result path."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from reliability import a4_config as C  # noqa: E402
from reliability import a4_sr as SR  # noqa: E402
from reliability import a4_surface as S  # noqa: E402
from reliability.a4_grid import Grid  # noqa: E402
from reliability.sr_selection import evaluate_sr as AV  # noqa: E402
from reliability.sr_selection import sr_config as CS  # noqa: E402
from reliability.sr_selection import sr_state as ES  # noqa: E402


def test_space_contains_published_and_a4_config_is_published_or_adopted():
    assert CS.PUBLISHED_SR == (1e-3, 2500)  # §2.4 literal
    assert CS.SR_SEARCH_LR[0] < 1e-3 < CS.SR_SEARCH_LR[1]
    assert CS.SR_SEARCH_ITERS[0] < 2500 < CS.SR_SEARCH_ITERS[1]
    assert CS.SR_SEARCH_EVALUATIONS == 20 and CS.SR_SEARCH_PARTITION == "val"
    assert len(CS.sr_selection_hash()) == 16
    current = (C.SR_LR, C.SR_ITERS)
    if (
        current != CS.PUBLISHED_SR
    ):  # adoption: a4_config must be the recorded adopted configuration
        p = os.path.join(C.A4_RESULTS, "sr_selection_hebo.json")
        a = json.load(open(p, encoding="utf-8"))["summary"]["adopted"]
        assert current == (a["lr"], a["iters"]), (current, a)


def test_validation_cases_are_the_50_of_the_val_partition():
    pairs = [
        ("f%d" % i, "e%d" % i, "CT%03d" % i, "val" if i < 50 else "test")
        for i in range(150)
    ]
    sel = AV.validation_cases(pairs)
    assert len(sel) == 50 and all(p[3] == "val" for p in sel)
    with pytest.raises(AssertionError):
        AV.validation_cases(pairs + [("x", "y", "CT999", "?")])
    if os.path.exists(C.RPLHR_PAIRS):  # real data present: 50 val / 100 test (§2.2.2)
        real = AV.validation_cases()
        assert len(real) == C.N_THORAX_VAL and all("val" in p[0] for p in real)


def test_objective_is_median_of_case_medians():
    pc = {"a": {"e_median": 0.2}, "b": {"e_median": 0.9}, "c": {"e_median": 0.5}}
    assert AV.objective(pc) == 0.5


def test_state_adoption_tiebreak_design_and_sobol(tmp_path):
    p = str(tmp_path / "state.json")
    e = ES.load(p)
    assert e["observations"] == [] and ES.best(e) is None and ES.adopted(e) is None
    e["observations"] += [
        {
            "index": 0,
            "lr": 1e-3,
            "iters": 3000,
            "objective_mm": 0.50,
            "sobol_generated": 1,
        },
        {
            "index": 1,
            "lr": 2e-3,
            "iters": 1000,
            "objective_mm": 0.50,
            "sobol_generated": 2,
        },
        {
            "index": 2,
            "lr": 5e-4,
            "iters": 4000,
            "objective_mm": 0.51,
            "sobol_generated": 4,
        },
    ]
    e["reference"] = {"lr": 1e-3, "iters": 2500, "objective_mm": 0.52}
    ES.save(e, p)
    e2 = ES.load(p)
    assert ES.best(e2)["index"] == 1  # tie at 0.50: fewer iterations wins
    assert (
        ES.sobol_consumed(e2) == 4
    )  # real recorded consumption (the GP phase may use the Sobol)
    r = ES.summary(e2)
    assert abs(r["gain_mm_vs_published"] - 0.02) < 1e-12
    assert r["adopted"]["index"] == 1 and r["adopted_and_published"] is False
    e2["reference"]["objective_mm"] = (
        0.49  # published better than all: it is the adopted one (sr_config.ADOPTION)
    )
    r2 = ES.summary(e2)
    assert r2["adopted"] is e2["reference"] and r2["adopted_and_published"] is True
    assert r2["best_of_20"]["index"] == 1  # the 20 of HEBO continue to be reported
    d = CS.current_design(20, [], "0.3.6")
    e2["design"] = d
    ES.check_design(e2, d)  # equal: passes
    with pytest.raises(RuntimeError, match="design"):
        ES.check_design(e2, CS.current_design(30, [], "0.3.6"))
    with pytest.raises(RuntimeError, match="design"):
        ES.check_design(e2, CS.current_design(20, ["--n-cases", "1"], "0.3.6"))
    assert not os.path.exists(p + ".tmp")


class _Sobol:
    def __init__(self):
        self.num_generated = 0

    def fast_forward(self, n):
        self.num_generated += n


class _OptDouble:
    """HEBO test double: records observations and the Sobol advance (resumes without hebo installed)."""

    rand_sample = 3

    def __init__(self):
        self.sobol = _Sobol()
        self.X, self.y = None, None

    def observe(self, X, y):
        self.X, self.y = X, y


def test_resume_reobserves_effective_values_and_advances_sobol():
    sys.path.insert(0, os.path.dirname(AV.__file__))
    import optimize_sr_hebo as OT

    state = ES.new()
    assert OT.resume(_OptDouble(), state) == 0
    state["observations"] = [
        {"lr": 0.00089, "iters": 3166, "objective_mm": 0.94, "sobol_generated": 1},
        {"lr": 0.00144, "iters": 667, "objective_mm": 1.04, "sobol_generated": 2},
        {"lr": 0.00797, "iters": 4381, "objective_mm": 1.14, "sobol_generated": 3},
        {"lr": 0.00089, "iters": 3236, "objective_mm": 0.93, "sobol_generated": 5},
    ]
    opt = _OptDouble()
    assert OT.resume(opt, state) == 4
    assert opt.sobol.num_generated == 5  # not min(len, rand_sample) = 3
    assert list(opt.X.columns) == ["lr", "iters"] and opt.X.shape == (4, 2)
    assert opt.X["iters"].dtype.kind == "i" and opt.X["lr"].dtype.kind == "f"
    assert opt.y.shape == (4, 1) and opt.y[3, 0] == 0.93
    f = OT.frame_observed(1e-3, 2500.0)
    assert f["iters"].iloc[0] == 2500 and f["iters"].dtype.kind == "i"
    assert OT.A4_OUT == C.A4_OUT  # path rebuilt without importing a4_config


def _synthetic_case(folder, hash_cache=None, bone=True):
    """Thick volume (k = 2) with a 'bone' block (1000 HU) on a 0 background; reference points on the faces of the
block."""
    rng = np.random.default_rng(0)
    Zt, Y, X = 12, 64, 64
    thick = rng.normal(0, 5, size=(Zt, Y, X)).astype(np.float32)
    if bone:
        thick[3:9, 16:48, 16:48] = 1000.0
    spacing = (0.7, 0.7, 1.0)
    k = 2
    g = Grid(
        (spacing[0], spacing[1], spacing[2] * k), (0.0, 0.0, 0.0), None, (Zt, Y, X)
    ).refined_z(k)
    z0 = 3 * k * g.spacing[2] + g.origin[2] - g.spacing[2] / 2
    z1 = 9 * k * g.spacing[2] + g.origin[2] - g.spacing[2] / 2
    y0, y1 = 16 * spacing[1] - spacing[1] / 2, 48 * spacing[1] - spacing[1] / 2
    x0, x1 = 16 * spacing[0] - spacing[0] / 2, 48 * spacing[0] - spacing[0] / 2
    pts = []
    for _ in range(2000):
        f = rng.integers(6)
        p = [rng.uniform(x0, x1), rng.uniform(y0, y1), rng.uniform(z0, z1)]
        p[f // 2] = (x0, x1, y0, y1, z0, z1)[f]
        pts.append(p)
    cp = os.path.join(folder, "synth.npz")
    meta = {
        "ident": "synth",
        "k": k,
        "grid": g.as_dict(),
        "e_tri_median": 0.5,
        "environment": {"code_hash_reliability": hash_cache or C.code_hash()},
        "sr_selection_hash": CS.sr_selection_hash(),
    }
    np.savez_compressed(
        cp,
        thick=thick,
        roi=np.ones(g.shape, bool),
        pts_ref=np.asarray(pts, float),
        meta=json.dumps(meta),
    )
    return cp


def test_evaluate_case_cpu_end_to_end_and_deterministic(tmp_path):
    import torch

    cp = _synthetic_case(str(tmp_path))
    r = AV.evaluate_case(cp, lr=1e-3, iters=3, dev=torch.device("cpu"))
    assert np.isfinite(r["e_median"]) and r["n_vertices"] > 100 and r["failure"] is None
    assert r["n_iterations_run"] == 3 and r["e_tri_median"] == 0.5
    assert 0.0 <= r["e_median"] < 5.0
    r2 = AV.evaluate_case(cp, lr=1e-3, iters=3, dev=torch.device("cpu"))
    assert r2["e_median"] == r["e_median"]  # fixed seed


def test_evaluate_case_passes_same_arguments_as_process_case(tmp_path, monkeypatch):
    """Fidelity: train(slices, k, iters, dev, seed=SR_SEED_EVALUATED, lr=lr, verbose)
 without patch/batch/val_frac; apply(network, tn, k) with default orientations; segment_bone(rec, roi, spacing_f);
 mask_to_mesh(mask, grid)."""
    import torch

    cp = _synthetic_case(str(tmp_path))
    calls = {}
    train0, apply0, seg0, mesh0 = (
        SR.train,
        SR.apply,
        S.segment_bone,
        S.mask_to_mesh,
    )

    def train(*a, **kw):
        calls["train"] = (a[1:], kw)
        return train0(*a, **kw)

    def apply(*a, **kw):
        calls["apply"] = (len(a), kw)
        return apply0(*a, **kw)

    def segment(*a, **kw):
        calls["segment"] = (a[2], kw)
        return seg0(*a, **kw)

    def mesh(*a, **kw):
        calls["mesh"] = (a[1].as_dict(), kw)
        return mesh0(*a, **kw)

    monkeypatch.setattr(SR, "train", train)
    monkeypatch.setattr(SR, "apply", apply)
    monkeypatch.setattr(S, "segment_bone", segment)
    monkeypatch.setattr(S, "mask_to_mesh", mesh)
    AV.evaluate_case(cp, lr=2e-3, iters=2, dev=torch.device("cpu"))
    (k, iters, dev), kw = calls["train"]
    assert (k, iters) == (2, 2) and kw == {
        "seed": C.SR_SEED_EVALUATED,
        "lr": 2e-3,
        "verbose": False,
    }
    assert calls["apply"] == (3, {})  # (network, tn, k), default orientations
    assert tuple(map(float, calls["segment"][0])) == (
        0.7,
        0.7,
        1.0,
    )  # spacing_f of the thin grid (z = 2.0/k)
    assert calls["segment"][1] == {}
    assert (
        list(calls["mesh"][0]["shape_zyx"]) == [24, 64, 64]
        and calls["mesh"][1] == {}
    )


def test_failure_rule_and_cache_hash(tmp_path):
    import torch

    cp = _synthetic_case(
        str(tmp_path), bone=False
    )  # no bone -> empty mask -> failure
    r = AV.evaluate_case_with_rule(cp, lr=1e-3, iters=2, dev=torch.device("cpu"))
    assert r["failure"] and r["e_median"] == 0.5 and r["n_vertices"] == 0
    pc = {"a": r, "b": {"e_median": 0.3, "failure": None}}
    assert AV.objective(pc) == 0.4
    os.makedirs(str(tmp_path / "old"), exist_ok=True)
    cp2 = _synthetic_case(str(tmp_path / "old"), hash_cache="0000000000000000")
    with pytest.raises(RuntimeError, match="hash"):
        AV.load_cache(cp2)


def test_dropsembles_prior_keeps_its_own_rate(monkeypatch):
    """Reading of the post-Stage-B regeneration: the prior (§2.8.2) uses DS_PRIOR_LR = 1e-3, not the selected SR rate."""
    from reliability import a4_uncertainty as I

    seen = {}

    def train(*a, **kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(SR, "train", train)
    with pytest.raises(RuntimeError, match="stop"):
        I.train_prior([np.zeros((8, 8))], 2, None, seed=0, iters=1, verbose=False)
    assert seen["lr"] == C.DS_PRIOR_LR == 1e-3 and C.SR_LR != C.DS_PRIOR_LR


def test_section_36_uses_only_the_100_test_cases():
    from reliability import a4_analyses as A

    T = {"v%d" % i: {"meta": {"partition": "val"}} for i in range(50)}
    T |= {"t%d" % i: {"meta": {"partition": "test"}} for i in range(100)}
    r = A.thorax_for_r36(T)
    assert sorted(r) == sorted("t%d" % i for i in range(100))
    with pytest.raises(RuntimeError, match="expected 100"):
        A.thorax_for_r36({h: d for h, d in T.items() if h != "t0"})
    assert C.R36_PARTITION == "test"
