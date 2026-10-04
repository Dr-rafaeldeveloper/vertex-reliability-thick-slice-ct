"""Persistent state of the search (JSON), shared between the optimizer (hebo310) and the evaluator (main).
No dependencies beyond the standard library. Resume: the recorded observations are re-observed by HEBO and the Sobol
engine is advanced up to the recorded consumption (`sobol_generated`), so that the sequence is that of an
uninterrupted run; the recorded design is checked against the current one."""

from __future__ import annotations

import json
import os


def new() -> dict:
    return {
        "observations": [],
        "failures": [],
        "reference": None,
        "best": None,
        "done": False,
    }


def load(filepath: str) -> dict:
    if not os.path.exists(filepath):
        return new()
    with open(filepath, encoding="utf-8") as f:
        e = json.load(f)
    for k, v in new().items():
        e.setdefault(k, v)
    return e


def save(state: dict, filepath: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    tmp = filepath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1, ensure_ascii=False)
    os.replace(tmp, filepath)


def tiebreak_key(o: dict):
    """Smallest objective; tie -> fewer iterations; then smaller rate (sr_config.TIEBREAK)."""
    return (o["objective_mm"], o["iters"], o["lr"])


def best(state: dict) -> dict | None:
    """Best among the HEBO observations (without the published reference)."""
    obs = [o for o in state["observations"] if o.get("objective_mm") is not None]
    return min(obs, key=tiebreak_key) if obs else None


def adopted(state: dict) -> dict | None:
    """sr_config.ADOPTION: smallest objective among the observations AND the published reference, same tie-break."""
    cand = [o for o in state["observations"] if o.get("objective_mm") is not None]
    ref = state.get("reference")
    if ref and ref.get("objective_mm") is not None:
        cand.append(ref)
    return min(cand, key=tiebreak_key) if cand else None


def check_design(state: dict, current: dict) -> None:
    """Resume only with the SAME design; state without a design (first run) is accepted."""
    recorded = state.get("design")
    if not recorded or not state["observations"] and state.get("reference") is None:
        return
    diff = {k: (recorded.get(k), current.get(k)) for k in current if recorded.get(k) != current.get(k)}
    if diff:
        raise RuntimeError("design of the resume differs from the recorded one: %s" % diff)


def sobol_consumed(state: dict) -> int:
    """Sobol points generated up to the last recorded observation (0 if nothing recorded)."""
    obs = state["observations"]
    return int(obs[-1].get("sobol_generated", 0)) if obs else 0


def summary(state: dict) -> dict:
    m = best(state)
    a = adopted(state)
    ref = state.get("reference")
    out = {
        "n_evaluations": len(state["observations"]),
        "n_failures": len(state.get("failures", [])),
        "best_of_20": m,
        "published_reference": ref,
        "adopted": a,
    }
    if m and ref and ref.get("objective_mm") is not None:
        out["gain_mm_vs_published"] = ref["objective_mm"] - m["objective_mm"]
        out["adopted_and_published"] = a is ref
        out["best_equals_published"] = (
            m["lr"] == ref["lr"] and m["iters"] == ref["iters"]
        )
    return out
