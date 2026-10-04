"""Optimizer of the SR selection (conda environment `hebo310`; HEBO 0.3.6). Proposes (lr, iters) with HEBO (GP + MACE; 3
initial Sobol points; scramble_seed = SR_SEARCH_SEED; torch/np seeded with SR_SEARCH_SEED + i before each suggest),
calls the evaluator of the main environment by subprocess and observes the objective with the values ACTUALLY
evaluated. State persisted after each evaluation (resumable: re-observes, advances the Sobol up to the recorded
consumption, checks the design). Smoke test (`--extra`) requires its own `--state` (does not contaminate the real
run).

Before the 20 evaluations, the published configuration (1e-3, 2500) is evaluated as a REFERENCE point (it does not
enter HEBO; it competes only in the ADOPTION — sr_config.ADOPTION). A subprocess that fails entirely receives
objective = trilinear objective of the reference (sr_config.FAILURE_RULE) and is observed, so that the search does
not repeat the point. At the end (run without `--extra`), writes `results/sr_selection_hebo.json`.

Usage (in the hebo310 environment):
 conda run -n hebo310 --no-capture-output python optimize_sr_hebo.py --python <main python> [--n 20]
 [--state <json>] [--extra "--n-cases 2"]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sr_config as CS  # noqa: E402
import sr_state as ES  # noqa: E402

EVALUATE = os.path.join(HERE, "evaluate_sr.py")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # repository root
A4_OUT = os.path.join(
    ROOT, "output", "validation", "reliability"
)  # = a4_config.A4_OUT (test checks)
FOLDER = os.path.join(A4_OUT, "_sr_selection")
STATE = os.path.join(FOLDER, "hebo_state.json")
RESULT = os.path.join(A4_OUT, "results", "sr_selection_hebo.json")


def space():
    from hebo.design_space.design_space import DesignSpace

    return DesignSpace().parse(
        [
            {
                "name": "lr",
                "type": "pow",
                "lb": CS.SR_SEARCH_LR[0],
                "ub": CS.SR_SEARCH_LR[1],
                "base": 10,
            },
            {
                "name": "iters",
                "type": "int",
                "lb": CS.SR_SEARCH_ITERS[0],
                "ub": CS.SR_SEARCH_ITERS[1],
            },
        ]
    )


def new_optimizer():
    from hebo.optimizers.hebo import HEBO

    opt = HEBO(space(), scramble_seed=CS.SR_SEARCH_SEED)
    assert opt.rand_sample == CS.HEBO_DEFAULTS["rand_sample"], opt.rand_sample
    return opt


def seed(i: int):
    """sr_config.SR_SEARCH_GP_SEED: GP (torch) and NSGA-II/select_id (numpy) deterministic per evaluation."""
    import torch

    torch.manual_seed(CS.SR_SEARCH_SEED + i)
    np.random.seed(CS.SR_SEARCH_SEED + i)


def frame_observed(lr: float, iters: int) -> pd.DataFrame:
    """Observes exactly what was evaluated (float, int), not the raw frame of the suggest."""
    return pd.DataFrame([{"lr": float(lr), "iters": int(iters)}])


def resume(opt, state):
    """Re-observes the recorded evaluations and advances the Sobol up to the recorded consumption: sequence identical to
 that of an uninterrupted run, also when the GP phase fell back to the Sobol because of a duplicate."""
    obs = state["observations"]
    n_sobol = ES.sobol_consumed(state)
    if n_sobol:
        opt.sobol.fast_forward(n_sobol)
    if obs:
        X = pd.concat(
            [frame_observed(o["lr"], o["iters"]) for o in obs], ignore_index=True
        )
        y = np.array([[o["objective_mm"]] for o in obs], float)
        opt.observe(X, y)
    return len(obs)


def evaluate_external(python, lr, iters, output, extra):
    cmd = [
        python,
        EVALUATE,
        "--lr",
        repr(float(lr)),
        "--iters",
        str(int(iters)),
        "--output",
        output,
    ] + extra
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    with open(output, encoding="utf-8") as f:
        return json.load(f)


def registration(r, index, phase, output):
    return {
        "index": index,
        "phase": phase,
        "lr": r["lr"],
        "iters": r["iters"],
        "objective_mm": r["objective_mm"],
        "tri_objective_mm": r["aggregated"]["tri_objective_mm"],
        "sr_better_than_tri_in": r["aggregated"]["sr_better_than_tri_in"],
        "n_cases": r["aggregated"]["n_cases"],
        "n_failures": r["aggregated"]["n_failures"],
        "cases": r["cases"],
        "seconds": r["seconds"],
        "file": os.path.relpath(output, os.path.dirname(os.path.dirname(output))),
        "code_hash_reliability": r["environment"]["code_hash_reliability"],
        "sr_selection_hash": r["sr_selection_hash"],
        "hashes_cache": r["hashes_cache"],
    }


def hebo_environment():
    import hebo
    import torch

    return {
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", CS.HEBO_ENVIRONMENT),
        "python": sys.version.split()[0],
        "hebo": getattr(hebo, "__version__", CS.HEBO_VERSION),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--python", required=True, help="python of the main environment (torch + GPU)"
    )
    ap.add_argument("--n", type=int, default=CS.SR_SEARCH_EVALUATIONS)
    ap.add_argument("--state", default=STATE)
    ap.add_argument(
        "--extra", default="", help='extra arguments to the evaluator, e.g. "--n-cases 2"'
    )
    args = ap.parse_args()
    extra = args.extra.split() if args.extra else []
    if extra:
        assert os.path.abspath(args.state) != os.path.abspath(STATE), (
            "smoke test (--extra) requires its own --state"
        )
    env = hebo_environment()
    design = CS.current_design(args.n, extra, env["hebo"])
    eval_folder = os.path.join(
        os.path.dirname(os.path.abspath(args.state)), "evaluations"
    )
    os.makedirs(eval_folder, exist_ok=True)
    state = ES.load(args.state)
    ES.check_design(state, design)
    cases_without = state.get("design", {}).get("cases")
    state["design"] = design
    if cases_without:
        state["design"]["cases"] = cases_without
        state["design"]["sr_seed"] = state["reference"]["sr_seed"]
    state["hebo_environment"] = env
    ES.save(state, args.state)
    t0 = time.time()
    if state["reference"] is None:
        output = os.path.join(eval_folder, "published_reference.json")
        r = evaluate_external(
            args.python, CS.PUBLISHED_SR[0], CS.PUBLISHED_SR[1], output, extra
        )
        state["reference"] = registration(r, -1, "published_reference", output)
        state["reference"]["sr_seed"] = r["sr_seed"]
        state["design"]["cases"] = r["cases"]
        state["design"]["sr_seed"] = r["sr_seed"]
        ES.save(state, args.state)
    ref = state["reference"]
    opt = new_optimizer()
    done = resume(opt, state)
    print(
        "resuming with %d observations; %d remaining" % (done, max(0, args.n - done)),
        flush=True,
    )
    for i in range(done, args.n):
        seed(i)
        rec = opt.suggest(n_suggestions=1)
        sobol_generated = int(opt.sobol.num_generated)
        phase = "sobol" if i < opt.rand_sample else "gp_mace"
        lr = float(rec["lr"].iloc[0])
        iters = int(round(float(rec["iters"].iloc[0])))
        assert CS.SR_SEARCH_LR[0] <= lr <= CS.SR_SEARCH_LR[1], lr
        assert CS.SR_SEARCH_ITERS[0] <= iters <= CS.SR_SEARCH_ITERS[1], iters
        output = os.path.join(eval_folder, "eval_%02d.json" % i)
        try:
            r = evaluate_external(args.python, lr, iters, output, extra)
            o = registration(r, i, phase, output)
            assert o["cases"] == state["design"]["cases"], "set of cases changed"
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            AssertionError,
        ) as exc:
            # sr_config.FAILURE_RULE: whole evaluation failed -> trilinear objective of the reference, observed
            o = {
                "index": i,
                "phase": phase,
                "lr": lr,
                "iters": iters,
                "objective_mm": ref["tri_objective_mm"],
                "failure": "%s: %s" % (type(exc).__name__, str(exc)[:300]),
                "file": None,
            }
            state["failures"].append(o)
            print("  FAILURE in evaluation %d: %s" % (i, o["failure"]), flush=True)
        o["sobol_generated"] = sobol_generated
        opt.observe(frame_observed(lr, iters), np.array([[o["objective_mm"]]], float))
        state["observations"].append(o)
        state["best"] = ES.best(state)
        ES.save(state, args.state)
        print(
            "[%d/%d] %s lr=%.3g iters=%d -> %.4f mm | best %.4f (lr=%.3g, iters=%d) | %.1f h"
            % (
                i + 1,
                args.n,
                phase,
                lr,
                iters,
                o["objective_mm"],
                state["best"]["objective_mm"],
                state["best"]["lr"],
                state["best"]["iters"],
                (time.time() - t0) / 3600,
            ),
            flush=True,
        )
    state["done"] = len(state["observations"]) >= args.n
    state["summary"] = ES.summary(state)
    ES.save(state, args.state)
    if state["done"] and not extra:
        os.makedirs(os.path.dirname(RESULT), exist_ok=True)
        ES.save(state, RESULT)
        print("result:", RESULT, flush=True)
    print(json.dumps(state["summary"], ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
