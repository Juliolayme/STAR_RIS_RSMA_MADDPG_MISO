"""Validation-only feasibility study for the QoS threshold R_min (V4 item 10).

For each scenario in a DEVELOPMENT or VALIDATION bank, runs the Hybrid AO
local search with the max-min objective (J = min_k R_k) and reports how often
the achievable worst-user rate reaches R_min. This decides -- BEFORE the
40-shard experiment -- whether any hard-QoS phrasing is defensible and informs
the frozen value of model_select_constraint_tolerance.

The script refuses to touch the locked test split: feasibility evidence must
come from validation/development scenarios only.

Usage:
    python -m experiments.feasibility --config config/config.yaml \
        --split validation --scenarios-per-seed 2 --out feasibility_validation.csv
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env import build_eval_bank                        # noqa: E402
from experiments.train import _make_env, configure_cpu_threads  # noqa: E402
from experiments.baselines_ao import (                 # noqa: E402
    AOHybridLocalSearch, solver_params_from_config, stratified_ao_scenarios,
)


def run_feasibility(cfg: dict, split: str = "validation",
                    scenarios_per_seed: int = 1, seed: int = 0) -> pd.DataFrame:
    """Max-min achievable rate per scenario on a non-test split."""
    if split == "test":
        raise ValueError(
            "Feasibility analysis must not consume the locked test split; "
            "use --split validation or dev.")
    bank = build_eval_bank(cfg, split)
    scenarios = stratified_ao_scenarios(bank.scenarios, per_seed=scenarios_per_seed)
    r_min = float(cfg["env"]["qos_rate_min"])
    solver_params = solver_params_from_config(cfg)
    rows = []
    for sc in scenarios:
        env = _make_env(cfg, seed=seed)
        env.reset(options={"scenario": sc})
        solver = AOHybridLocalSearch(env, seed=seed + int(sc["evaluation_seed"]),
                                     objective="maxmin", **solver_params)
        t0 = time.perf_counter()
        sol = solver.solve()
        maxmin_rate = float(np.min(sol["per_user_rate"]))
        max_qos_violation = float(max(0.0, r_min - maxmin_rate))
        rows.append({
            "split": split,
            "scenario_id": sc["scenario_id"],
            "evaluation_seed": int(sc["evaluation_seed"]),
            "episode_idx": int(sc["episode_idx"]),
            "R_min": r_min,
            "maxmin_rate": maxmin_rate,
            "objective": float(sol["objective"]),
            "feasible": bool(maxmin_rate >= r_min),
            "max_qos_violation": max_qos_violation,
            "sum_rate_at_maxmin": float(sol["sum_rate"]),
            "converged": bool(sol["converged"]),
            "n_evals": int(sol["n_evals"]),
            "solve_time_ms": float(sol["solve_time_ms"]),
            "wall_time_ms": (time.perf_counter() - t0) * 1000.0,
        })
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description="R_min feasibility study (validation-only)")
    p.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config", "config.yaml"))
    p.add_argument("--split", default="validation", choices=["validation", "dev"])
    p.add_argument("--scenarios-per-seed", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="feasibility_validation.csv")
    args = p.parse_args(argv)

    configure_cpu_threads(1, 1)
    from main import load_config
    cfg = load_config(args.config)
    df = run_feasibility(cfg, split=args.split,
                         scenarios_per_seed=args.scenarios_per_seed,
                         seed=args.seed)
    df.to_csv(args.out, index=False)
    frac = float(df["feasible"].mean()) if len(df) else float("nan")
    print(df.to_string(index=False))
    print(f"\nFeasible fraction (maxmin rate >= R_min={df['R_min'].iloc[0] if len(df) else '?'}): "
          f"{frac:.3f} over {len(df)} scenarios")
    if len(df):
        print(f"Converged fraction: {df['converged'].mean():.3f}; "
              f"mean solve time: {df['solve_time_ms'].mean():.1f} ms; "
              f"max QoS violation: {df['max_qos_violation'].max():.6f}")
    print("Decision guidance: if this fraction is low, hard-QoS claims are "
          "indefensible -- keep the expected-rate surrogate framing and choose "
          "model_select_constraint_tolerance accordingly (then freeze it).")


if __name__ == "__main__":
    main()
