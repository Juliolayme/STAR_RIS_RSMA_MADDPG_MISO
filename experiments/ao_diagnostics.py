"""Validation-only diagnostics for the pre-registered Hybrid AO solver.

This command evaluates the exact penalty/Lagrangian objective used by the AO
reference. It refuses the locked test split and reports per-scenario objective,
QoS violation, convergence, evaluation count, and solve time so solver
hyperparameters can be frozen before final-paper aggregation.
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env import build_eval_bank  # noqa: E402
from experiments.train import _make_env, configure_cpu_threads  # noqa: E402
from experiments.baselines_ao import (  # noqa: E402
    AOHybridLocalSearch, ao_reference_lambda, solver_params_from_config,
    stratified_ao_scenarios,
)


def run_ao_diagnostics(cfg: dict, split: str = "validation",
                       scenarios_per_seed: int = 1, seed: int = 0) -> pd.DataFrame:
    if split == "test":
        raise ValueError(
            "AO solver diagnostics must not consume the locked test split; "
            "use --split validation or dev.")
    bank = build_eval_bank(cfg, split)
    scenarios = stratified_ao_scenarios(bank.scenarios, per_seed=scenarios_per_seed)
    params = solver_params_from_config(cfg)
    reference_lambda = ao_reference_lambda(cfg)
    rows = []
    for sc in scenarios:
        env = _make_env(cfg, seed=seed)
        env.set_qos_lambda_vec(reference_lambda)
        env.reset(options={"scenario": sc})
        solver = AOHybridLocalSearch(
            env, seed=seed + int(sc["evaluation_seed"]), objective="penalty", **params)
        sol = solver.solve()
        rates = np.asarray(sol["per_user_rate"], dtype=np.float64)
        violations = np.maximum(env.qos_min - rates, 0.0)
        trace = np.asarray(sol["objective_trace"], dtype=np.float64)
        rows.append({
            "split": split,
            "scenario_id": sc["scenario_id"],
            "evaluation_seed": int(sc["evaluation_seed"]),
            "episode_idx": int(sc["episode_idx"]),
            "sum_rate": float(sol["sum_rate"]),
            "user_qos_fraction": float(np.mean(rates >= env.qos_min)),
            "all_users_qos": bool(np.all(rates >= env.qos_min)),
            "min_user_rate": float(np.min(rates)),
            "mean_qos_violation": float(np.mean(violations)),
            "max_qos_violation": float(np.max(violations)),
            "objective": float(sol["objective"]),
            "objective_gain": float(trace[-1] - trace[0]),
            "converged": bool(sol["converged"]),
            "n_evals": int(sol["n_evals"]),
            "solve_time_ms": float(sol["solve_time_ms"]),
        })
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description="Hybrid AO diagnostics (validation-only)")
    p.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config", "config.yaml"))
    p.add_argument("--split", default="validation", choices=["validation", "dev"])
    p.add_argument("--scenarios-per-seed", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="ao_solver_validation.csv")
    args = p.parse_args(argv)

    configure_cpu_threads(1, 1)
    from main import load_config
    cfg = load_config(args.config)
    df = run_ao_diagnostics(
        cfg, split=args.split, scenarios_per_seed=args.scenarios_per_seed, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    if len(df):
        objective_mean = float(df["objective"].mean())
        objective_std = float(df["objective"].std(ddof=1)) if len(df) > 1 else 0.0
        objective_cv = objective_std / max(abs(objective_mean), 1e-12)
        print(
            f"\nConverged fraction: {df['converged'].mean():.3f}; "
            f"mean/p95 solve time: {df['solve_time_ms'].mean():.1f}/"
            f"{df['solve_time_ms'].quantile(0.95):.1f} ms; "
            f"max QoS violation: {df['max_qos_violation'].max():.6f}; "
            f"objective mean/std/CV: {objective_mean:.6f}/{objective_std:.6f}/"
            f"{objective_cv:.3f}")


if __name__ == "__main__":
    main()
