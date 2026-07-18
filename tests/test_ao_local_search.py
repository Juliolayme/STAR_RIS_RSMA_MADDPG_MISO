"""Hybrid AO Local Search (SLSQP + projected gradient) reference."""
from __future__ import annotations
import numpy as np
import pytest

scipy = pytest.importorskip("scipy")

from env import StarRisRsmaEnv
from experiments.baselines_ao import (
    AOHybridLocalSearch, ao_reference_lambda, stratified_ao_scenarios,
)
from conftest import base_env_cfg


def _solver(env, **kw):
    defaults = dict(n_starts=2, max_outer=2, pg_steps=5, pg_lr=0.3, seed=0)
    defaults.update(kw)
    return AOHybridLocalSearch(env, **defaults)


def _small_env():
    env = StarRisRsmaEnv(base_env_cfg(num_ris_elements=4, num_users=2,
                                      num_users_reflection=1), seed=1)
    env.reset(seed=2)
    return env


def test_objective_trace_nondecreasing_and_flags():
    env = _small_env()
    sol = _solver(env).solve()
    trace = np.asarray(sol["objective_trace"])
    assert np.all(np.diff(trace) >= -1e-9)
    assert isinstance(sol["qos_feasible"], bool)
    assert isinstance(sol["converged"], bool)
    assert sol["n_evals"] > 0


def test_solution_feasibility_and_full_beta_vector():
    env = _small_env()
    sol = _solver(env).solve()
    total = sol["P_c"] + sol["P_k"].sum()
    np.testing.assert_allclose(total, env.p_max, rtol=1e-6)
    np.testing.assert_allclose(sol["common_split"].sum(), 1.0, rtol=1e-9)
    # Full per-element beta vector (variable space of P0, not one shared beta).
    assert sol["beta_r"].shape == (env.N,)
    assert np.all(sol["beta_r"] >= 1e-4) and np.all(sol["beta_r"] <= 1 - 1e-4)


def test_switching_cost_enters_objective():
    env = _small_env()
    solver = _solver(env)
    x = np.concatenate([np.full(env.K + 1, 1.0 / (env.K + 1)),
                        np.full(env.K, 1.0 / env.K)])
    beta = 0.5 * np.ones(env.N)
    phi = np.zeros(env.N)
    same_prev = {"phi_r": phi, "phi_t": phi, "beta_r": beta,
                 "power_weights": x[: env.K + 1]}
    far_prev = {"phi_r": phi + np.pi, "phi_t": phi + np.pi,
                "beta_r": 1.0 - beta + 0.3, "power_weights": x[: env.K + 1][::-1]}
    j_same = solver._objective(x, beta, phi, phi, same_prev)
    j_far = solver._objective(x, beta, phi, phi, far_prev)
    j_none = solver._objective(x, beta, phi, phi, None)
    assert j_far < j_same, "reconfiguration must be penalised"
    np.testing.assert_allclose(j_same, j_none, rtol=1e-12)


def test_beats_or_matches_fixed_ris_start():
    env = _small_env()
    solver = _solver(env, max_outer=3, pg_steps=10)
    x0 = np.concatenate([np.full(env.K + 1, 1.0 / (env.K + 1)),
                         np.full(env.K, 1.0 / env.K)])
    j_uniform = solver._objective(x0, 0.5 * np.ones(env.N),
                                  np.zeros(env.N), np.zeros(env.N), None)
    sol = solver.solve()
    assert sol["objective"] >= j_uniform - 1e-9


def test_solve_reports_wall_clock_time():
    env = _small_env()
    sol = _solver(env).solve()
    assert "solve_time_ms" in sol and sol["solve_time_ms"] > 0.0


def test_solver_params_from_config_validated():
    from experiments.baselines_ao import solver_params_from_config
    cfg = {"evaluation": {"ao_solver": {"n_starts": 2, "max_outer": 3,
                                        "tol": 1e-4, "pg_steps": 5, "pg_lr": 0.2}}}
    params = solver_params_from_config(cfg)
    assert params["n_starts"] == 2 and params["pg_steps"] == 5
    assert solver_params_from_config({"evaluation": {}}) == {}
    with pytest.raises(ValueError):
        solver_params_from_config({"evaluation": {"ao_solver": {"bogus": 1}}})


def test_maxmin_objective_for_feasibility_study():
    env = _small_env()
    solver = _solver(env, objective="maxmin", max_outer=2, pg_steps=5)
    x = np.concatenate([np.full(env.K + 1, 1.0 / (env.K + 1)),
                        np.full(env.K, 1.0 / env.K)])
    beta = 0.5 * np.ones(env.N)
    phi = np.zeros(env.N)
    j = solver._objective(x, beta, phi, phi, None)
    h = env._effective_channels(beta, phi, phi)
    rs = env._rsma_rates(h, float(x[0] * env.p_max),
                         (x[1:env.K + 1] * env.p_max), x[env.K + 1:])
    assert abs(j - float(np.min(rs["per_user"]))) < 1e-9
    sol = solver.solve()
    # The solved max-min objective equals the worst-user rate of the solution.
    assert abs(sol["objective"] - float(np.min(sol["per_user_rate"]))) < 1e-6


def test_ao_reference_lambda_is_preregistered_and_validated():
    cfg = {"env": {"num_users": 2},
           "evaluation": {"ao_reference_lambda": [1.0, 2.0]}}
    np.testing.assert_array_equal(ao_reference_lambda(cfg), [1.0, 2.0])
    with pytest.raises(ValueError):
        ao_reference_lambda({"env": {"num_users": 2}, "evaluation": {}})
    with pytest.raises(ValueError):
        ao_reference_lambda({"env": {"num_users": 2},
                             "evaluation": {"ao_reference_lambda": [1.0]}})


def test_ao_scenario_subset_is_stratified_across_every_seed():
    scenarios = [
        {"evaluation_seed": seed, "episode_idx": ep,
         "scenario_id": f"test_seed{seed}_ep{ep:03d}"}
        for seed in (70001, 70002, 70003) for ep in range(4)
    ]
    selected = stratified_ao_scenarios(scenarios, per_seed=2)
    assert len(selected) == 6
    assert {s["evaluation_seed"] for s in selected} == {70001, 70002, 70003}
    assert all(sum(s["evaluation_seed"] == seed for s in selected) == 2
               for seed in (70001, 70002, 70003))
