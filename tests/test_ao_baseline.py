"""Coarse AO-grid heuristic: feasibility, monotone objective, honest naming."""
from __future__ import annotations
import warnings
import numpy as np

from env import StarRisRsmaEnv
from conftest import base_env_cfg


def test_ao_grid_objective_nondecreasing_and_feasible():
    env = StarRisRsmaEnv(base_env_cfg(), seed=1, ris_mode="ao_grid")
    env.reset(seed=2)
    sol = env._coarse_ao_grid(n_iter=10)
    trace = np.asarray(sol["ao_grid_objective_trace"])
    assert np.all(np.diff(trace) >= -1e-9), "greedy grid accept must be monotone"
    assert sol["ao_grid_n_evals"] > 0
    total = sol["P_c"] + sol["P_k"].sum()
    np.testing.assert_allclose(total, env.p_max, rtol=1e-9)
    np.testing.assert_allclose(sol["common_split"].sum(), 1.0, rtol=1e-12)
    assert np.all(sol["beta_r"] > 0) and np.all(sol["beta_r"] < 1)


def test_bcd_alias_is_deprecated():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = StarRisRsmaEnv(base_env_cfg(), seed=1, ris_mode="bcd")
    assert env.ris_mode == "ao_grid"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_no_upper_bound_claims_in_env_source():
    import env.star_ris_env as mod
    src = open(mod.__file__, "r", encoding="utf-8").read().lower()
    for banned in ("upper bound", "upper-bound", "near-optimal",
                   "classical optimum"):
        # The only allowed occurrences are explicit negations
        # ("not an upper bound").
        for line in src.splitlines():
            if banned in line:
                assert "not an upper bound" in line or "not an upper-bound" in line, \
                    f"banned claim {banned!r} in env source: {line.strip()}"


def test_policy_independent_ao_grid_uses_registered_not_trained_lambda(monkeypatch):
    import experiments.ablation as ablation
    from conftest import full_cfg

    cfg = full_cfg()
    registered = np.array([1.0, 2.0, 3.0, 4.0])
    cfg["evaluation"]["ao_reference_lambda"] = registered.tolist()
    trained = np.array([9.0, 9.0, 9.0, 9.0])
    captured = []

    def fake_eval(*args, **kwargs):
        captured.append(np.asarray(kwargs["qos_lambda_vec"]))
        return {
            "per_episode_sum_rate": np.array([1.0]),
            "per_episode_user_qos_fraction": np.array([0.5]),
            "per_episode_all_users_qos": np.array([0.0]),
            "rate_common_mean": 0.1,
            "h_eff_abs_T_mean": 0.2,
            "phase_entropy_T_mean": 0.3,
            "common_power_frac_mean": 0.4,
        }

    monkeypatch.setattr(ablation, "ABLATION_CELLS",
                        [("AO-Grid", "ao_grid", False, False)])
    monkeypatch.setattr(ablation, "eval_on_scenarios", fake_eval)
    ablation.ablation_study(
        [{"agent": object(), "trained_qos_lambda_vec": trained, "seed": 1000}],
        cfg, [{"evaluation_seed": 1, "episode_idx": 0, "scenario_id": "s"}],
    )
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], registered)
