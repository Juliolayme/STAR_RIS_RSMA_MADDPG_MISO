"""Every results_raw row carries full provenance (item 8): evaluation_seed,
episode_idx, scenario_id, config_sha and (checkpoint_sha or solver_config_sha).
"""
from __future__ import annotations

from env import ScenarioBank
from experiments.evaluate import scenario_rows
from conftest import base_env_cfg


def _fake_metrics(n):
    return {
        "per_episode_sum_rate": [1.0] * n,
        "per_episode_user_qos_fraction": [0.5] * n,
        "per_episode_all_users_qos": [0.0] * n,
        "per_episode_return": [0.1] * n,
    }


REQUIRED = {"algorithm", "evaluation_seed", "episode_idx", "scenario_id",
            "config_sha", "checkpoint_sha", "solver_config_sha", "metric", "value"}


def test_learned_rows_have_full_metadata():
    cfg = base_env_cfg()
    bank = ScenarioBank(cfg, split="test", evaluation_seeds=[9101], episodes_per_seed=2)
    rows = scenario_rows("MADDPG", _fake_metrics(len(bank)), bank.scenarios,
                         training_seed=1000, config_sha="cfgsha",
                         checkpoint_sha="cksha")
    assert rows
    for r in rows:
        assert REQUIRED.issubset(r.keys())
        assert r["config_sha"] == "cfgsha"
        assert r["checkpoint_sha"] == "cksha"
        assert r["scenario_id"].startswith("test_seed9101_ep")
        assert r["evaluation_seed"] == 9101
        assert r["episode_idx"] in (0, 1)


def test_baseline_rows_use_solver_config_sha():
    cfg = base_env_cfg()
    bank = ScenarioBank(cfg, split="test", evaluation_seeds=[9101], episodes_per_seed=1)
    rows = scenario_rows("AO-Grid", _fake_metrics(len(bank)), bank.scenarios,
                         training_seed=None, config_sha="cfgsha",
                         solver_config_sha="solversha")
    for r in rows:
        assert REQUIRED.issubset(r.keys())
        assert r["checkpoint_sha"] == ""            # no learned checkpoint
        assert r["solver_config_sha"] == "solversha"
