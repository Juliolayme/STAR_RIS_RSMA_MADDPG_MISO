"""Time-limit truncation should end episodes without zeroing bootstrap targets."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import full_cfg
from experiments.train import (
    _with_qos_penalty_disabled,
    train_maddpg,
    train_single_agent,
    train_ppo,
)


def _cfg_for_one_truncated_step():
    cfg = full_cfg(max_steps=1)
    cfg["training"]["total_episodes"] = 1
    cfg["training"]["eval_every"] = 1000
    cfg["training"]["checkpoint_every"] = 1000
    for key in ("maddpg", "ddpg", "td3"):
        cfg[key]["warmup_steps"] = 100
        cfg[key]["batch_size"] = 16
    cfg["ppo"]["rollout_length"] = 8
    return cfg


def test_disable_qos_penalty_zeroes_all_penalty_terms():
    cfg = full_cfg()
    disabled = _with_qos_penalty_disabled(cfg)
    assert disabled["env"]["qos_lambda_init"] == 0.0
    assert disabled["env"]["dual_lambda_max"] == 0.0
    assert disabled["env"]["augmented_penalty_weight"] == 0.0
    assert disabled["env"]["enable_qos_shaping_bonus"] is False
    assert cfg["env"]["augmented_penalty_weight"] == 1.0


def test_offpolicy_buffers_bootstrap_through_time_limit(tmp_path):
    cfg = _cfg_for_one_truncated_step()
    res = train_maddpg(
        cfg,
        total_episodes=1,
        run_name="maddpg_trunc",
        log_dir=str(tmp_path / "logs"),
        ckpt_dir=str(tmp_path / "ckpt"),
        seed_override=123,
        disable_obs_norm=True,
    )
    dones = res["train_agent"].buffer.dones[: res["train_agent"].buffer.size]
    assert dones.size == 1
    np.testing.assert_allclose(dones, 0.0)

    for kind in ("ddpg", "td3"):
        res = train_single_agent(
            cfg,
            kind=kind,
            total_episodes=1,
            run_name=f"{kind}_trunc",
            log_dir=str(tmp_path / "logs"),
            ckpt_dir=str(tmp_path / "ckpt"),
            seed_override=123,
            disable_obs_norm=True,
        )
        dones = res["train_agent"].buffer.dones[: res["train_agent"].buffer.size]
        assert dones.size == 1
        np.testing.assert_allclose(dones, 0.0)


def test_ppo_bootstraps_through_time_limit(tmp_path, monkeypatch):
    captured = []

    def fake_learn(self, last_value=None):
        captured.extend(self.rollout.next_values[:self.rollout.size].tolist())
        self.rollout.reset()
        return {}

    monkeypatch.setattr("algorithms.ppo.agent.PPOAgent.learn", fake_learn)
    cfg = _cfg_for_one_truncated_step()
    train_ppo(
        cfg,
        total_episodes=1,
        run_name="ppo_trunc",
        log_dir=str(tmp_path / "logs"),
        ckpt_dir=str(tmp_path / "ckpt"),
        seed_override=123,
    )
    assert captured
    assert captured[-1] != pytest.approx(0.0)
