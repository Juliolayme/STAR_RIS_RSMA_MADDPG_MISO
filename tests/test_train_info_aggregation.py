"""Regression coverage for one-and-only-one per-step rate sample."""
from __future__ import annotations
from collections import defaultdict

import numpy as np
import pytest

from env import StarRisRsmaEnv
from experiments.train import _aggregate_info, _summarize
from conftest import base_env_cfg


def test_per_user_rate_sample_count_equals_environment_steps():
    env = StarRisRsmaEnv(base_env_cfg(max_steps=7), seed=123)
    env.reset(seed=456)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    buf = defaultdict(list)
    steps = 0
    while True:
        _, reward, terminated, truncated, info = env.step(action)
        _aggregate_info(buf, info, reward)
        steps += 1
        if terminated or truncated:
            break

    assert steps == env.max_steps
    assert len(buf["per_user_rate"]) == steps
    assert len(buf["qos_constraint_signed"]) == steps
    summary = _summarize(buf)
    expected = np.mean(np.stack(buf["per_user_rate"]), axis=0)
    np.testing.assert_allclose(summary["episode_mean_per_user_rate"], expected)


def test_explicit_cuda_never_silently_falls_back_to_cpu(monkeypatch):
    import experiments.train as train

    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="refusing to silently fall back"):
        train._select_device("cuda")
