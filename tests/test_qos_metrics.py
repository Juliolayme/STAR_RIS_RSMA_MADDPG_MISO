"""QoS metric definitions (P0-2): the two probabilities are distinct and
correctly computed."""
from __future__ import annotations
import numpy as np

from env import StarRisRsmaEnv
from conftest import base_env_cfg


def test_metric_definitions_from_info():
    env = StarRisRsmaEnv(base_env_cfg(), seed=1)
    env.reset(seed=2)
    rng = np.random.default_rng(3)
    for _ in range(5):
        action = rng.uniform(-1, 1, size=env.act_dim_flat).astype(np.float32)
        _, _, _, _, info = env.step(action)
        r = np.asarray(info["per_user_rate"])
        sat = (r >= env.qos_min)
        # user_qos_fraction = (1/K) sum_k 1[R_k >= R_min]
        assert abs(info["user_qos_fraction"] - sat.mean()) < 1e-12
        # all_users_qos_satisfied = 1[min_k R_k >= R_min]
        assert info["all_users_qos_satisfied"] == bool(sat.all())
        np.testing.assert_allclose(info["per_user_qos_satisfied"],
                                   sat.astype(float))
        assert abs(info["min_user_rate"] - r.min()) < 1e-12
        deficit = np.maximum(env.qos_min - r, 0.0)
        assert abs(info["mean_qos_deficit"] - deficit.mean()) < 1e-12
        assert abs(info["max_qos_deficit"] - deficit.max()) < 1e-12
        np.testing.assert_allclose(info["qos_constraint_signed"],
                                   env.qos_min - r)


def test_two_metrics_differ_when_some_users_fail():
    """Constructed rates: 3 of 4 users satisfied -> fraction 0.75, all-flag 0."""
    env = StarRisRsmaEnv(base_env_cfg(), seed=1)
    rates = np.array([0.5, 0.4, 0.35, 0.1])   # last user below 0.3
    sat = rates >= env.qos_min
    assert sat.mean() == 0.75
    assert not sat.all()


def test_legacy_ambiguous_key_removed():
    env = StarRisRsmaEnv(base_env_cfg(), seed=1)
    env.reset(seed=2)
    rng = np.random.default_rng(3)
    _, _, _, _, info = env.step(
        rng.uniform(-1, 1, size=env.act_dim_flat).astype(np.float32))
    assert "qos_satisfied" not in info, \
        "ambiguous legacy key must not be reported"
    assert "qos_prob" not in info
