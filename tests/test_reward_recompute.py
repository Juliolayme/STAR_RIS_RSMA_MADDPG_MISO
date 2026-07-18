"""Replay reward recomputation under the current dual multipliers (item 1)."""
from __future__ import annotations
import numpy as np

from env import StarRisRsmaEnv
from utils.replay_buffer import ReplayBuffer, MAReplayBuffer, _recompute_reward
from conftest import base_env_cfg


def test_recompute_formula():
    base = np.array([[1.0]])
    c = np.array([[0.2, -0.1]])
    # reward = clip(base - scale * (c @ lambda))
    r1 = _recompute_reward(base, c, [1.0, 1.0], reward_scale=0.5, reward_clip=50.0)
    r2 = _recompute_reward(base, c, [2.0, 2.0], reward_scale=0.5, reward_clip=50.0)
    assert abs(r1.item() - (1.0 - 0.5 * (0.2 - 0.1))) < 1e-6   # 0.95
    assert abs(r2.item() - (1.0 - 0.5 * (0.4 - 0.2))) < 1e-6   # 0.90
    assert not np.allclose(r1, r2)


def test_same_transition_differs_under_different_lambda():
    buf = ReplayBuffer(8, obs_dim=3, act_dim=2, n_users=2)
    buf.add(np.zeros(3), np.zeros(2), reward=0.0, next_obs=np.zeros(3), done=0.0,
            base_reward=1.0, c_gap=[0.2, -0.1])
    kw = dict(reward_scale=0.5, reward_clip=50.0)
    r_a = buf.sample(1, rng=np.random.default_rng(0), lambda_vec=[1.0, 1.0], **kw)[2]
    r_b = buf.sample(1, rng=np.random.default_rng(0), lambda_vec=[3.0, 3.0], **kw)[2]
    assert not np.allclose(r_a, r_b), "reward must depend on current lambda"


def test_stationary_reward_after_freeze():
    """Once lambda is fixed (post-freeze), the recomputed reward for a fixed
    transition set is invariant across resamples (same seed -> same indices)."""
    buf = ReplayBuffer(4, 3, 2, n_users=2)
    for i in range(4):
        buf.add(np.full(3, i), np.zeros(2), 0.0, np.zeros(3), 0.0,
                base_reward=float(i), c_gap=[0.1, 0.1])
    kw = dict(lambda_vec=[2.0, 2.0], reward_scale=0.1, reward_clip=50.0)
    r1 = buf.sample(4, rng=np.random.default_rng(1), **kw)[2]
    r2 = buf.sample(4, rng=np.random.default_rng(1), **kw)[2]
    np.testing.assert_allclose(r1, r2)


def test_ma_buffer_shared_reward_recompute():
    buf = MAReplayBuffer(4, obs_dims=[2, 3], act_dims=[1, 2], n_users=2)
    buf.add([np.zeros(2), np.zeros(3)], [np.zeros(1), np.zeros(2)],
            [0.0, 0.0], [np.zeros(2), np.zeros(3)], 0.0,
            base_reward=1.0, c_gap=[0.3, 0.0])
    kw = dict(reward_scale=0.5, reward_clip=50.0)
    _, _, rew_a, _, _ = buf.sample(1, rng=np.random.default_rng(0), lambda_vec=[0.0, 0.0], **kw)
    _, _, rew_b, _, _ = buf.sample(1, rng=np.random.default_rng(0), lambda_vec=[4.0, 4.0], **kw)
    # both agents share the recomputed reward
    np.testing.assert_allclose(rew_a[0], rew_a[1])
    assert not np.allclose(rew_a[0], rew_b[0])


def test_env_base_reward_reconstructs_collection_reward():
    """The env's info gives base_reward + c so the collection-time reward equals
    clip(base - reward_scale * dot(lambda, c))."""
    env = StarRisRsmaEnv(base_env_cfg(), seed=1)
    env.reset(seed=2)
    rng = np.random.default_rng(3)
    for _ in range(5):
        a = rng.uniform(-1, 1, size=env.act_dim_flat).astype(np.float32)
        _, reward, _, _, info = env.step(a)
        base = info["base_reward"]
        c = np.asarray(info["qos_constraint_signed"])
        lam = np.asarray(info["qos_lambda_vec"])
        recon = np.clip(base - env.r_scale * np.dot(lam, c), -env.r_clip, env.r_clip)
        assert abs(recon - reward) < 1e-6
