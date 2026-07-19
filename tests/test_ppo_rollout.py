"""PPO on-policy consistency (item 2): the stored old_log_prob equals the
log-prob recomputed on the stored (normalized) observation with the current
actor, before any optimizer update. Before the fix, re-normalizing raw
observations with drifted running statistics broke this identity."""
from __future__ import annotations
import numpy as np

from env import StarRisRsmaEnv
from algorithms import PPOAgent
from utils import ObservationNormalizer
from conftest import base_env_cfg, full_cfg


def _make_ppo(cfg):
    env = StarRisRsmaEnv(cfg["env"], seed=1)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, act_dim, cfg["networks"]["hidden_sizes"],
                     cfg["ppo"], cfg["networks"], device="cpu", seed=0)
    agent.attach_obs_normalizer(ObservationNormalizer(shape=(obs_dim,)))
    return env, agent


def test_rollout_logprob_consistency_while_stats_drift():
    cfg = full_cfg()
    env, agent = _make_ppo(cfg)
    obs, _ = env.reset(seed=2)
    # Collect a short rollout the SAME way the training loop does: select_action
    # normalizes and caches last_norm_obs; we store that exact vector. The
    # normalizer keeps updating (NOT frozen), which is precisely the condition
    # that used to break the identity.
    for _ in range(24):
        action, log_prob, value = agent.select_action(obs, explore=True)
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        next_value = 0.0 if term else agent.value(next_obs)
        agent.store(agent.last_norm_obs, action, log_prob, reward, value,
                    terminated=term, episode_end=done,
                    next_value=next_value)
        obs = next_obs if not done else env.reset(seed=3)[0]

    old_lp, new_lp = agent.rollout_logprob_consistency()
    assert old_lp.size > 0
    # Any residual difference is only the tanh round-trip (atanh) numerical
    # error; the normalization mismatch bug produced differences of order 0.1-10.
    np.testing.assert_allclose(new_lp, old_lp, rtol=0, atol=1e-3)


def test_rollout_stores_normalized_not_raw():
    cfg = full_cfg()
    env, agent = _make_ppo(cfg)
    obs, _ = env.reset(seed=5)
    action, log_prob, value = agent.select_action(obs, explore=True)
    agent.store(agent.last_norm_obs, action, log_prob, 0.0, value,
                terminated=False, episode_end=False,
                next_value=agent.value(obs))
    stored = agent.rollout.obs[0]
    np.testing.assert_allclose(stored, agent.last_norm_obs, rtol=1e-6)
    # After warmup the raw obs and normalized obs differ; ensure we did not
    # accidentally store the raw observation.
    assert not np.allclose(stored, obs.astype(np.float32))
