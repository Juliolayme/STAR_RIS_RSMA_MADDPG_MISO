"""Dynamic-MDP channel evolution, transition order, RNG separation."""
from __future__ import annotations
import numpy as np
import pytest

from env import StarRisRsmaEnv
from experiments.train import validate_formulation_config
from conftest import base_env_cfg, full_cfg


def _rand_action(env, rng):
    return rng.uniform(-1.0, 1.0, size=env.act_dim_flat).astype(np.float32)


def test_gauss_markov_correlation_on_small_scale():
    """Lag-1 correlation of the SMALL-SCALE channel matches channel_rho.

    Measured on _G_small (not on h_eff, which mixes in large-scale gains and
    the RIS configuration)."""
    rho = 0.8
    cfg = base_env_cfg(num_ris_elements=64, max_steps=200, channel_rho=rho)
    env = StarRisRsmaEnv(cfg, seed=10)
    env.reset(seed=11)
    rng = np.random.default_rng(12)
    prev = env._G_small.copy()
    num, den = 0.0, 0.0
    for _ in range(cfg["max_steps"]):
        env.step(_rand_action(env, rng))
        cur = env._G_small
        num += float(np.real(np.vdot(prev, cur)))
        den += float(np.real(np.vdot(prev, prev)))
        prev = cur.copy()
    rho_hat = num / den
    assert abs(rho_hat - rho) < 0.05, f"estimated rho {rho_hat} vs {rho}"


def test_reward_computed_on_observed_channel_then_evolve():
    """Transition order: reward on h_t, evolution AFTER (next_obs has h_{t+1})."""
    cfg = base_env_cfg(equal_power_mode=True)
    env = StarRisRsmaEnv(cfg, seed=5, ris_mode="fixed")
    env.reset(seed=6)
    h_d_pre = env._h_d.copy()
    G_pre = env._G.copy()
    g_pre = env._g.copy()
    rng = np.random.default_rng(7)
    _, _, _, _, info = env.step(_rand_action(env, rng))
    # Channel must have evolved AFTER the reward computation.
    assert not np.allclose(env._h_d, h_d_pre)
    # Reconstruct the physics on the PRE-step channel with the deterministic
    # decoded action (fixed RIS: beta=0.5, phases=0; equal power).
    beta = 0.5 * np.ones(env.N)
    coeff = np.sqrt(beta)  # phases are zero
    # MISO cascade: h_eff,k = h_d,k + sum_n conj(g_k[n]) * coeff_n * G[n, :].
    h_eff_manual = np.array([
        h_d_pre[k] + np.sum((np.conj(g_pre[k]) * coeff)[:, None] * G_pre, axis=0)
        for k in range(env.K)])
    P_c = env.p_max / (env.K + 1)
    P_k = np.full(env.K, env.p_max / (env.K + 1))
    split = np.ones(env.K) / env.K
    rs = env._rsma_rates(h_eff_manual, P_c, P_k, split)
    np.testing.assert_allclose(info["sum_rate"], rs["sum_rate"], rtol=1e-9)


def test_same_seed_same_trajectory():
    cfg = base_env_cfg()
    rewards = []
    for _ in range(2):
        env = StarRisRsmaEnv(cfg, seed=42)
        env.reset(seed=43)
        rng = np.random.default_rng(44)
        rs = [env.step(_rand_action(env, rng))[1] for _ in range(cfg["max_steps"])]
        rewards.append(rs)
    np.testing.assert_allclose(rewards[0], rewards[1], rtol=0, atol=0)


def test_channel_rng_independent_of_actions():
    """Two different policies, same seed: identical channel trajectory."""
    cfg = base_env_cfg()
    trajs = []
    for action_seed in (100, 200):
        env = StarRisRsmaEnv(cfg, seed=42)
        env.reset(seed=43)
        rng = np.random.default_rng(action_seed)
        seq = []
        for _ in range(cfg["max_steps"]):
            env.step(_rand_action(env, rng))
            seq.append(env._h_d_small.copy())
        trajs.append(np.stack(seq))
    np.testing.assert_allclose(trajs[0], trajs[1], rtol=0, atol=0)


def test_contextual_bandit_one_step_and_gamma_validation():
    cfg = base_env_cfg(env_formulation="contextual_bandit", max_steps=50)
    env = StarRisRsmaEnv(cfg, seed=1)
    assert env.max_steps == 1
    env.reset(seed=2)
    rng = np.random.default_rng(3)
    _, _, term, trunc, _ = env.step(_rand_action(env, rng))
    assert trunc and not term
    # gamma != 0 must raise a validation error (no silent override).
    tcfg = full_cfg(env_formulation="contextual_bandit")
    with pytest.raises(ValueError):
        validate_formulation_config(tcfg)
    tcfg2 = full_cfg(env_formulation="contextual_bandit")
    for algo in ("maddpg", "ddpg", "td3", "ppo"):
        tcfg2[algo]["gamma"] = 0.0
    validate_formulation_config(tcfg2)   # must not raise


def test_switching_cost_zero_at_t0_and_for_repeated_action():
    cfg = base_env_cfg(phase_action_mode="absolute")
    env = StarRisRsmaEnv(cfg, seed=1)
    env.reset(seed=2)
    rng = np.random.default_rng(3)
    action = _rand_action(env, rng)
    _, _, _, _, info0 = env.step(action)
    assert info0["phase_switch_cost"] == 0.0
    assert info0["power_switch_cost"] == 0.0
    assert info0["beta_switch_cost"] == 0.0
    # Same action again (absolute phase mapping -> same physical config).
    _, _, _, _, info1 = env.step(action)
    assert abs(info1["phase_switch_cost"]) < 1e-12
    assert abs(info1["power_switch_cost"]) < 1e-12
    assert abs(info1["beta_switch_cost"]) < 1e-12
    # A different action must incur a positive cost.
    _, _, _, _, info2 = env.step(-action)
    assert info2["phase_switch_cost"] > 0.0


def test_advance_to_next_block_innovation_index_advances():
    """External driver (AO baselines) advances the channel with innovation
    indices 0, 1, 2 -- never reusing index 0 (item 5)."""
    from env import ScenarioBank
    cfg = base_env_cfg(max_steps=4, channel_rho=0.7)
    bank = ScenarioBank(cfg, split="test", evaluation_seeds=[3], episodes_per_seed=1)
    sc = bank[0]
    env = StarRisRsmaEnv(cfg, seed=5)
    env.reset(options={"scenario": sc})
    rho = cfg["channel_rho"]
    scale = np.sqrt(1 - rho ** 2)
    h = np.asarray(sc["h_d_small0"])
    assert env.current_step() == 0
    for t in range(3):
        np.testing.assert_allclose(env._h_d_small, h, rtol=1e-12)
        env.advance_to_next_block()
        assert env.current_step() == t + 1
        h = rho * h + scale * np.asarray(sc["innov_h_d"][t])  # uses innov[t]
    np.testing.assert_allclose(env._h_d_small, h, rtol=1e-12)


def test_static_block_keeps_channel_within_block():
    cfg = base_env_cfg(env_formulation="static_block", max_steps=6,
                       channel_block_steps=3)
    env = StarRisRsmaEnv(cfg, seed=1)
    env.reset(seed=2)
    rng = np.random.default_rng(3)
    h0 = env._h_d.copy()
    env.step(_rand_action(env, rng))          # step 0 (same block)
    np.testing.assert_allclose(env._h_d, h0)
    env.step(_rand_action(env, rng))          # step 1 (same block)
    np.testing.assert_allclose(env._h_d, h0)
    env.step(_rand_action(env, rng))          # step 2 (same block)
    np.testing.assert_allclose(env._h_d, h0)
    env.step(_rand_action(env, rng))          # step 3 -> refreshed at start
    assert not np.allclose(env._h_d, h0)
