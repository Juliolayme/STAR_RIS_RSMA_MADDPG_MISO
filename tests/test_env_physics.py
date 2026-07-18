"""Physical invariants of the STAR-RIS RSMA environment."""
from __future__ import annotations
import math
import numpy as np

from env import StarRisRsmaEnv
from conftest import base_env_cfg


def _rand_action(env, rng):
    return rng.uniform(-1.0, 1.0, size=env.act_dim_flat).astype(np.float32)


def test_power_and_split_feasibility():
    env = StarRisRsmaEnv(base_env_cfg(), seed=1)
    env.reset(seed=2)
    rng = np.random.default_rng(3)
    decoded = env._decode_action(env._split_action(_rand_action(env, rng)))
    total = decoded["P_c"] + decoded["P_k"].sum()
    assert abs(total - env.p_max) < 1e-6 * env.p_max
    assert decoded["P_c"] >= 0 and np.all(decoded["P_k"] >= 0)
    assert abs(decoded["common_split"].sum() - 1.0) < 1e-6
    assert np.all(decoded["common_split"] >= 0)


def test_beta_and_phase_ranges():
    env = StarRisRsmaEnv(base_env_cfg(), seed=1)
    env.reset(seed=2)
    rng = np.random.default_rng(3)
    decoded = env._decode_action(env._split_action(_rand_action(env, rng)))
    beta_r = decoded["beta_r"]
    beta_t = 1.0 - beta_r
    assert np.all(beta_r > 0) and np.all(beta_r < 1)
    np.testing.assert_allclose(beta_r + beta_t, 1.0, rtol=1e-12)
    for phi in (decoded["phi_r"], decoded["phi_t"]):
        assert np.all(phi >= 0.0) and np.all(phi <= 2 * math.pi + 1e-12)


def test_no_ris_equals_direct_channel():
    env = StarRisRsmaEnv(base_env_cfg(), seed=1, ris_mode="none")
    env.reset(seed=2)
    rng = np.random.default_rng(3)
    env.step(_rand_action(env, rng))
    np.testing.assert_allclose(env._h_eff, env._h_d, rtol=1e-12)


def test_rsma_rate_identities():
    env = StarRisRsmaEnv(base_env_cfg(), seed=1)
    env.reset(seed=2)
    rng = np.random.default_rng(3)
    decoded = env._decode_action(env._split_action(_rand_action(env, rng)))
    h = env._effective_channels(decoded["beta_r"], decoded["phi_r"], decoded["phi_t"])
    rs = env._rsma_rates(h, decoded["P_c"], decoded["P_k"], decoded["common_split"])
    # R_sum = R_c + sum_k R_private_k
    np.testing.assert_allclose(rs["sum_rate"], rs["rate_c"] + rs["rate_p"].sum(),
                               rtol=1e-12)
    # sum_k per-user = R_sum because the common split sums to one
    # (float32 softmax logits -> loosened tolerance).
    np.testing.assert_allclose(rs["per_user"].sum(), rs["sum_rate"], rtol=1e-6)
    assert rs["rate_c"] >= 0 and np.all(rs["rate_p"] >= 0)


def test_analytical_fallback_when_direct_link_vanishes():
    env = StarRisRsmaEnv(base_env_cfg(num_users=2, num_users_reflection=1), seed=1)
    env.reset(seed=2)
    # Kill the direct links: the fallback must align cascaded terms to a
    # zero-phase reference (coherent combining), not use the phase of ~0.
    env._h_d = np.zeros((env.K, env.M), dtype=np.complex128)
    phi_r, phi_t = env._analytical_phases()
    assert np.all(np.isfinite(phi_r)) and np.all(np.isfinite(phi_t))
    # With the fallback phases, each per-element antenna-summed cascade term of
    # the reference T user is real positive -> the total cascade magnitude
    # equals the sum of per-element moduli (coherent combining).
    k_T = env.K_r  # only T user
    terms = np.conj(env._g[k_T]) * np.exp(1j * phi_t) * np.sum(env._G, axis=1)
    np.testing.assert_allclose(np.abs(terms.sum()), np.abs(terms).sum(), rtol=1e-9)


def test_qos_lambda_projection():
    env = StarRisRsmaEnv(base_env_cfg(), seed=1)
    env.set_qos_lambda_vec(np.array([-1.0, 5.0, 100.0, 0.0]))
    lam = env.qos_lambda_vec
    assert lam[0] == 0.0
    assert lam[1] == 5.0
    assert lam[2] == env.dual_lambda_max
    env.set_qos_lambda(3.0)
    np.testing.assert_allclose(env.qos_lambda_vec, 3.0)
    assert abs(env.qos_lambda - 3.0) < 1e-12
