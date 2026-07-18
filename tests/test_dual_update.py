"""Projected dual-gradient update on the signed constraint gap."""
from __future__ import annotations
import numpy as np

from env import StarRisRsmaEnv
from experiments.train import DualUpdater
from conftest import base_env_cfg


def _env_and_dual(**cfg_over):
    cfg = base_env_cfg(**cfg_over)
    env = StarRisRsmaEnv(cfg, seed=1)
    dual = DualUpdater(env.K, cfg)
    return env, dual


def test_lambda_increases_under_violation():
    env, dual = _env_and_dual(dual_ema=0.0)   # no smoothing -> direct response
    lam0 = env.qos_lambda_vec.copy()
    c = np.full(env.K, 0.2)                   # every user violates by 0.2
    dual.update(env, c)
    assert np.all(env.qos_lambda_vec > lam0)


def test_lambda_decreases_with_slack_signed_gap():
    env, dual = _env_and_dual(dual_ema=0.0)
    env.set_qos_lambda_vec(np.full(env.K, 5.0))
    c = np.full(env.K, -0.5)                  # every user has 0.5 slack
    dual.update(env, c)
    assert np.all(env.qos_lambda_vec < 5.0)


def test_lambda_nonnegative_and_capped():
    env, dual = _env_and_dual(dual_ema=0.0, dual_lr=100.0, dual_lambda_max=7.0)
    env.set_qos_lambda_vec(np.zeros(env.K))
    dual.update(env, np.full(env.K, -10.0))   # huge slack -> projection at 0
    assert np.all(env.qos_lambda_vec == 0.0)
    dual2 = DualUpdater(env.K, base_env_cfg(dual_ema=0.0, dual_lr=100.0,
                                            dual_lambda_max=7.0))
    dual.ema[:] = 0.0
    dual2.update(env, np.full(env.K, 10.0))   # huge violation -> cap at max
    assert np.all(env.qos_lambda_vec == 7.0)


def test_lambda_stationary_at_zero_gap():
    env, dual = _env_and_dual(dual_ema=0.0)
    lam0 = env.qos_lambda_vec.copy()
    dual.update(env, np.zeros(env.K))
    np.testing.assert_allclose(env.qos_lambda_vec, lam0)


def test_lambda_vector_has_k_entries_and_is_per_user():
    env, dual = _env_and_dual(dual_ema=0.0)
    assert env.qos_lambda_vec.shape == (env.K,)
    c = np.zeros(env.K); c[0] = 0.3           # only user 0 violates
    lam0 = env.qos_lambda_vec.copy()
    dual.update(env, c)
    assert env.qos_lambda_vec[0] > lam0[0]
    np.testing.assert_allclose(env.qos_lambda_vec[1:], lam0[1:])


def test_two_stage_freeze_holds_lambda():
    env, dual = _env_and_dual(dual_ema=0.0,
                              two_stage_dual_freeze_fraction=0.5)
    lam0 = env.qos_lambda_vec.copy()
    # Past the freeze point: no update even under violation.
    dual.update(env, np.full(env.K, 1.0), ep=8, total_episodes=10)
    np.testing.assert_allclose(env.qos_lambda_vec, lam0)
    # Before the freeze point: updates apply.
    dual.update(env, np.full(env.K, 1.0), ep=2, total_episodes=10)
    assert np.all(env.qos_lambda_vec > lam0)


def test_ema_smoothing():
    env, dual = _env_and_dual(dual_ema=0.9, dual_lr=1.0)
    env.set_qos_lambda_vec(np.zeros(env.K))
    dual.update(env, np.ones(env.K))
    # EMA after one step: 0.9*0 + 0.1*1 = 0.1 -> lambda = 0.1
    np.testing.assert_allclose(env.qos_lambda_vec, 0.1, rtol=1e-12)
