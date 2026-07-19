import numpy as np

from env.star_ris_env import StarRisRsmaEnv
from conftest import base_env_cfg


def _zero_actions(env):
    return [np.zeros(d, dtype=np.float32) for d in env.act_dims]


def test_structured_bs_reduces_action_dimension_and_zero_is_strong_prior():
    env = StarRisRsmaEnv(base_env_cfg(), seed=10)
    env.reset(seed=11)
    assert env.act_dims[0] == 3 * env.K + 2
    assert env.act_dims[0] < 2 * env.M * (env.K + 1) + env.K
    decoded = env._decode_action(_zero_actions(env))
    np.testing.assert_allclose(
        decoded["power_weights"], np.ones(env.K + 1) / (env.K + 1), atol=1e-12)
    np.testing.assert_allclose(
        decoded["common_split"], np.ones(env.K) / env.K, atol=1e-12)
    assert decoded["bs_action_mode"] == "structured_rzf"
    assert np.isclose(decoded["bs_private_rzf_mix"], env.bs_rzf_mix_prior)


def test_structured_beamformers_obey_power_and_are_finite():
    env = StarRisRsmaEnv(base_env_cfg(), seed=12)
    env.reset(seed=13)
    rng = np.random.default_rng(14)
    actions = [rng.uniform(-1, 1, d).astype(np.float32) for d in env.act_dims]
    decoded = env._decode_action(actions)
    total = np.sum(np.abs(decoded["W_c"]) ** 2) + np.sum(np.abs(decoded["W_k"]) ** 2)
    assert np.isclose(total, env.p_max, rtol=1e-10, atol=1e-12)
    assert np.all(np.isfinite(decoded["W_c"]))
    assert np.all(np.isfinite(decoded["W_k"]))
    assert np.all(decoded["power_weights"] >= env.bs_min_stream_power_fraction - 1e-12)


def test_rzf_convention_suppresses_cross_user_interference():
    cfg = base_env_cfg(num_bs_antennas=4, num_users=4, num_users_reflection=2,
                       bs_rzf_regularization=1e-4, bs_rzf_mix_prior=1.0,
                       bs_rzf_mix_span=0.0)
    env = StarRisRsmaEnv(cfg, seed=15)
    env.reset(seed=16)
    # Full-rank deterministic channel under h_k^H w convention.
    H = np.array([
        [1.0 + .2j, .1 - .1j, .2 + .1j, .05j],
        [.1 + .1j, 1.0 - .1j, .05, .1j],
        [.2, .05j, .9 + .2j, .1],
        [.05, .1, .1j, 1.1 - .1j],
    ], dtype=np.complex128)
    dirs = env._rzf_directions(H)
    received = np.conj(H) @ dirs.T
    diag = np.abs(np.diag(received))
    off = np.abs(received - np.diag(np.diag(received)))
    assert diag.min() > 20.0 * off.max()


def test_clean_ablation_flags_change_only_requested_bs_component():
    cfg = base_env_cfg(phase_action_mode="absolute")
    env = StarRisRsmaEnv(cfg, seed=17)
    env.reset(seed=18)
    rng = np.random.default_rng(19)
    actions = [rng.uniform(-0.7, 0.7, d).astype(np.float32) for d in env.act_dims]
    base = env._decode_action(actions)

    env_eq = StarRisRsmaEnv({**cfg, "force_equal_stream_power": True}, seed=17)
    env_eq.reset(seed=18)
    eq = env_eq._decode_action(actions)
    np.testing.assert_allclose(eq["power_weights"], np.ones(env.K + 1)/(env.K+1))
    np.testing.assert_allclose(eq["common_split"], base["common_split"])
    assert np.isclose(eq["bs_private_rzf_mix"], base["bs_private_rzf_mix"])

    env_mrt = StarRisRsmaEnv({**cfg, "force_mrt_directions": True}, seed=17)
    env_mrt.reset(seed=18)
    mrt = env_mrt._decode_action(actions)
    np.testing.assert_allclose(mrt["power_weights"], base["power_weights"])
    np.testing.assert_allclose(mrt["common_split"], base["common_split"])
    assert mrt["bs_private_rzf_mix"] == 0.0

    env_cs = StarRisRsmaEnv({**cfg, "force_uniform_common_split": True}, seed=17)
    env_cs.reset(seed=18)
    cs = env_cs._decode_action(actions)
    np.testing.assert_allclose(cs["common_split"], np.ones(env.K)/env.K)
    np.testing.assert_allclose(cs["power_weights"], base["power_weights"])


def test_ao_grid_uses_explicit_shared_rzf_backbone():
    cfg = base_env_cfg()
    env = StarRisRsmaEnv(cfg, seed=20, ris_mode="ao_grid")
    env.reset(seed=21)
    decoded = env._decode_action(_zero_actions(env))
    assert decoded["bs_action_mode"] == "classical_rzf_grid"
    assert decoded["W_c"].shape == (env.M,)
    assert decoded["W_k"].shape == (env.K, env.M)
    total = (np.sum(np.abs(decoded["W_c"]) ** 2)
             + np.sum(np.abs(decoded["W_k"]) ** 2))
    assert np.isclose(total, env.p_max, rtol=1e-10, atol=1e-12)
    assert np.isclose(decoded["bs_private_rzf_mix"], env.bs_rzf_mix_prior)
