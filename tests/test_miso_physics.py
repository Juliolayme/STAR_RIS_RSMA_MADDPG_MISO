import numpy as np
import yaml

from env.star_ris_env import StarRisRsmaEnv


def _cfg():
    with open('config/smoke.yaml', encoding='utf-8') as f:
        return yaml.safe_load(f)['env']


def test_miso_shapes_and_power_projection():
    env = StarRisRsmaEnv(_cfg(), seed=17)
    env.reset()
    actions = [np.linspace(-0.8, 0.8, d, dtype=np.float32) for d in env.act_dims]
    decoded = env._decode_action(actions)
    assert decoded['W_c'].shape == (env.M,)
    assert decoded['W_k'].shape == (env.K, env.M)
    total = np.sum(np.abs(decoded['W_c']) ** 2) + np.sum(np.abs(decoded['W_k']) ** 2)
    assert np.isclose(total, env.p_max, rtol=1e-8, atol=1e-12)


def test_miso_interference_depends_on_beam_direction():
    env = StarRisRsmaEnv(_cfg(), seed=23)
    env.reset()
    H = env._h_eff.copy()
    powers = np.full(env.K, env.p_max / (2 * env.K))
    pc = env.p_max / 2
    split = np.ones(env.K) / env.K

    dirs = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), env.eps)
    wk_matched = np.sqrt(powers)[:, None] * dirs
    wc = np.sqrt(pc) * np.sum(dirs, axis=0) / max(np.linalg.norm(np.sum(dirs, axis=0)), env.eps)
    rate_matched = env._rsma_rates(H, pc, powers, split, wc, wk_matched)['sum_rate']

    wk_bad = np.roll(wk_matched, shift=1, axis=0)
    rate_bad = env._rsma_rates(H, pc, powers, split, wc, wk_bad)['sum_rate']
    assert not np.isclose(rate_matched, rate_bad)


def test_effective_channel_matches_direct_signal_oracle():
    cfg = _cfg()
    cfg.update({
        "num_bs_antennas": 2,
        "num_users": 1,
        "num_users_reflection": 1,
        "num_ris_elements": 1,
        "direct_block_T": False,
    })
    env = StarRisRsmaEnv(cfg, seed=5)
    env.reset(seed=6)

    env._h_d = np.array([[0.7 + 0.2j, -0.3 + 0.5j]], dtype=np.complex128)
    env._G = np.array([[0.4 - 0.6j, -0.2 + 0.9j]], dtype=np.complex128)
    env._g = np.array([[0.8 + 0.1j]], dtype=np.complex128)
    beta_r = np.array([0.36], dtype=np.float64)
    phi_r = np.array([0.37 * np.pi], dtype=np.float64)
    phi_t = np.array([0.0], dtype=np.float64)
    theta = np.sqrt(beta_r[0]) * np.exp(1j * phi_r[0])
    w = np.array([0.2 - 0.4j, 0.5 + 0.3j], dtype=np.complex128)

    h_eff = env._effective_channels(beta_r, phi_r, phi_t)
    via_h_eff = np.vdot(h_eff[0], w)
    direct_signal = np.vdot(env._h_d[0], w) + np.conj(env._g[0, 0]) * theta * (env._G[0] @ w)
    np.testing.assert_allclose(via_h_eff, direct_signal, rtol=1e-12, atol=1e-12)

    rates = env._rsma_rates(
        h_eff,
        P_c=float(np.vdot(w, w).real),
        P_k=np.array([0.0]),
        common_split=np.array([1.0]),
        W_c=w,
        W_k=np.zeros((1, 2), dtype=np.complex128),
    )
    expected_sinr = abs(direct_signal) ** 2 / (env.sigma2 + env.eps)
    np.testing.assert_allclose(rates["sinr_c"][0], expected_sinr, rtol=1e-12, atol=1e-12)


def test_common_split_can_approach_simplex_boundary():
    env = StarRisRsmaEnv(_cfg(), seed=11)
    env.reset(seed=12)
    actions = [np.zeros(d, dtype=np.float32) for d in env.act_dims]
    field = next(f for f in env.action_schema()["agents"][0]
                 if f["name"] == "common_split_logits")
    actions[0][field["start"]:field["stop"]] = -1.0
    actions[0][field["start"]] = 1.0
    split = env._decode_action(actions)["common_split"]
    assert split[0] > 0.999
    assert split[1:].sum() < 0.001

def test_analytical_phase_aligns_nonzero_direct_received_signal():
    cfg = _cfg()
    cfg.update({
        "num_bs_antennas": 2,
        "num_users": 1,
        "num_users_reflection": 1,
        "num_ris_elements": 3,
        "direct_block_T": False,
    })
    env = StarRisRsmaEnv(cfg, seed=41)
    env.reset(seed=42)
    env._h_d = np.array([[0.7 + 0.8j, -0.25 + 0.35j]], dtype=np.complex128)
    env._G = np.array([
        [0.4 - 0.6j, -0.2 + 0.9j],
        [0.1 + 0.8j, 0.7 - 0.3j],
        [-0.5 + 0.2j, 0.6 + 0.4j],
    ], dtype=np.complex128)
    env._g = np.array([[0.8 + 0.1j, -0.3 + 0.6j, 0.4 - 0.7j]],
                      dtype=np.complex128)

    phi_r, _ = env._analytical_phases()
    q = np.ones(env.M, dtype=np.complex128) / np.sqrt(env.M)
    direct = np.vdot(env._h_d[0], q)
    terms = np.conj(env._g[0]) * np.exp(1j * phi_r) * (env._G @ q)
    aligned = terms * np.conj(direct)
    np.testing.assert_allclose(aligned.imag, 0.0, atol=1e-12, rtol=1e-12)
    assert np.all(aligned.real > 0.0)


def test_global_critic_state_contains_each_physical_feature_once():
    env = StarRisRsmaEnv(_cfg(), seed=91)
    env.reset(seed=92)
    spec = env.spec()
    assert env.global_state().shape == (spec.global_state_dim,)
    assert spec.global_state_dim == env.single_agent_obs_dim
    assert spec.global_state_dim < sum(spec.obs_dims)
