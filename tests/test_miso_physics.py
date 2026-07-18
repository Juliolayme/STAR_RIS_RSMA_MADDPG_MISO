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
