"""Raw-observation replay + frozen normalizer consistency (P1-2)."""
from __future__ import annotations
import numpy as np

from env import StarRisRsmaEnv
from algorithms import MADDPG
from utils import ObservationNormalizer
from conftest import base_env_cfg, full_cfg


def test_frozen_normalizer_is_deterministic():
    norm = ObservationNormalizer(shape=(4,))
    rng = np.random.default_rng(0)
    for _ in range(100):
        norm(rng.normal(size=4), update=True)
    norm.freeze()
    x = rng.normal(size=4)
    y1 = norm(x, update=True)    # update ignored after freeze
    for _ in range(50):
        norm(rng.normal(size=4), update=True)
    y2 = norm(x, update=True)
    np.testing.assert_allclose(y1, y2)
    np.testing.assert_allclose(norm.normalize_batch(x[None, :])[0], y1)


def test_replay_buffer_stores_raw_observations():
    cfg = full_cfg()
    env = StarRisRsmaEnv(cfg["env"], seed=1)
    spec = env.spec()
    agent = MADDPG(spec, hidden_sizes=[16, 16], maddpg_cfg=cfg["maddpg"],
                   net_cfg=cfg["networks"], device="cpu", seed=0)
    norms = [ObservationNormalizer(shape=(d,)) for d in spec.obs_dims]
    agent.attach_obs_normalizers(norms)

    env.reset(seed=2)
    raw = env.per_agent_observations()
    global_state = env.global_state()
    actions = agent.select_actions(raw, explore=True)
    env.step(actions)
    raw_next = env.per_agent_observations()
    next_global_state = env.global_state()
    agent.add_transition(raw, actions, 0.5, raw_next, 0.0,
                         global_state=global_state,
                         next_global_state=next_global_state)

    for i in range(agent.n_agents):
        stored = agent.buffer.obs[i][0]
        np.testing.assert_allclose(stored, raw[i], rtol=1e-6,
                                   err_msg="replay must store RAW observations")
        normalized = norms[i].normalize_batch(raw[i][None, :])[0]
        assert not np.allclose(stored, normalized), \
            "stored observation must not be pre-normalized"


def test_learn_uses_frozen_stats_consistently():
    """After freeze, select-time and learn-time normalization agree."""
    norm = ObservationNormalizer(shape=(3,))
    rng = np.random.default_rng(1)
    for _ in range(200):
        norm(rng.normal(loc=2.0, scale=3.0, size=3), update=True)
    norm.freeze()
    x = rng.normal(size=3)
    a = norm(x, update=False)
    b = norm.normalize_batch(np.stack([x, x]))
    np.testing.assert_allclose(a, b[0])
    np.testing.assert_allclose(a, b[1])


def test_state_dict_roundtrip_preserves_frozen_flag():
    norm = ObservationNormalizer(shape=(2,))
    norm(np.array([1.0, 2.0]), update=True)
    norm.freeze()
    st = norm.state_dict()
    norm2 = ObservationNormalizer(shape=(2,))
    norm2.load_state_dict(st)
    assert norm2.frozen
    x = np.array([0.3, -0.7])
    np.testing.assert_allclose(norm(x, update=False), norm2(x, update=False))
