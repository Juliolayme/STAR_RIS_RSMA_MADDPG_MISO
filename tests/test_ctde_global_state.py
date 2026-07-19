"""Canonical centralized critic state regression tests."""
import numpy as np

from utils.replay_buffer import MAReplayBuffer


def test_ma_replay_roundtrips_explicit_global_state():
    buf = MAReplayBuffer(capacity=4, obs_dims=[2, 3, 1], act_dims=[1, 1, 1],
                         n_users=2, global_state_dim=5)
    obs = [np.ones(2), np.ones(3) * 2, np.ones(1) * 3]
    nxt = [x + 1 for x in obs]
    gs = np.arange(5, dtype=np.float32)
    ngs = gs + 10
    buf.add(obs, [np.zeros(1)] * 3, [1.0] * 3, nxt, 0.0,
            base_reward=1.0, c_gap=np.zeros(2),
            global_state=gs, next_global_state=ngs)
    sample = buf.sample(1, rng=np.random.default_rng(0),
                        include_global_state=True)
    np.testing.assert_array_equal(sample[-2][0], gs)
    np.testing.assert_array_equal(sample[-1][0], ngs)
