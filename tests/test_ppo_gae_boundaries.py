"""PPO time-limit bootstrap must not leak GAE across episode resets."""
import numpy as np

from algorithms.ppo.agent import _RolloutBuffer


def test_gae_uses_distinct_bootstrap_and_continuation_masks():
    buf = _RolloutBuffer(capacity=4, obs_dim=1, act_dim=1)
    # Episode A ends by time limit: bootstrap from final observation, but do not
    # recurse into the reset state/advantage of episode B.
    buf.add(np.array([0.0]), np.array([0.0]), 0.0,
            r=1.0, v=2.0, terminated=False, episode_end=True,
            next_value=10.0)
    buf.add(np.array([1.0]), np.array([0.0]), 0.0,
            r=100.0, v=3.0, terminated=False, episode_end=True,
            next_value=20.0)
    buf.compute_gae(gamma=1.0, lam=1.0)
    assert buf.adv[0] == np.float32(9.0)
    assert buf.adv[1] == np.float32(117.0)


def test_true_terminal_does_not_bootstrap():
    buf = _RolloutBuffer(capacity=2, obs_dim=1, act_dim=1)
    buf.add(np.array([0.0]), np.array([0.0]), 0.0,
            r=4.0, v=1.5, terminated=True, episode_end=True,
            next_value=999.0)
    buf.compute_gae(gamma=0.99, lam=0.95)
    assert buf.adv[0] == np.float32(2.5)
