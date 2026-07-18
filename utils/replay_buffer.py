"""Replay buffers for single- and multi-agent algorithms.

Reward recomputation (item 1 reviewer fix)
------------------------------------------
Off-policy replay outlives many dual (lambda) updates, so a scalar reward
stored at collection time becomes stale: the critic would train on a mixture of
obsolete reward functions. To avoid this, each transition ALSO stores
  - base_reward: reward WITHOUT the linear dual term (already reward_scale-d),
  - c_gap: the signed per-user constraint gap c_k = R_min - R_k,
so the reward can be recomputed at sample time under the CURRENT lambda:
  reward_current = clip(base_reward - reward_scale * sum_k lambda_cur_k * c_k).
`sample_recomputed(...)` performs this reconstruction.
"""
from __future__ import annotations
import numpy as np
from typing import Sequence


def _recompute_reward(base_reward: np.ndarray, c_gaps: np.ndarray,
                      lambda_vec, reward_scale: float, reward_clip: float
                      ) -> np.ndarray:
    """reward = clip(base - reward_scale * (c_gaps @ lambda), -clip, clip)."""
    lam = np.asarray(lambda_vec, dtype=np.float32).reshape(-1)
    dual = reward_scale * (c_gaps @ lam)                 # (batch,)
    r = base_reward.reshape(-1) - dual
    return np.clip(r, -reward_clip, reward_clip).reshape(-1, 1).astype(np.float32)


class ReplayBuffer:
    """Standard FIFO replay buffer for single-agent off-policy algorithms."""
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, n_users: int = 0):
        self.capacity = int(capacity)
        self.n_users = int(n_users)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.base_rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.c_gaps = np.zeros((self.capacity, max(self.n_users, 1)), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, obs, action, reward, next_obs, done,
            base_reward=None, c_gap=None):
        i = self.idx
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i, 0] = reward
        self.base_rewards[i, 0] = float(reward if base_reward is None else base_reward)
        if c_gap is not None and self.n_users > 0:
            self.c_gaps[i, :] = np.asarray(c_gap, dtype=np.float32).reshape(-1)
        self.next_obs[i] = next_obs
        self.dones[i, 0] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None,
               lambda_vec=None, reward_scale: float = 1.0, reward_clip: float = 1e9):
        rng = rng or np.random.default_rng()
        idxs = rng.integers(0, self.size, size=batch_size)
        if lambda_vec is not None and self.n_users > 0:
            rewards = _recompute_reward(self.base_rewards[idxs], self.c_gaps[idxs],
                                        lambda_vec, reward_scale, reward_clip)
        else:
            rewards = self.rewards[idxs].copy()
        return (
            self.obs[idxs].copy(),
            self.actions[idxs].copy(),
            rewards,
            self.next_obs[idxs].copy(),
            self.dones[idxs].copy(),
        )


class MAReplayBuffer:
    """Replay buffer storing per-agent observations and actions for CTDE algorithms.

    The reward is cooperative (shared). It is reconstructed at sample time from
    a single stored base_reward + c_gap vector (item 1), not from stale per-agent
    scalar rewards.
    """
    def __init__(self, capacity: int, obs_dims: Sequence[int], act_dims: Sequence[int],
                 n_users: int = 0):
        assert len(obs_dims) == len(act_dims)
        self.capacity = int(capacity)
        self.n_agents = len(obs_dims)
        self.n_users = int(n_users)
        self.obs = [np.zeros((self.capacity, d), dtype=np.float32) for d in obs_dims]
        self.next_obs = [np.zeros((self.capacity, d), dtype=np.float32) for d in obs_dims]
        self.actions = [np.zeros((self.capacity, d), dtype=np.float32) for d in act_dims]
        self.rewards = [np.zeros((self.capacity, 1), dtype=np.float32) for _ in range(self.n_agents)]
        self.base_rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.c_gaps = np.zeros((self.capacity, max(self.n_users, 1)), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, obs_list, action_list, reward_list, next_obs_list, done,
            base_reward=None, c_gap=None):
        i = self.idx
        for a in range(self.n_agents):
            self.obs[a][i] = obs_list[a]
            self.next_obs[a][i] = next_obs_list[a]
            self.actions[a][i] = action_list[a]
            self.rewards[a][i, 0] = float(reward_list[a])
        self.base_rewards[i, 0] = float(reward_list[0] if base_reward is None else base_reward)
        if c_gap is not None and self.n_users > 0:
            self.c_gaps[i, :] = np.asarray(c_gap, dtype=np.float32).reshape(-1)
        self.dones[i, 0] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None,
               lambda_vec=None, reward_scale: float = 1.0, reward_clip: float = 1e9):
        rng = rng or np.random.default_rng()
        idxs = rng.integers(0, self.size, size=batch_size)
        obs = [o[idxs].copy() for o in self.obs]
        next_obs = [o[idxs].copy() for o in self.next_obs]
        actions = [a[idxs].copy() for a in self.actions]
        if lambda_vec is not None and self.n_users > 0:
            shared = _recompute_reward(self.base_rewards[idxs], self.c_gaps[idxs],
                                       lambda_vec, reward_scale, reward_clip)
            rewards = [shared.copy() for _ in range(self.n_agents)]
        else:
            rewards = [r[idxs].copy() for r in self.rewards]
        dones = self.dones[idxs].copy()
        return obs, actions, rewards, next_obs, dones
