"""Running statistics for observation / reward normalization."""
from __future__ import annotations
import numpy as np


class RunningMeanStd:
    """Welford's online algorithm for stable running mean / variance."""
    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.mean.shape):
            x = x[None, ...]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, b_mean, b_var, b_count):
        delta = b_mean - self.mean
        tot = self.count + b_count
        new_mean = self.mean + delta * b_count / tot
        m_a = self.var * self.count
        m_b = b_var * b_count
        M2 = m_a + m_b + (delta ** 2) * self.count * b_count / tot
        self.mean = new_mean
        self.var = M2 / tot
        self.count = tot

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.var, 1e-8))


class ObservationNormalizer:
    """Normalize observations using a RunningMeanStd, clipped to a range.

    Freeze semantics (P1-2 reviewer fix): statistics are updated only until
    `freeze()` is called; afterwards every call maps the same raw observation
    to the same normalized output, so replay batches sampled at different
    times live in one consistent coordinate system. Replay buffers must store
    RAW observations; agents normalize at action-selection and at batch-sample
    time using this (eventually frozen) normalizer.
    """
    def __init__(self, shape: tuple[int, ...], clip: float = 10.0):
        self.rms = RunningMeanStd(shape)
        self.clip = clip
        self.enabled = True
        self.frozen = False

    def freeze(self) -> None:
        self.frozen = True

    def __call__(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        if update and self.enabled and not self.frozen:
            self.rms.update(obs)
        z = (obs - self.rms.mean) / self.rms.std
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def normalize_batch(self, obs: np.ndarray) -> np.ndarray:
        """Normalize a batch WITHOUT updating statistics (learn-time path)."""
        obs = np.asarray(obs, dtype=np.float64)
        z = (obs - self.rms.mean) / self.rms.std
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.rms.mean.copy(), "var": self.rms.var.copy(),
                "count": float(self.rms.count), "clip": float(self.clip),
                "enabled": bool(self.enabled), "frozen": bool(self.frozen)}

    def load_state_dict(self, state: dict) -> None:
        self.rms.mean = np.asarray(state["mean"], dtype=np.float64)
        self.rms.var = np.asarray(state["var"], dtype=np.float64)
        self.rms.count = float(state["count"])
        self.clip = float(state.get("clip", self.clip))
        self.enabled = bool(state.get("enabled", True))
        self.frozen = bool(state.get("frozen", False))

    def save(self, path: str) -> None:
        """Persist running statistics so deterministic eval can be reproduced after
        --skip-train. Saves mean, var, count, clip, enabled, frozen to .npz."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, mean=self.rms.mean, var=self.rms.var,
                 count=np.array([self.rms.count], dtype=np.float64),
                 clip=np.array([self.clip], dtype=np.float64),
                 enabled=np.array([1 if self.enabled else 0], dtype=np.int8),
                 frozen=np.array([1 if self.frozen else 0], dtype=np.int8))

    def load(self, path: str) -> None:
        """Restore running statistics from .npz file (matched by `save`)."""
        import os
        if not os.path.exists(path):
            return
        d = np.load(path)
        self.rms.mean = d["mean"]
        self.rms.var = d["var"]
        self.rms.count = float(d["count"][0])
        if "clip" in d.files:
            self.clip = float(d["clip"][0])
        if "enabled" in d.files:
            self.enabled = bool(d["enabled"][0])
        if "frozen" in d.files:
            self.frozen = bool(d["frozen"][0])


class RewardScaler:
    """Scales rewards by running std (no centering, Engstrom et al. 2020)."""
    def __init__(self, gamma: float = 0.99):
        self.rms = RunningMeanStd(shape=())
        self.gamma = gamma
        self.returns = 0.0

    def __call__(self, reward: float) -> float:
        self.returns = self.returns * self.gamma + reward
        self.rms.update(np.array([self.returns]))
        return float(reward / max(float(self.rms.std), 1e-6))

    def reset(self):
        self.returns = 0.0
