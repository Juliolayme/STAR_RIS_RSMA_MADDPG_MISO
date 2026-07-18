"""Experiment-level training checkpoints (kept OUTSIDE the agent classes).

Two checkpoint kinds exist in this project:

1. Inference checkpoints (`agent.save_inference`): weights + frozen
   normalizer statistics + metadata. Enough for deterministic evaluation and
   deployment. NOT enough to resume training.

2. `TrainingCheckpoint` (this module): captured ONLY at episode boundaries.
   Contains everything needed for exact resume:
     - agent weights, optimizer states, OU noise state, global/learn steps,
       agent RNG, observation normalizers (+ freeze state);
     - replay buffer contents;
     - trainer state: episode index, dual multipliers EMA, best-validation
       selector state, training history, total env steps;
     - environment state: channel small-scale state, prev-action bookkeeping,
       RNG streams (geometry/channel/misc);
     - global RNG states (python / numpy / torch / CUDA);
     - the full effective config.

   The "training continues identically after loading" guarantee holds only
   for this checkpoint kind (it includes the replay buffer). It is verified
   by tests/test_checkpoint.py on a small replay fixture.
"""
from __future__ import annotations
import json
import os
import random

import numpy as np
import torch


def _save_replay_sidecar(path: str, replay: dict) -> None:
    """Persist replay (already sliced to [:size]) as a compressed .npz sidecar.

    Keeps the main torch checkpoint small and stores the transition arrays
    compressed. Per-agent lists are flattened to obs_0, obs_1, ... keys."""
    arrays: dict[str, np.ndarray] = {}
    meta = {"capacity": int(replay["capacity"]), "idx": int(replay["idx"]),
            "size": int(replay["size"])}
    if isinstance(replay["obs"], list):          # multi-agent
        meta["multi_agent"] = True
        meta["n_agents"] = len(replay["obs"])
        for group in ("obs", "next_obs", "actions", "rewards"):
            for i, arr in enumerate(replay[group]):
                arrays[f"{group}_{i}"] = arr
    else:                                        # single-agent
        meta["multi_agent"] = False
        for group in ("obs", "next_obs", "actions", "rewards"):
            arrays[group] = replay[group]
    arrays["dones"] = replay["dones"]
    # Optional reward-recomputation arrays (item 1).
    for key in ("base_rewards", "c_gaps"):
        if key in replay:
            arrays[key] = replay[key]
    np.savez_compressed(path, __meta__=np.frombuffer(
        json.dumps(meta).encode(), dtype=np.uint8), **arrays)


def _load_replay_sidecar(path: str) -> dict:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(data["__meta__"]).decode())
    out = {"capacity": meta["capacity"], "idx": meta["idx"], "size": meta["size"]}
    if meta.get("multi_agent", True):
        n = meta["n_agents"]
        for group in ("obs", "next_obs", "actions", "rewards"):
            out[group] = [data[f"{group}_{i}"] for i in range(n)]
    else:
        for group in ("obs", "next_obs", "actions", "rewards"):
            out[group] = data[group]
    out["dones"] = data["dones"]
    for key in ("base_rewards", "c_gaps"):
        if key in data.files:
            out[key] = data[key]
    return out


class TrainingCheckpoint:
    def __init__(self, payload: dict):
        self.payload = payload

    # ------------------------------------------------------------ properties
    @property
    def trainer_state(self) -> dict:
        return self.payload["trainer_state"]

    @property
    def config(self) -> dict:
        return self.payload["config"]

    # ------------------------------------------------------------ capture
    @staticmethod
    def _global_rng_state() -> dict:
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_global_rng_state(state: dict) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8))
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([torch.as_tensor(s, dtype=torch.uint8)
                                          for s in state["cuda"]])

    @classmethod
    def capture_maddpg(cls, agent, env, dual, selector,
                       trainer_state: dict, cfg: dict) -> "TrainingCheckpoint":
        payload = {
            "kind": "training_checkpoint_maddpg",
            "weights": agent.weights_state_dict(),
            "agent_train_state": agent.train_state_dict(),
            "replay": agent.replay_state(),
            "env_state": env.get_state(),
            "dual_state": dual.state_dict(),
            "selector_state": selector.state_dict(),
            "trainer_state": dict(trainer_state),
            "global_rng": cls._global_rng_state(),
            "config": cfg,
        }
        return cls(payload)

    def restore_maddpg(self, agent, env, dual, selector) -> None:
        p = self.payload
        assert p["kind"] == "training_checkpoint_maddpg", p["kind"]
        agent.load_weights_state_dict(p["weights"])
        agent.load_train_state_dict(p["agent_train_state"])
        agent.load_replay_state(p["replay"])
        env.set_state(p["env_state"])
        dual.load_state_dict(p["dual_state"])
        selector.load_state_dict(p["selector_state"])
        self._restore_global_rng_state(p["global_rng"])

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = dict(self.payload)
        replay = payload.pop("replay", None)
        torch.save(payload, path)
        if replay is not None:
            _save_replay_sidecar(path + ".replay.npz", replay)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "TrainingCheckpoint":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        sidecar = path + ".replay.npz"
        if os.path.exists(sidecar):
            payload["replay"] = _load_replay_sidecar(sidecar)
        return cls(payload)
