"""PPO with clipped surrogate objective and GAE-lambda, continuous (tanh-squashed Gaussian) policy."""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.actor import StochasticActor
from networks.critic import ValueNet
from utils import ObservationNormalizer


def _to_t(x, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


class _RolloutBuffer:
    """On-policy rollout with separate bootstrap and episode-boundary masks."""
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.cap = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.next_values = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.episode_ends = np.zeros(capacity, dtype=np.float32)
        self.adv = np.zeros(capacity, dtype=np.float32)
        self.ret = np.zeros(capacity, dtype=np.float32)
        self.size = 0

    def add(self, o, a, lp, r, v, terminated, episode_end, next_value):
        i = self.size
        assert i < self.cap, "Rollout buffer overflow."
        self.obs[i] = o
        self.actions[i] = a
        self.log_probs[i] = lp
        self.rewards[i] = r
        self.values[i] = v
        self.next_values[i] = float(next_value)
        self.terminated[i] = float(terminated)
        self.episode_ends[i] = float(episode_end)
        self.size += 1

    def reset(self):
        self.size = 0

    def compute_gae(self, gamma: float, lam: float):
        """Bootstrap at time limits, but never recurse across a reset."""
        adv = 0.0
        for t in reversed(range(self.size)):
            bootstrap_mask = 1.0 - self.terminated[t]
            continuation_mask = 1.0 - self.episode_ends[t]
            delta = (self.rewards[t]
                     + gamma * self.next_values[t] * bootstrap_mask
                     - self.values[t])
            adv = delta + gamma * lam * continuation_mask * adv
            self.adv[t] = adv
        self.ret[:self.size] = self.adv[:self.size] + self.values[:self.size]

    def get(self):
        n = self.size
        adv = self.adv[:n]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return {
            "obs": self.obs[:n],
            "actions": self.actions[:n],
            "log_probs": self.log_probs[:n],
            "returns": self.ret[:n],
            "advantages": adv,
        }


class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int],
                 ppo_cfg: dict, net_cfg: dict, device: str = "cpu", seed: int = 0):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # PPO traditionally uses tanh activations and *no* LayerNorm — keep that here.
        self.actor = StochasticActor(obs_dim, act_dim, hidden_sizes,
                                     activation="tanh", layer_norm=False,
                                     ortho=net_cfg.get("ortho_init", True)).to(self.device)
        self.critic = ValueNet(obs_dim, hidden_sizes,
                               activation="tanh", layer_norm=False,
                               ortho=net_cfg.get("ortho_init", True)).to(self.device)
        self.lr = float(ppo_cfg["lr"])
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()),
                                    lr=self.lr)

        self.gamma = float(ppo_cfg["gamma"])
        self.lam = float(ppo_cfg["gae_lambda"])
        self.clip_eps = float(ppo_cfg["clip_eps"])
        self.vf_coef = float(ppo_cfg["vf_coef"])
        self.ent_coef = float(ppo_cfg["ent_coef"])
        self.epochs = int(ppo_cfg["epochs"])
        self.minibatch_size = int(ppo_cfg["minibatch_size"])
        self.rollout_length = int(ppo_cfg["rollout_length"])
        self.grad_clip = float(ppo_cfg["grad_clip"])
        self.target_kl = float(ppo_cfg.get("target_kl", 0.0))

        self.rollout = _RolloutBuffer(self.rollout_length, obs_dim, act_dim)
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.obs_norm: ObservationNormalizer | None = None
        # Exact normalized observation used by the most recent select_action.
        self.last_norm_obs: np.ndarray | None = None

    # -------------------------------------------------- normalization
    def attach_obs_normalizer(self, norm: ObservationNormalizer) -> None:
        self.obs_norm = norm

    def freeze_normalizers(self) -> None:
        if self.obs_norm is not None:
            self.obs_norm.freeze()

    def _norm(self, obs: np.ndarray, update: bool) -> np.ndarray:
        if self.obs_norm is None:
            return np.asarray(obs, dtype=np.float32)
        return self.obs_norm(obs, update=update)

    def _norm_batch(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_norm is None:
            return np.asarray(obs, dtype=np.float32)
        return self.obs_norm.normalize_batch(obs)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, explore: bool = True):
        """Select an action from a RAW observation.

        The observation is normalized ONCE here; the exact normalized vector
        used to compute log_prob/value is cached in `self.last_norm_obs` so the
        training loop can store it and learn() can reuse it WITHOUT
        re-normalizing (item 2 reviewer fix: on-policy old_log_prob/value must
        match the vector actually fed to the network at collection time, even
        though the running statistics keep changing during warmup).
        """
        obs_n = self._norm(obs, update=explore)
        self.last_norm_obs = obs_n.astype(np.float32)
        obs_t = _to_t(obs_n, self.device).unsqueeze(0)
        squashed, log_prob, _ = self.actor.act(obs_t, deterministic=not explore)
        value = self.critic(obs_t).cpu().numpy()[0]
        return squashed.cpu().numpy()[0].astype(np.float32), float(log_prob.item()), float(value)

    def store(self, norm_obs, a, lp, r, v, *, terminated: bool,
              episode_end: bool, next_value: float):
        """Store exact behavior input and explicit boundary semantics."""
        self.rollout.add(norm_obs, a, lp, r, v, terminated,
                         episode_end, next_value)

    def buffer_full(self) -> bool:
        return self.rollout.size >= self.rollout_length

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> float:
        obs_t = _to_t(self._norm(obs, update=False), self.device).unsqueeze(0)
        return float(self.critic(obs_t).cpu().item())

    @torch.no_grad()
    def rollout_logprob_consistency(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (stored_old_log_probs, recomputed_log_probs) on the current
        rollout using the CURRENT (pre-update) actor. With the item-2 fix these
        must match to numerical precision. Used by tests."""
        n = self.rollout.size
        if n == 0:
            return np.zeros(0), np.zeros(0)
        obs_t = _to_t(self.rollout.obs[:n], self.device)
        act_t = _to_t(self.rollout.actions[:n], self.device)
        new_lp, _ = self.actor.log_prob(obs_t, act_t)
        return self.rollout.log_probs[:n].copy(), new_lp.cpu().numpy()

    def learn(self, last_value: float | None = None) -> dict:
        if self.rollout.size == 0:
            return {}
        self.rollout.compute_gae(self.gamma, self.lam)
        data = self.rollout.get()
        # Rollout already holds NORMALIZED observations -- do NOT re-normalize.
        obs_t = _to_t(data["obs"], self.device)
        act_t = _to_t(data["actions"], self.device)
        oldlp_t = _to_t(data["log_probs"], self.device)
        ret_t = _to_t(data["returns"], self.device)
        adv_t = _to_t(data["advantages"], self.device)

        n = obs_t.shape[0]
        idxs = np.arange(n)
        info = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clipfrac": 0.0}
        n_updates = 0
        early_stopped = False

        for epoch in range(self.epochs):
            self._rng.shuffle(idxs)
            for start in range(0, n, self.minibatch_size):
                mb = idxs[start: start + self.minibatch_size]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=self.device)
                mb_obs = obs_t[mb_t]
                mb_act = act_t[mb_t]
                mb_oldlp = oldlp_t[mb_t]
                mb_ret = ret_t[mb_t]
                mb_adv = adv_t[mb_t]

                new_lp, entropy = self.actor.log_prob(mb_obs, mb_act)
                value = self.critic(mb_obs)

                ratio = torch.exp(new_lp - mb_oldlp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, mb_ret)
                ent = entropy.mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * ent

                if not torch.isfinite(loss):
                    continue

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
                self.opt.step()

                with torch.no_grad():
                    approx_kl = (mb_oldlp - new_lp).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()
                info["policy_loss"] += float(policy_loss.item())
                info["value_loss"] += float(value_loss.item())
                info["entropy"] += float(ent.item())
                info["kl"] += approx_kl
                info["clipfrac"] += clipfrac
                n_updates += 1

            if self.target_kl > 0 and n_updates > 0 and (info["kl"] / n_updates) > 1.5 * self.target_kl:
                early_stopped = True
                break

        for k in info:
            info[k] = info[k] / max(n_updates, 1)
        info["early_stopped"] = float(early_stopped)
        self.rollout.reset()
        return info

    # -------------------------------------------------- checkpoints
    def weights_state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}

    def load_weights_state_dict(self, s: dict) -> None:
        self.actor.load_state_dict(s["actor"])
        self.critic.load_state_dict(s["critic"])

    def train_state_dict(self) -> dict:
        return {
            "optim": {"opt": self.opt.state_dict()},
            "rng": self._rng.bit_generator.state,
            "obs_norm": None if self.obs_norm is None else self.obs_norm.state_dict(),
        }

    def load_train_state_dict(self, state: dict) -> None:
        self.opt.load_state_dict(state["optim"]["opt"])
        self._rng.bit_generator.state = state["rng"]
        if state.get("obs_norm") is not None:
            if self.obs_norm is None:
                self.obs_norm = ObservationNormalizer(shape=(self.obs_dim,))
            self.obs_norm.load_state_dict(state["obs_norm"])

    def save_inference(self, path: str, extra_meta: dict | None = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "kind": "single_agent_inference",
            "weights": self.weights_state_dict(),
            "obs_norm": None if self.obs_norm is None else self.obs_norm.state_dict(),
            "obs_dim": self.obs_dim, "act_dim": self.act_dim,
            "meta": extra_meta or {},
        }, path)

    def load_inference(self, path: str):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.load_weights_state_dict(payload["weights"])
        if payload.get("obs_norm") is not None:
            if self.obs_norm is None:
                self.obs_norm = ObservationNormalizer(shape=(self.obs_dim,))
            self.obs_norm.load_state_dict(payload["obs_norm"])

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.weights_state_dict(), path)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        s = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(s, dict) and s.get("kind") == "single_agent_inference":
            self.load_weights_state_dict(s["weights"])
        else:
            self.load_weights_state_dict(s)
