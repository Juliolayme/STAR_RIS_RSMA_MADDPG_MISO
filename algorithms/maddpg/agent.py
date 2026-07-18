"""MADDPG: Multi-agent DDPG with centralized critics and decentralized actors (CTDE).

Observation-normalization ownership (P1-2 reviewer fix): the agent owns the
per-agent ObservationNormalizers. Callers pass RAW observations everywhere
(select_actions, add_transition); the replay buffer stores RAW observations,
and normalization happens exactly once -- inside select_actions() at rollout
time and inside learn() at batch-sample time. After the normalizers are frozen
(training driver calls freeze_normalizers()), a given raw observation always
maps to the same normalized vector, for training and evaluation alike.
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import Actor, CentralizedCritic, soft_update, hard_update
from utils import MAReplayBuffer, ObservationNormalizer
from .noise import OUNoise


def _to_t(x, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


class _PerAgent:
    """Holds actor, target actor, centralized critic, target critic, optimizers, and noise."""
    def __init__(self, obs_dim: int, act_dim: int,
                 total_obs_dim: int, total_act_dim: int,
                 hidden_sizes: list[int], cfg: dict, device: torch.device,
                 seed: int = 0):
        self.device = device
        self.act_dim = act_dim
        self.actor = Actor(obs_dim, act_dim, hidden_sizes,
                           activation=cfg.get("activation", "relu"),
                           layer_norm=cfg.get("layer_norm", True),
                           ortho=cfg.get("ortho_init", True)).to(device)
        self.actor_target = Actor(obs_dim, act_dim, hidden_sizes,
                                  activation=cfg.get("activation", "relu"),
                                  layer_norm=cfg.get("layer_norm", True),
                                  ortho=cfg.get("ortho_init", True)).to(device)
        hard_update(self.actor, self.actor_target)

        self.critic = CentralizedCritic(total_obs_dim, total_act_dim, hidden_sizes,
                                        activation=cfg.get("activation", "relu"),
                                        layer_norm=cfg.get("layer_norm", True),
                                        ortho=cfg.get("ortho_init", True)).to(device)
        self.critic_target = CentralizedCritic(total_obs_dim, total_act_dim, hidden_sizes,
                                               activation=cfg.get("activation", "relu"),
                                               layer_norm=cfg.get("layer_norm", True),
                                               ortho=cfg.get("ortho_init", True)).to(device)
        hard_update(self.critic, self.critic_target)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])

        self.noise = OUNoise(act_dim, sigma=cfg["noise_sigma_start"], seed=seed)


class MADDPG:
    def __init__(self, env_spec, hidden_sizes: list[int],
                 maddpg_cfg: dict, net_cfg: dict, device: str = "cpu", seed: int = 0,
                 n_users: int = 0, reward_scale: float = 1.0, reward_clip: float = 1e9):
        self.device = torch.device(device)
        self.cfg = maddpg_cfg
        self.n_agents = env_spec.n_agents
        self.obs_dims = env_spec.obs_dims
        self.act_dims = env_spec.act_dims
        self.total_obs_dim = int(sum(self.obs_dims))
        self.total_act_dim = int(sum(self.act_dims))

        cfg = {**net_cfg, **maddpg_cfg}
        self.agents = [
            _PerAgent(self.obs_dims[i], self.act_dims[i],
                      self.total_obs_dim, self.total_act_dim,
                      hidden_sizes, cfg, self.device, seed=seed + i)
            for i in range(self.n_agents)
        ]
        self.n_users = int(n_users)
        self.reward_scale = float(reward_scale)
        self.reward_clip = float(reward_clip)
        self.buffer = MAReplayBuffer(maddpg_cfg["buffer_size"], self.obs_dims,
                                     self.act_dims, n_users=self.n_users)
        self.gamma = float(maddpg_cfg["gamma"])
        self.tau = float(maddpg_cfg["tau"])
        self.batch_size = int(maddpg_cfg["batch_size"])
        self.warmup_steps = int(maddpg_cfg["warmup_steps"])
        self.grad_clip = float(maddpg_cfg["grad_clip"])
        self.noise_start = float(maddpg_cfg["noise_sigma_start"])
        self.noise_end = float(maddpg_cfg["noise_sigma_end"])
        self.noise_decay = int(maddpg_cfg["noise_decay_steps"])
        self.policy_update_every = int(maddpg_cfg.get("policy_update_every", 1))

        # Current dual multipliers used to recompute replay rewards (item 1).
        # None -> fall back to stored collection-time rewards.
        self.current_lambda = None

        self._learn_step = 0
        self._global_step = 0
        self._rng = np.random.default_rng(seed)
        # Agent-owned observation normalizers (one per agent). Attached by the
        # training driver; identity if never attached.
        self.obs_norms: list[ObservationNormalizer] | None = None

    # -------------------------------------------------- normalization
    def attach_obs_normalizers(self, norms: list[ObservationNormalizer]) -> None:
        assert len(norms) == self.n_agents
        self.obs_norms = norms

    def freeze_normalizers(self) -> None:
        if self.obs_norms is not None:
            for n in self.obs_norms:
                n.freeze()

    def normalizers_frozen(self) -> bool:
        return self.obs_norms is not None and all(n.frozen for n in self.obs_norms)

    def _norm_one(self, i: int, obs: np.ndarray, update: bool) -> np.ndarray:
        if self.obs_norms is None:
            return np.asarray(obs, dtype=np.float32)
        return self.obs_norms[i](obs, update=update)

    def _norm_batch(self, i: int, obs: np.ndarray) -> np.ndarray:
        if self.obs_norms is None:
            return np.asarray(obs, dtype=np.float32)
        return self.obs_norms[i].normalize_batch(obs)

    # -------------------------------------------------- exploration noise
    def _current_noise_sigma(self) -> float:
        frac = min(1.0, self._global_step / max(self.noise_decay, 1))
        return float(self.noise_start + (self.noise_end - self.noise_start) * frac)

    def reset_noise(self):
        for a in self.agents:
            a.noise.reset()

    # -------------------------------------------------- action selection
    @torch.no_grad()
    def select_actions(self, per_agent_obs: list[np.ndarray], explore: bool = True) -> list[np.ndarray]:
        """Select actions from RAW per-agent observations.

        Normalization happens here (statistics update only while exploring and
        while the normalizers are not frozen). During warmup the (untrained)
        policy is ignored and uniform actions are sampled, but the normalizer
        still sees the observation stream so its statistics warm up too.
        """
        norm_obs = [self._norm_one(i, o, update=explore)
                    for i, o in enumerate(per_agent_obs)]
        if explore and self._global_step < self.warmup_steps:
            return [self._rng.uniform(-1.0, 1.0, size=d).astype(np.float32) for d in self.act_dims]
        actions: list[np.ndarray] = []
        sigma = self._current_noise_sigma()
        for i, a in enumerate(self.agents):
            obs_t = _to_t(norm_obs[i], self.device).unsqueeze(0)
            act = a.actor(obs_t).cpu().numpy()[0]
            if explore:
                a.noise.set_sigma(sigma)
                act = act + a.noise.sample()
            act = np.clip(act, -1.0, 1.0)
            if not np.all(np.isfinite(act)):
                act = np.nan_to_num(act, nan=0.0, posinf=1.0, neginf=-1.0)
            actions.append(act.astype(np.float32))
        return actions

    def step_count(self) -> int:
        return self._global_step

    def increment_step(self):
        self._global_step += 1

    # -------------------------------------------------- dual reward recomputation
    def set_current_lambda(self, lambda_vec) -> None:
        """Set the current per-user dual multipliers used to recompute replay
        rewards at sample time (call after every dual update)."""
        self.current_lambda = (None if lambda_vec is None
                               else np.asarray(lambda_vec, dtype=np.float32).reshape(-1))

    # -------------------------------------------------- buffer
    def add_transition(self, obs_list, action_list, reward, next_obs_list, done,
                       base_reward=None, c_gap=None):
        """Store RAW observations; cooperative reward broadcast across agents.
        base_reward + c_gap enable reward recomputation under the current lambda."""
        rewards = [reward] * self.n_agents
        self.buffer.add(obs_list, action_list, rewards, next_obs_list, done,
                        base_reward=base_reward, c_gap=c_gap)

    # -------------------------------------------------- learning
    def learn(self) -> dict:
        if len(self.buffer) < max(self.batch_size, self.warmup_steps):
            return {}
        # Recompute rewards under the CURRENT lambda so the critic never trains
        # on stale reward functions (item 1).
        obs, actions, rewards, next_obs, dones = self.buffer.sample(
            self.batch_size, rng=self._rng, lambda_vec=self.current_lambda,
            reward_scale=self.reward_scale, reward_clip=self.reward_clip)
        # Buffer holds RAW observations -> normalize per agent at sample time.
        obs = [self._norm_batch(i, o) for i, o in enumerate(obs)]
        next_obs = [self._norm_batch(i, o) for i, o in enumerate(next_obs)]
        obs_t = [_to_t(o, self.device) for o in obs]
        next_obs_t = [_to_t(o, self.device) for o in next_obs]
        act_t = [_to_t(a, self.device) for a in actions]
        rew_t = [_to_t(r, self.device) for r in rewards]
        done_t = _to_t(dones, self.device)

        joint_obs = torch.cat(obs_t, dim=-1)
        joint_next_obs = torch.cat(next_obs_t, dim=-1)
        joint_act = torch.cat(act_t, dim=-1)

        with torch.no_grad():
            target_actions = [self.agents[i].actor_target(next_obs_t[i]) for i in range(self.n_agents)]
            joint_target_act = torch.cat(target_actions, dim=-1)

        info: dict = {}

        # ---- Critic updates ----
        for i, agent in enumerate(self.agents):
            with torch.no_grad():
                q_next = agent.critic_target(joint_next_obs, joint_target_act)
                y = rew_t[i] + self.gamma * (1.0 - done_t) * q_next
            q = agent.critic(joint_obs, joint_act)
            critic_loss = F.mse_loss(q, y)
            if not torch.isfinite(critic_loss):
                continue
            agent.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            gn = nn.utils.clip_grad_norm_(agent.critic.parameters(), self.grad_clip)
            agent.critic_opt.step()
            info[f"critic_loss_{i}"] = float(critic_loss.detach().cpu().item())
            info[f"critic_gradnorm_{i}"] = float(gn.detach().cpu().item() if hasattr(gn, "detach") else float(gn))

        # ---- Critic target soft update (every learn step, per Lowe et al. 2017) ----
        for agent in self.agents:
            soft_update(agent.critic, agent.critic_target, self.tau)

        # ---- Actor updates (with optional policy delay) ----
        if self._learn_step % self.policy_update_every == 0:
            for i, agent in enumerate(self.agents):
                joint_act_pi = self._actor_joint_action(i, obs_t)
                actor_loss = -agent.critic(joint_obs, joint_act_pi).mean()
                if not torch.isfinite(actor_loss):
                    continue
                agent.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                gn = nn.utils.clip_grad_norm_(agent.actor.parameters(), self.grad_clip)
                agent.actor_opt.step()
                info[f"actor_loss_{i}"] = float(actor_loss.detach().cpu().item())
                info[f"actor_gradnorm_{i}"] = float(gn.detach().cpu().item() if hasattr(gn, "detach") else float(gn))

            for agent in self.agents:
                soft_update(agent.actor, agent.actor_target, self.tau)

        self._learn_step += 1
        return info

    def _actor_action_parts(self, i: int, obs_t: list) -> list:
        """Per-agent actions for the actor-i update (Lowe et al. 2017, item 3):
        a_i = actor_i(o_i) with gradient; a_j = actor_j(o_j).detach() for j != i.
        The other agents' actions come from their CURRENT policies (detached),
        NOT from the replay buffer."""
        parts = []
        for j in range(self.n_agents):
            a_j = self.agents[j].actor(obs_t[j])
            parts.append(a_j if j == i else a_j.detach())
        return parts

    def _actor_joint_action(self, i: int, obs_t: list) -> "torch.Tensor":
        return torch.cat(self._actor_action_parts(i, obs_t), dim=-1)

    # -------------------------------------------------- checkpoints
    def weights_state_dict(self) -> dict:
        state = {}
        for i, a in enumerate(self.agents):
            state[f"actor_{i}"] = a.actor.state_dict()
            state[f"critic_{i}"] = a.critic.state_dict()
            state[f"actor_target_{i}"] = a.actor_target.state_dict()
            state[f"critic_target_{i}"] = a.critic_target.state_dict()
        return state

    def load_weights_state_dict(self, state: dict) -> None:
        for i, a in enumerate(self.agents):
            a.actor.load_state_dict(state[f"actor_{i}"])
            a.critic.load_state_dict(state[f"critic_{i}"])
            a.actor_target.load_state_dict(state[f"actor_target_{i}"])
            a.critic_target.load_state_dict(state[f"critic_target_{i}"])

    def train_state_dict(self) -> dict:
        """Optimizer / noise / counter / RNG state for exact training resume.
        Composed by experiments.checkpointing.TrainingCheckpoint (kept out of
        the per-experiment logic on purpose)."""
        return {
            "optim": {f"actor_opt_{i}": a.actor_opt.state_dict() for i, a in enumerate(self.agents)}
                     | {f"critic_opt_{i}": a.critic_opt.state_dict() for i, a in enumerate(self.agents)},
            "noise": [{"state": a.noise.state.copy(),
                       "sigma": a.noise.sigma,
                       "rng": a.noise.rng.bit_generator.state} for a in self.agents],
            "learn_step": self._learn_step,
            "global_step": self._global_step,
            "rng": self._rng.bit_generator.state,
            "obs_norms": None if self.obs_norms is None else [n.state_dict() for n in self.obs_norms],
            "current_lambda": (None if self.current_lambda is None
                               else self.current_lambda.tolist()),
        }

    def load_train_state_dict(self, state: dict) -> None:
        for i, a in enumerate(self.agents):
            a.actor_opt.load_state_dict(state["optim"][f"actor_opt_{i}"])
            a.critic_opt.load_state_dict(state["optim"][f"critic_opt_{i}"])
        for a, ns in zip(self.agents, state["noise"]):
            a.noise.state = np.asarray(ns["state"], dtype=np.float32)
            a.noise.sigma = float(ns["sigma"])
            a.noise.rng.bit_generator.state = ns["rng"]
        self._learn_step = int(state["learn_step"])
        self._global_step = int(state["global_step"])
        self._rng.bit_generator.state = state["rng"]
        cl = state.get("current_lambda")
        self.current_lambda = None if cl is None else np.asarray(cl, dtype=np.float32)
        if state.get("obs_norms") is not None:
            if self.obs_norms is None:
                self.obs_norms = [ObservationNormalizer(shape=(d,)) for d in self.obs_dims]
            for n, s in zip(self.obs_norms, state["obs_norms"]):
                n.load_state_dict(s)

    def replay_state(self) -> dict:
        """Only the VALID [0:size] entries are saved (item 3 reviewer fix)."""
        b = self.buffer
        s = int(b.size)
        return {"capacity": int(b.capacity), "idx": int(b.idx), "size": s,
                "obs": [o[:s].copy() for o in b.obs],
                "next_obs": [o[:s].copy() for o in b.next_obs],
                "actions": [a[:s].copy() for a in b.actions],
                "rewards": [r[:s].copy() for r in b.rewards],
                "base_rewards": b.base_rewards[:s].copy(),
                "c_gaps": b.c_gaps[:s].copy(),
                "dones": b.dones[:s].copy()}

    def load_replay_state(self, state: dict) -> None:
        b = self.buffer
        s = int(state["size"])
        for tgt, src in ((b.obs, state["obs"]), (b.next_obs, state["next_obs"]),
                         (b.actions, state["actions"]), (b.rewards, state["rewards"])):
            for t, arr in zip(tgt, src):
                t[:s] = arr
        if "base_rewards" in state:
            b.base_rewards[:s] = state["base_rewards"]
            b.c_gaps[:s] = state["c_gaps"]
        b.dones[:s] = state["dones"]
        b.idx = int(state["idx"])
        b.size = s

    def save_inference(self, path: str, extra_meta: dict | None = None):
        """Inference checkpoint: weights + normalizer statistics + metadata.
        Sufficient for deterministic evaluation/deployment; NOT for resume."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "kind": "maddpg_inference",
            "weights": self.weights_state_dict(),
            "obs_norms": None if self.obs_norms is None else [n.state_dict() for n in self.obs_norms],
            "obs_dims": list(self.obs_dims),
            "act_dims": list(self.act_dims),
            "meta": extra_meta or {},
        }
        torch.save(payload, path)

    def load_inference(self, path: str):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.load_weights_state_dict(payload["weights"])
        if payload.get("obs_norms") is not None:
            if self.obs_norms is None:
                self.obs_norms = [ObservationNormalizer(shape=(d,)) for d in self.obs_dims]
            for n, s in zip(self.obs_norms, payload["obs_norms"]):
                n.load_state_dict(s)

    # Back-compat plain weight save/load.
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.weights_state_dict(), path)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        state = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and state.get("kind") == "maddpg_inference":
            self.load_weights_state_dict(state["weights"])
        else:
            self.load_weights_state_dict(state)
