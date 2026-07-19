"""Training drivers for MADDPG / DDPG / TD3 / PPO on the STAR-RIS RSMA env.

Key mechanisms (post-refactor):
- Projected dual-gradient update of the per-user multipliers lambda_k on the
  SIGNED constraint c_k = R_min - R_k (EMA-smoothed episode means), consistent
  with the Lagrangian-surrogate reward evaluated by the environment.
- Optional two-stage dual freeze (heuristic training schedule, no convergence
  guarantee claimed): lambda is frozen for the final fraction of training so
  the reward becomes stationary in the tail.
- Agents own the observation normalizers; the replay buffers store RAW
  observations; normalizer statistics update until
  `obs_norm_freeze_after_env_steps` env steps, then freeze.
- Best checkpoint selected on the VALIDATION ScenarioBank by a deterministic
  lexicographic criterion matched to the expected-rate constraint
  (feasible iff max_k(R_min - mean R_k) <= model_select_constraint_tolerance).
  Test scenarios are never used for selection.
"""
from __future__ import annotations
import math
import os
from collections import deque, defaultdict

import numpy as np
import torch
from tqdm import tqdm

from env import StarRisRsmaEnv, build_eval_bank
from algorithms import MADDPG, DDPGAgent, TD3Agent, PPOAgent
from utils import Logger, ObservationNormalizer


# --------------------------------------------------------------------- helpers
def _set_seed(seed: int):
    """Deterministic seeding: numpy, torch (CPU + CUDA), cuDNN."""
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _select_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_cfg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was explicitly requested, but this Python/PyTorch runtime has no "
            "available CUDA device. Install a CUDA-enabled PyTorch build and run on "
            "GPU hardware; refusing to silently fall back to CPU.")
    if device_cfg not in ("cpu", "cuda"):
        raise ValueError(f"Unknown device setting: {device_cfg!r}")
    return device_cfg


def configure_cpu_threads(intra: int = 1, inter: int = 1) -> None:
    """Pin torch CPU thread counts (item 10 reviewer fix).

    On many-core review/CI machines the default oversubscription makes small
    MLP updates dramatically slower (MADDPG.learn measured 18.8 s at 56 threads
    vs ~0.006 s at 1 thread). Tests and CPU smoke runs call this explicitly.
    set_num_interop_threads must be set before any parallel work and only once;
    both calls are guarded.
    """
    try:
        torch.set_num_threads(int(intra))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(int(inter))
    except Exception:
        # Already initialised (can only be set once) -- leave as is.
        pass


def validate_formulation_config(cfg: dict) -> None:
    """Contextual-bandit runs must configure gamma = 0 explicitly.

    A silent runtime override would leave a misleading effective config, so a
    nonzero gamma is a hard error (final-revision requirement).
    """
    formulation = str(cfg["env"].get("env_formulation", "dynamic_mdp")).lower()
    if formulation != "contextual_bandit":
        return
    for algo in ("maddpg", "ddpg", "td3", "ppo"):
        if algo in cfg and float(cfg[algo].get("gamma", 0.0)) != 0.0:
            raise ValueError(
                f"env_formulation=contextual_bandit requires {algo}.gamma = 0 "
                f"(got {cfg[algo]['gamma']}). Set gamma: 0 in the config; the "
                "runtime does not silently override it.")


def _make_env(cfg: dict, seed: int, ris_mode: str = "optimized",
              equal_power: bool = False, qos_lambda_override: float | None = None,
              qos_lambda_vec_override=None) -> StarRisRsmaEnv:
    env_cfg = dict(cfg["env"])
    env_cfg["equal_power_mode"] = bool(equal_power)
    env = StarRisRsmaEnv(env_cfg, seed=seed, ris_mode=ris_mode)
    # Per-user vector takes precedence over the scalar (item 9 reviewer fix):
    # the trained multipliers are heterogeneous across users and must NOT be
    # collapsed to their scalar mean during validation/testing.
    if qos_lambda_vec_override is not None:
        env.set_qos_lambda_vec(np.asarray(qos_lambda_vec_override, dtype=np.float64))
    elif qos_lambda_override is not None:
        env.set_qos_lambda(qos_lambda_override)
    return env


def _with_qos_penalty_disabled(cfg: dict) -> dict:
    cfg2 = {**cfg, "env": dict(cfg["env"])}
    cfg2["env"]["qos_lambda_init"] = 0.0
    cfg2["env"]["dual_lambda_max"] = 0.0
    cfg2["env"]["augmented_penalty_weight"] = 0.0
    cfg2["env"]["enable_qos_shaping_bonus"] = False
    return cfg2


def _action_stats(action_arr: np.ndarray) -> dict:
    a = np.asarray(action_arr).reshape(-1)
    return {
        "act_mean": float(np.mean(a)),
        "act_std": float(np.std(a)),
        "act_abs_max": float(np.max(np.abs(a))),
        "act_sat_frac": float(np.mean(np.abs(a) > 0.95)),
    }


# --------------------------------------------------------------------- dual update
class DualUpdater:
    """Projected dual-gradient ascent on per-user multipliers.

    Uses the SAME signed constraint function as the reward:
        c_k     = mean_episode(R_min - R_k)
        ema_k  <- dual_ema * ema_k + (1 - dual_ema) * c_k
        lam_k  <- clip(lam_k + dual_lr * ema_k, 0, dual_lambda_max)

    The signed gap lets lambda_k DECREASE when user k has rate slack.
    Optional two-stage freeze (`two_stage_dual_freeze_fraction`): a heuristic
    schedule that keeps lambda constant for the final fraction of training so
    the reward scale is stationary. No saddle-point convergence is claimed.
    """

    def __init__(self, n_users: int, env_cfg: dict):
        self.lr = float(env_cfg.get("dual_lr", 0.01))
        self.ema_coef = float(env_cfg.get("dual_ema", 0.9))
        self.lambda_max = float(env_cfg.get("dual_lambda_max", 20.0))
        freeze = env_cfg.get("two_stage_dual_freeze_fraction", None)
        self.freeze_fraction = None if freeze is None else float(freeze)
        self.ema = np.zeros(n_users, dtype=np.float64)

    def frozen(self, ep: int | None, total_episodes: int | None) -> bool:
        return (self.freeze_fraction is not None
                and total_episodes is not None and ep is not None
                and self.freeze_fraction < 1.0
                and ep >= self.freeze_fraction * total_episodes)

    def update(self, env: StarRisRsmaEnv, episode_mean_c: np.ndarray,
               ep: int | None = None, total_episodes: int | None = None) -> np.ndarray:
        if self.frozen(ep, total_episodes):
            return env.qos_lambda_vec
        c = np.asarray(episode_mean_c, dtype=np.float64)
        self.ema = self.ema_coef * self.ema + (1.0 - self.ema_coef) * c
        lam = np.clip(env.qos_lambda_vec + self.lr * self.ema, 0.0, self.lambda_max)
        env.set_qos_lambda_vec(lam)
        return env.qos_lambda_vec

    def state_dict(self) -> dict:
        return {"ema": self.ema.copy()}

    def load_state_dict(self, s: dict) -> None:
        self.ema = np.asarray(s["ema"], dtype=np.float64)


# --------------------------------------------------------------------- info aggregation
_SCALAR_INFO_KEYS = (
    "sum_rate", "rate_common", "user_qos_fraction", "min_user_rate",
    "mean_qos_deficit", "max_qos_deficit",
    "h_eff_abs_mean", "h_eff_abs_R", "h_eff_abs_T",
    "phase_entropy_R", "phase_entropy_T", "phase_var_R", "phase_var_T",
    "beta_r_mean", "common_power_frac", "total_power_W",
    "reward_sr", "reward_dual", "reward_aug", "reward_switch", "reward_bonus",
    "phase_switch_cost", "power_switch_cost", "beta_switch_cost",
)


def _aggregate_info(buf: dict, info: dict, reward: float):
    """Per-step info aggregation within a single episode."""
    for k in _SCALAR_INFO_KEYS:
        if k in info:
            buf[k].append(float(info[k]))
    buf["all_users_qos_satisfied"].append(float(info.get("all_users_qos_satisfied", False)))
    buf["qos_constraint_signed"].append(np.asarray(info["qos_constraint_signed"], dtype=np.float64))
    buf["per_user_rate"].append(np.asarray(info["per_user_rate"], dtype=np.float64))
    buf["reward"].append(float(reward))


def _summarize(buf: dict) -> dict:
    out = {}
    for k, v in buf.items():
        if not v:
            continue
        if k in ("qos_constraint_signed", "per_user_rate"):
            continue
        out[f"{k}_mean"] = float(np.mean(v))
    if buf.get("qos_constraint_signed"):
        out["episode_mean_c"] = np.mean(np.stack(buf["qos_constraint_signed"]), axis=0)
    if buf.get("per_user_rate"):
        out["episode_mean_per_user_rate"] = np.mean(np.stack(buf["per_user_rate"]), axis=0)
    return out


# --------------------------------------------------------------------- history
_HISTORY_KEYS = ("episode_return", "sum_rate", "user_qos_fraction",
                 "all_users_qos_satisfied", "qos_lambda_mean",
                 "min_user_rate", "mean_qos_deficit",
                 "h_eff_abs_T", "phase_entropy_T",
                 "rate_common", "common_power_frac", "ma_return")


def _new_history(n_users: int) -> dict:
    h = {k: [] for k in _HISTORY_KEYS}
    h["qos_lambda_per_user"] = []   # list of K-vectors per episode
    return h


def _push_history(history: dict, ep_return: float, ep_summary: dict,
                  lam_vec: np.ndarray, ma_return: float):
    history["episode_return"].append(ep_return)
    history["sum_rate"].append(ep_summary.get("sum_rate_mean", 0.0))
    history["user_qos_fraction"].append(ep_summary.get("user_qos_fraction_mean", 0.0))
    history["all_users_qos_satisfied"].append(ep_summary.get("all_users_qos_satisfied_mean", 0.0))
    history["qos_lambda_mean"].append(float(np.mean(lam_vec)))
    history["qos_lambda_per_user"].append(np.asarray(lam_vec, dtype=np.float64).copy())
    history["min_user_rate"].append(ep_summary.get("min_user_rate_mean", 0.0))
    history["mean_qos_deficit"].append(ep_summary.get("mean_qos_deficit_mean", 0.0))
    history["h_eff_abs_T"].append(ep_summary.get("h_eff_abs_T_mean", 0.0))
    history["phase_entropy_T"].append(ep_summary.get("phase_entropy_T_mean", 0.0))
    history["rate_common"].append(ep_summary.get("rate_common_mean", 0.0))
    history["common_power_frac"].append(ep_summary.get("common_power_frac_mean", 0.0))
    history["ma_return"].append(ma_return)


# --------------------------------------------------------------------- model selection
class BestCheckpointSelector:
    """Deterministic lexicographic model selection on VALIDATION scenarios.

    With c_bar_k = R_min - mean(R_k) over the validation set:
      1. feasible iff max_k(c_bar_k) <= model_select_constraint_tolerance;
      2. among feasible checkpoints pick the highest validation sum-rate;
      3. otherwise pick the smallest max violation, then smallest mean
         violation, then highest sum-rate.
    Implemented as a sort key so comparisons are total and deterministic.
    Test scenarios are NEVER used here.
    """

    def __init__(self, tolerance: float):
        self.tolerance = float(tolerance)
        self.best_key = None
        self.best_info = None
        # Per-user dual multipliers active at the SELECTED best episode. These
        # (not the final-episode lambdas) must be used to evaluate best.pt so
        # the reward scale matches the checkpoint (items 1 & 9).
        self.best_lambda_vec = None

    def key(self, val_metrics: dict) -> tuple:
        c_bar = np.asarray(val_metrics["c_bar_per_user"], dtype=np.float64)
        max_v = float(c_bar.max())
        mean_v = float(np.maximum(c_bar, 0.0).mean())
        sr = float(val_metrics["sum_rate_mean"])
        feasible = max_v <= self.tolerance
        if feasible:
            return (0, -sr, 0.0, 0.0)
        return (1, max_v, mean_v, -sr)

    def consider(self, val_metrics: dict, episode: int,
                 lambda_vec=None) -> bool:
        k = self.key(val_metrics)
        if self.best_key is None or k < self.best_key:
            self.best_key = k
            self.best_lambda_vec = (None if lambda_vec is None
                                    else np.asarray(lambda_vec, dtype=np.float64).copy())
            self.best_info = {
                "episode": episode, "key": k,
                "sum_rate_mean": float(val_metrics["sum_rate_mean"]),
                "c_bar_per_user": list(map(float, val_metrics["c_bar_per_user"])),
                "lambda_vec": (None if lambda_vec is None
                               else list(map(float, np.asarray(lambda_vec).ravel()))),
            }
            return True
        return False

    def state_dict(self) -> dict:
        return {"best_key": self.best_key, "best_info": self.best_info,
                "tolerance": self.tolerance,
                "best_lambda_vec": (None if self.best_lambda_vec is None
                                    else self.best_lambda_vec.tolist())}

    def load_state_dict(self, s: dict) -> None:
        self.best_key = None if s["best_key"] is None else tuple(s["best_key"])
        self.best_info = s["best_info"]
        self.tolerance = float(s["tolerance"])
        blv = s.get("best_lambda_vec")
        self.best_lambda_vec = None if blv is None else np.asarray(blv, dtype=np.float64)


def _get_validation_bank(cfg: dict):
    """Training jobs materialize the VALIDATION bank only (V4 review P0-3):
    the locked test ScenarioBank must never be constructed inside a training
    process -- only the final evaluation/aggregate job builds it."""
    return build_eval_bank(cfg, "validation")


# Provenance merged into every inference checkpoint's extra_meta (V4 review
# P0-1): the caller (main.py) registers source_sha/config_sha once, so every
# best.pt records which source tree produced it.
_CHECKPOINT_PROVENANCE: dict = {}


def set_checkpoint_provenance(meta: dict | None) -> None:
    _CHECKPOINT_PROVENANCE.clear()
    _CHECKPOINT_PROVENANCE.update(meta or {})


def checkpoint_provenance() -> dict:
    return dict(_CHECKPOINT_PROVENANCE)


def load_maddpg_inference(cfg: dict, best_path: str, seed: int,
                          device: str | None = None) -> MADDPG:
    """Rebuild a MADDPG agent and load an inference checkpoint (best.pt).

    Downstream evaluation (test/sweep/ablation/diagnostics/latency) must run on
    the SELECTED best checkpoint, not the final in-memory agent, so that results
    are consistent with the reported checkpoint_sha (item 1 reviewer fix).
    """
    device = device or _select_device(cfg.get("device", "auto"))
    env = _make_env(cfg, seed)
    agent = MADDPG(env.spec(), hidden_sizes=cfg["networks"]["hidden_sizes"],
                   maddpg_cfg=cfg["maddpg"], net_cfg=cfg["networks"],
                   device=device, seed=seed)
    agent.attach_obs_normalizers(
        [ObservationNormalizer(shape=(d,)) for d in env.spec().obs_dims])
    agent.load_inference(best_path)
    return agent


def load_single_agent_inference(cfg: dict, kind: str, best_path: str, seed: int,
                                hidden: list[int], device: str | None = None):
    """Rebuild a DDPG/TD3 agent and load its inference checkpoint (best.pt)."""
    device = device or _select_device(cfg.get("device", "auto"))
    env = _make_env(cfg, seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    if kind == "ddpg":
        agent = DDPGAgent(obs_dim, act_dim, hidden, cfg["ddpg"], cfg["networks"],
                          device=device, seed=seed)
    elif kind in ("td3", "td3_matched"):
        agent = TD3Agent(obs_dim, act_dim, hidden, cfg["td3"], cfg["networks"],
                         device=device, seed=seed)
    else:
        raise ValueError(f"Unknown single-agent algorithm: {kind}")
    agent.attach_obs_normalizer(ObservationNormalizer(shape=(obs_dim,)))
    agent.load_inference(best_path)
    return agent


def load_ppo_inference(cfg: dict, best_path: str, seed: int,
                       device: str | None = None):
    device = device or _select_device(cfg.get("device", "auto"))
    env = _make_env(cfg, seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, act_dim, cfg["networks"]["hidden_sizes"],
                     cfg["ppo"], cfg["networks"], device=device, seed=seed)
    agent.attach_obs_normalizer(ObservationNormalizer(shape=(obs_dim,)))
    agent.load_inference(best_path)
    return agent


# --------------------------------------------------------------------- freeze helper
def _obs_norm_freeze_threshold(cfg: dict, algo_cfg: dict) -> int:
    """Env steps after which normalizer statistics freeze. Applies to every
    algorithm (PPO included, which has no warmup_steps of its own)."""
    if "obs_norm_freeze_after_env_steps" in cfg["env"]:
        return int(cfg["env"]["obs_norm_freeze_after_env_steps"])
    return int(algo_cfg.get("warmup_steps", 5000))


# ============================================================ MADDPG
def train_maddpg(cfg: dict, total_episodes: int | None = None,
                 run_name: str = "maddpg", log_dir: str = "logs",
                 ckpt_dir: str = "checkpoints", ris_mode: str = "optimized",
                 seed_override: int | None = None,
                 disable_qos_penalty: bool = False,
                 disable_obs_norm: bool = False,
                 resume_from: str | None = None) -> dict:
    validate_formulation_config(cfg)
    seed = int(seed_override if seed_override is not None else cfg["seed"])
    _set_seed(seed)
    device = _select_device(cfg.get("device", "auto"))
    cfg2 = dict(cfg)
    if disable_qos_penalty:
        cfg2 = _with_qos_penalty_disabled(cfg)
    env = _make_env(cfg2, seed, ris_mode=ris_mode)
    spec = env.spec()
    agent = MADDPG(spec,
                   hidden_sizes=cfg["networks"]["hidden_sizes"],
                   maddpg_cfg=cfg["maddpg"],
                   net_cfg=cfg["networks"],
                   device=device, seed=seed,
                   n_users=env.K, reward_scale=env.r_scale, reward_clip=env.r_clip)
    agent.set_current_lambda(env.qos_lambda_vec.copy())
    obs_norms = [ObservationNormalizer(shape=(d,)) for d in spec.obs_dims]
    if disable_obs_norm:
        for o in obs_norms:
            o.enabled = False
    agent.attach_obs_normalizers(obs_norms)
    logger = Logger(log_dir, run_name)

    total_episodes = int(total_episodes or cfg["training"]["total_episodes"])
    eval_every = int(cfg["training"]["eval_every"])
    ckpt_every = int(cfg["training"]["checkpoint_every"])
    ckpt_path = os.path.join(ckpt_dir, run_name, "latest.pt")
    best_path = os.path.join(ckpt_dir, run_name, "best.pt")
    smoothw = int(cfg["training"].get("reward_smoothing_window", 20))
    freeze_after = _obs_norm_freeze_threshold(cfg, cfg["maddpg"])

    dual = DualUpdater(env.K, cfg2["env"] if disable_qos_penalty else cfg["env"])
    selector = BestCheckpointSelector(
        float(cfg["training"].get("model_select_constraint_tolerance", 0.0)))
    val_bank = _get_validation_bank(cfg)

    history = _new_history(env.K)
    return_window = deque(maxlen=smoothw)
    total_env_steps = 0
    start_ep = 0

    if resume_from:
        from experiments.checkpointing import TrainingCheckpoint
        tc = TrainingCheckpoint.load(resume_from)
        tc.restore_maddpg(agent, env, dual, selector)
        history = tc.trainer_state["history"]
        total_env_steps = int(tc.trainer_state["total_env_steps"])
        start_ep = int(tc.trainer_state["episode_index"]) + 1
        return_window.extend(history["episode_return"][-smoothw:])

    pbar = tqdm(range(start_ep, total_episodes), desc=run_name, ncols=110)
    for ep in pbar:
        env.reset(seed=seed + ep)
        per_agent_obs = env.per_agent_observations()   # RAW observations
        agent.reset_noise()
        buf = defaultdict(list)
        ep_return, steps = 0.0, 0
        for t in range(env.max_steps):
            actions = agent.select_actions(per_agent_obs, explore=True)
            next_obs, reward, term, trunc, info = env.step(actions)
            episode_done = term or trunc
            bootstrap_done = term
            if not math.isfinite(reward):
                logger.buffer("nan_reward_count", 1.0)
                continue
            next_per_agent = env.per_agent_observations()   # RAW
            agent.add_transition(per_agent_obs, actions, reward, next_per_agent, float(bootstrap_done),
                                 base_reward=info["base_reward"],
                                 c_gap=info["qos_constraint_signed"])
            agent.increment_step()
            total_env_steps += 1
            if total_env_steps >= freeze_after and not agent.normalizers_frozen():
                agent.freeze_normalizers()
            losses = agent.learn()
            for k, v in losses.items():
                logger.buffer(k, v)
            if (steps % 10) == 0:
                flat = np.concatenate([np.asarray(a).reshape(-1) for a in actions])
                for ak, av in _action_stats(flat).items():
                    logger.buffer(ak, av)
            _aggregate_info(buf, info, reward)
            ep_return += reward
            per_agent_obs = next_per_agent
            steps += 1
            if episode_done:
                break

        ep_summary = _summarize(buf)
        return_window.append(ep_return)
        lam_vec = dual.update(env, ep_summary.get("episode_mean_c", np.zeros(env.K)),
                              ep=ep, total_episodes=total_episodes)
        # Off-policy critic must recompute replay rewards under the NEW lambda
        # from now on (item 1).
        agent.set_current_lambda(lam_vec)
        _push_history(history, ep_return, ep_summary, lam_vec,
                      float(np.mean(return_window)))

        log_row = {"episode_return": ep_return,
                   "ma_return": float(np.mean(return_window)),
                   "user_qos_fraction": ep_summary.get("user_qos_fraction_mean", 0.0),
                   "all_users_qos_ok": ep_summary.get("all_users_qos_satisfied_mean", 0.0),
                   "qos_lambda_mean": float(np.mean(lam_vec)),
                   "noise_sigma": agent._current_noise_sigma()}
        for k_u in range(env.K):
            log_row[f"qos_lambda_{k_u}"] = float(lam_vec[k_u])
        log_row.update({k: v for k, v in ep_summary.items()
                        if not isinstance(v, np.ndarray)})
        logger.log(ep, log_row)
        logger.flush_buffers(ep)
        pbar.set_postfix({"ret": f"{ep_return:.2f}",
                          "MA": f"{np.mean(return_window):.2f}",
                          "uqf": f"{ep_summary.get('user_qos_fraction_mean', 0.0):.2f}",
                          "lambda": f"{np.mean(lam_vec):.2f}"})

        # Validation + best-selector update FIRST so a checkpoint captured this
        # episode already reflects the updated selector state (item 4).
        if (ep + 1) % eval_every == 0:
            em = evaluate_agent(env_cfg=cfg, agent=agent, kind="maddpg",
                                scenarios=val_bank.scenarios,
                                qos_lambda_vec=lam_vec)
            logger.log(ep, {f"val_{k}": v for k, v in em.items()
                            if not isinstance(v, (list, np.ndarray))})
            if selector.consider(em, episode=ep, lambda_vec=lam_vec):
                agent.save_inference(best_path, extra_meta={
                    "selected_episode": ep, "criterion": selector.best_info,
                    "lambda_vec": list(map(float, np.asarray(lam_vec).ravel())),
                    **checkpoint_provenance()})

        if (ep + 1) % ckpt_every == 0 or ep == total_episodes - 1:
            from experiments.checkpointing import TrainingCheckpoint
            TrainingCheckpoint.capture_maddpg(
                agent, env, dual, selector,
                trainer_state={"episode_index": ep, "total_env_steps": total_env_steps,
                               "history": history, "seed": seed,
                               "run_name": run_name},
                cfg=cfg,
            ).save(ckpt_path)

    # If validation never ran (very short runs), fall back to final weights.
    if selector.best_info is None:
        agent.save_inference(best_path, extra_meta={
            "selected_episode": total_episodes - 1,
            "criterion": "final (no validation pass)",
            "lambda_vec": list(map(float, env.qos_lambda_vec.ravel())),
            **checkpoint_provenance()})
        best_lambda_vec = env.qos_lambda_vec.copy()
    else:
        best_lambda_vec = (selector.best_lambda_vec.copy()
                           if selector.best_lambda_vec is not None
                           else env.qos_lambda_vec.copy())
    for i, on in enumerate(obs_norms):
        on.save(os.path.join(ckpt_dir, run_name, f"obs_norm_{i}.npz"))
    logger.close()
    # Rebuild the eval agent from best.pt so all downstream evaluation runs on
    # the selected checkpoint (weights consistent with checkpoint_sha).
    eval_agent = load_maddpg_inference(cfg, best_path, seed, device=device)
    return {"agent": eval_agent,          # best.pt -- use for ALL evaluation
            "train_agent": agent,          # final in-memory agent (diagnostics only)
            "obs_norm": obs_norms, "history": history,
            "best_ckpt": best_path, "latest_ckpt": ckpt_path,
            "trained_qos_lambda_vec": best_lambda_vec,
            "trained_qos_lambda": float(np.mean(best_lambda_vec)),
            "best_selection": selector.best_info}


# ============================================================ DDPG / TD3
def train_single_agent(cfg: dict, kind: str, total_episodes: int | None = None,
                       run_name: str | None = None, log_dir: str = "logs",
                       ckpt_dir: str = "checkpoints",
                       seed_override: int | None = None,
                       disable_qos_penalty: bool = False,
                       disable_obs_norm: bool = False,
                       hidden_sizes_override: list[int] | None = None) -> dict:
    validate_formulation_config(cfg)
    seed = int(seed_override if seed_override is not None else cfg["seed"])
    _set_seed(seed)
    device = _select_device(cfg.get("device", "auto"))
    cfg2 = cfg
    if disable_qos_penalty:
        cfg2 = _with_qos_penalty_disabled(cfg)
    env = _make_env(cfg2, seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    algo_key = "td3" if kind.startswith("td3") else kind
    hidden = list(hidden_sizes_override or cfg["networks"]["hidden_sizes"])
    _rk = dict(n_users=env.K, reward_scale=env.r_scale, reward_clip=env.r_clip)
    if kind == "ddpg":
        agent = DDPGAgent(obs_dim, act_dim, hidden,
                          cfg["ddpg"], cfg["networks"], device=device, seed=seed, **_rk)
    elif kind in ("td3", "td3_matched"):
        agent = TD3Agent(obs_dim, act_dim, hidden,
                         cfg["td3"], cfg["networks"], device=device, seed=seed, **_rk)
    else:
        raise ValueError(f"Unknown single-agent algorithm: {kind}")
    agent.set_current_lambda(env.qos_lambda_vec.copy())

    obs_norm = ObservationNormalizer(shape=(obs_dim,))
    if disable_obs_norm:
        obs_norm.enabled = False
    agent.attach_obs_normalizer(obs_norm)
    run_name = run_name or kind
    logger = Logger(log_dir, run_name)
    total_episodes = int(total_episodes or cfg["training"]["total_episodes"])
    eval_every = int(cfg["training"]["eval_every"])
    ckpt_every = int(cfg["training"]["checkpoint_every"])
    smoothw = int(cfg["training"].get("reward_smoothing_window", 20))
    freeze_after = _obs_norm_freeze_threshold(cfg, cfg[algo_key])
    ckpt_path = os.path.join(ckpt_dir, run_name, "latest.pt")
    best_path = os.path.join(ckpt_dir, run_name, "best.pt")

    dual = DualUpdater(env.K, cfg2["env"] if disable_qos_penalty else cfg["env"])
    selector = BestCheckpointSelector(
        float(cfg["training"].get("model_select_constraint_tolerance", 0.0)))
    val_bank = _get_validation_bank(cfg)

    history = _new_history(env.K)
    return_window = deque(maxlen=smoothw)
    total_env_steps = 0

    pbar = tqdm(range(total_episodes), desc=run_name, ncols=110)
    for ep in pbar:
        obs, _ = env.reset(seed=seed + ep)     # RAW observation
        agent.reset_noise()
        buf = defaultdict(list)
        ep_return, steps = 0.0, 0
        for t in range(env.max_steps):
            action = agent.select_action(obs, explore=True)
            next_obs, reward, term, trunc, info = env.step(action)
            episode_done = term or trunc
            bootstrap_done = term
            if not math.isfinite(reward):
                logger.buffer("nan_reward_count", 1.0)
                continue
            agent.add_transition(obs, action, reward, next_obs, float(bootstrap_done),
                                 base_reward=info["base_reward"],
                                 c_gap=info["qos_constraint_signed"])
            agent.increment_step()
            total_env_steps += 1
            if total_env_steps >= freeze_after and (agent.obs_norm is None or not agent.obs_norm.frozen):
                agent.freeze_normalizers()
            losses = agent.learn()
            for k, v in losses.items():
                logger.buffer(k, v)
            if (steps % 10) == 0:
                for ak, av in _action_stats(action).items():
                    logger.buffer(ak, av)
            _aggregate_info(buf, info, reward)
            ep_return += reward
            obs = next_obs
            steps += 1
            if episode_done:
                break

        ep_summary = _summarize(buf)
        return_window.append(ep_return)
        lam_vec = dual.update(env, ep_summary.get("episode_mean_c", np.zeros(env.K)),
                              ep=ep, total_episodes=total_episodes)
        agent.set_current_lambda(lam_vec)
        _push_history(history, ep_return, ep_summary, lam_vec,
                      float(np.mean(return_window)))

        log_row = {"episode_return": ep_return, "ma_return": float(np.mean(return_window)),
                   "user_qos_fraction": ep_summary.get("user_qos_fraction_mean", 0.0),
                   "all_users_qos_ok": ep_summary.get("all_users_qos_satisfied_mean", 0.0),
                   "qos_lambda_mean": float(np.mean(lam_vec))}
        log_row.update({k: v for k, v in ep_summary.items()
                        if not isinstance(v, np.ndarray)})
        logger.log(ep, log_row)
        logger.flush_buffers(ep)
        pbar.set_postfix({"ret": f"{ep_return:.2f}", "MA": f"{np.mean(return_window):.2f}",
                          "uqf": f"{ep_summary.get('user_qos_fraction_mean', 0.0):.2f}",
                          "lambda": f"{np.mean(lam_vec):.2f}"})
        if (ep + 1) % eval_every == 0:
            em = evaluate_agent(env_cfg=cfg, agent=agent, kind=algo_key,
                                scenarios=val_bank.scenarios,
                                qos_lambda_vec=lam_vec)
            logger.log(ep, {f"val_{k}": v for k, v in em.items()
                            if not isinstance(v, (list, np.ndarray))})
            if selector.consider(em, episode=ep, lambda_vec=lam_vec):
                agent.save_inference(best_path, extra_meta={
                    "selected_episode": ep, "criterion": selector.best_info,
                    "lambda_vec": list(map(float, np.asarray(lam_vec).ravel())),
                    **checkpoint_provenance()})
        if (ep + 1) % ckpt_every == 0 or ep == total_episodes - 1:
            agent.save(ckpt_path)

    if selector.best_info is None:
        agent.save_inference(best_path, extra_meta={
            "selected_episode": total_episodes - 1,
            "criterion": "final (no validation pass)",
            "lambda_vec": list(map(float, env.qos_lambda_vec.ravel())),
            **checkpoint_provenance()})
        best_lambda_vec = env.qos_lambda_vec.copy()
    else:
        best_lambda_vec = (selector.best_lambda_vec.copy()
                           if selector.best_lambda_vec is not None
                           else env.qos_lambda_vec.copy())
    agent.save(ckpt_path)
    obs_norm.save(os.path.join(ckpt_dir, run_name, "obs_norm.npz"))
    logger.close()
    eval_agent = load_single_agent_inference(cfg, kind, best_path, seed, hidden,
                                             device=device)
    return {"agent": eval_agent, "train_agent": agent,
            "obs_norm": obs_norm, "history": history,
            "best_ckpt": best_path, "latest_ckpt": ckpt_path,
            "trained_qos_lambda_vec": best_lambda_vec,
            "trained_qos_lambda": float(np.mean(best_lambda_vec)),
            "best_selection": selector.best_info}


# ============================================================ PPO
def train_ppo(cfg: dict, total_episodes: int | None = None,
              run_name: str = "ppo", log_dir: str = "logs",
              ckpt_dir: str = "checkpoints",
              seed_override: int | None = None) -> dict:
    validate_formulation_config(cfg)
    seed = int(seed_override if seed_override is not None else cfg["seed"])
    _set_seed(seed)
    device = _select_device(cfg.get("device", "auto"))
    env = _make_env(cfg, seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, act_dim, cfg["networks"]["hidden_sizes"],
                     cfg["ppo"], cfg["networks"], device=device, seed=seed)
    obs_norm = ObservationNormalizer(shape=(obs_dim,))
    agent.attach_obs_normalizer(obs_norm)
    logger = Logger(log_dir, run_name)
    total_episodes = int(total_episodes or cfg["training"]["total_episodes"])
    ckpt_every = int(cfg["training"]["checkpoint_every"])
    eval_every = int(cfg["training"]["eval_every"])
    smoothw = int(cfg["training"].get("reward_smoothing_window", 20))
    freeze_after = _obs_norm_freeze_threshold(cfg, cfg["ppo"])
    ckpt_path = os.path.join(ckpt_dir, run_name, "latest.pt")
    best_path = os.path.join(ckpt_dir, run_name, "best.pt")

    dual = DualUpdater(env.K, cfg["env"])
    selector = BestCheckpointSelector(
        float(cfg["training"].get("model_select_constraint_tolerance", 0.0)))
    val_bank = _get_validation_bank(cfg)

    history = _new_history(env.K)
    return_window = deque(maxlen=smoothw)
    total_env_steps = 0
    lam_vec = env.qos_lambda_vec.copy()
    # PPO is on-policy: lambda stays FIXED for a whole rollout and is updated
    # only AFTER a PPO policy/GAE update (item 2), so every optimization batch
    # uses a single lambda vector. Constraint gaps are accumulated across the
    # rollout and applied to the dual variables when the rollout is consumed.
    rollout_c_sum = np.zeros(env.K, dtype=np.float64)
    rollout_c_cnt = 0

    def _apply_dual_after_rollout(ep_idx):
        nonlocal rollout_c_sum, rollout_c_cnt, lam_vec
        if rollout_c_cnt == 0:
            return
        mean_c = rollout_c_sum / rollout_c_cnt
        lam_vec = dual.update(env, mean_c, ep=ep_idx, total_episodes=total_episodes)
        rollout_c_sum = np.zeros(env.K, dtype=np.float64)
        rollout_c_cnt = 0

    pbar = tqdm(range(total_episodes), desc=run_name, ncols=110)
    for ep in pbar:
        obs, _ = env.reset(seed=seed + ep)     # RAW observation
        buf = defaultdict(list)
        ep_return, steps = 0.0, 0
        term = trunc = False
        for t in range(env.max_steps):
            action, log_prob, value = agent.select_action(obs, explore=True)
            next_obs, reward, term, trunc, info = env.step(action)
            episode_done = term or trunc
            bootstrap_done = term
            if not math.isfinite(reward):
                logger.buffer("nan_reward_count", 1.0)
                continue
            # Store the EXACT normalized observation used to compute log_prob/
            # value (item 2), not the raw observation.
            agent.store(agent.last_norm_obs, action, log_prob, reward, value, float(bootstrap_done))
            rollout_c_sum += np.asarray(info["qos_constraint_signed"], dtype=np.float64)
            rollout_c_cnt += 1
            total_env_steps += 1
            if total_env_steps >= freeze_after and (agent.obs_norm is None or not agent.obs_norm.frozen):
                agent.freeze_normalizers()
            if (steps % 10) == 0:
                for ak, av in _action_stats(action).items():
                    logger.buffer(ak, av)
            _aggregate_info(buf, info, reward)
            ep_return += reward
            obs = next_obs
            steps += 1
            if agent.buffer_full():
                last_v = 0.0 if bootstrap_done else agent.value(obs)
                losses = agent.learn(last_v)     # lambda fixed for this batch
                for k, v in losses.items():
                    logger.buffer(k, v)
                _apply_dual_after_rollout(ep)     # update lambda AFTER the batch
            if episode_done:
                break
        if agent.rollout.size > 0 and (ep == total_episodes - 1 or agent.buffer_full()):
            last_v = 0.0 if term else agent.value(obs)
            losses = agent.learn(last_v)
            for k, v in losses.items():
                logger.buffer(k, v)
            _apply_dual_after_rollout(ep)

        ep_summary = _summarize(buf)
        return_window.append(ep_return)
        _push_history(history, ep_return, ep_summary, lam_vec,
                      float(np.mean(return_window)))

        log_row = {"episode_return": ep_return, "ma_return": float(np.mean(return_window)),
                   "user_qos_fraction": ep_summary.get("user_qos_fraction_mean", 0.0),
                   "all_users_qos_ok": ep_summary.get("all_users_qos_satisfied_mean", 0.0),
                   "qos_lambda_mean": float(np.mean(lam_vec))}
        log_row.update({k: v for k, v in ep_summary.items()
                        if not isinstance(v, np.ndarray)})
        logger.log(ep, log_row)
        logger.flush_buffers(ep)
        pbar.set_postfix({"ret": f"{ep_return:.2f}", "MA": f"{np.mean(return_window):.2f}",
                          "uqf": f"{ep_summary.get('user_qos_fraction_mean', 0.0):.2f}",
                          "lambda": f"{np.mean(lam_vec):.2f}"})
        if (ep + 1) % eval_every == 0:
            em = evaluate_agent(env_cfg=cfg, agent=agent, kind="ppo",
                                scenarios=val_bank.scenarios,
                                qos_lambda_vec=lam_vec)
            logger.log(ep, {f"val_{k}": v for k, v in em.items()
                            if not isinstance(v, (list, np.ndarray))})
            if selector.consider(em, episode=ep, lambda_vec=lam_vec):
                agent.save_inference(best_path, extra_meta={
                    "selected_episode": ep, "criterion": selector.best_info,
                    "lambda_vec": list(map(float, np.asarray(lam_vec).ravel())),
                    **checkpoint_provenance()})
        if (ep + 1) % ckpt_every == 0 or ep == total_episodes - 1:
            agent.save(ckpt_path)

    if selector.best_info is None:
        agent.save_inference(best_path, extra_meta={
            "selected_episode": total_episodes - 1,
            "criterion": "final (no validation pass)",
            "lambda_vec": list(map(float, env.qos_lambda_vec.ravel())),
            **checkpoint_provenance()})
        best_lambda_vec = env.qos_lambda_vec.copy()
    else:
        best_lambda_vec = (selector.best_lambda_vec.copy()
                           if selector.best_lambda_vec is not None
                           else env.qos_lambda_vec.copy())
    agent.save(ckpt_path)
    obs_norm.save(os.path.join(ckpt_dir, run_name, "obs_norm.npz"))
    logger.close()
    eval_agent = load_ppo_inference(cfg, best_path, seed, device=device)
    return {"agent": eval_agent, "train_agent": agent,
            "obs_norm": obs_norm, "history": history,
            "best_ckpt": best_path, "latest_ckpt": ckpt_path,
            "trained_qos_lambda_vec": best_lambda_vec,
            "trained_qos_lambda": float(np.mean(best_lambda_vec)),
            "best_selection": selector.best_info}


# ============================================================ evaluation
def evaluate_agent(env_cfg: dict, agent, kind: str,
                   episodes: int = 5, seed: int = 12345,
                   ris_mode: str = "optimized",
                   equal_power: bool = False,
                   qos_lambda: float | None = None,
                   qos_lambda_vec=None,
                   scenarios: list[dict] | None = None,
                   obs_norm=None) -> dict:
    """Deterministic-policy evaluation with rich per-step diagnostics.

    Agents normalize observations internally (they own the frozen normalizer),
    so RAW observations are passed throughout. `obs_norm` is accepted for
    backward compatibility and ignored.

    qos_lambda_vec: per-user dual multipliers (preferred). Takes precedence over
    the scalar `qos_lambda`; the vector must NOT be collapsed to its mean
    (item 9 reviewer fix).

    scenarios: optional list of ScenarioBank scenarios. When given, each
    episode plays back one scenario (identical geometry + channel trajectory
    for every method); `episodes`/`seed` control the fallback random mode.
    """
    env = _make_env(env_cfg, seed, ris_mode=ris_mode, equal_power=equal_power,
                    qos_lambda_override=qos_lambda,
                    qos_lambda_vec_override=qos_lambda_vec)
    n_episodes = len(scenarios) if scenarios is not None else episodes
    rets, srs = [], []
    uqf, all_ok, min_rates, mean_defs = [], [], [], []
    rates_common, h_T_abs, phase_ent_T, common_frac = [], [], [], []
    per_user_rates = []
    is_maddpg = (kind == "maddpg")

    for ep in range(n_episodes):
        if scenarios is not None:
            obs, _ = env.reset(seed=seed + ep, options={"scenario": scenarios[ep]})
        else:
            obs, _ = env.reset(seed=seed + ep)
        if is_maddpg:
            per_agent_obs = env.per_agent_observations()
        ep_ret, steps = 0.0, 0
        ep_buf = defaultdict(list)
        ep_per_user = []
        for t in range(env.max_steps):
            if is_maddpg:
                actions = agent.select_actions(per_agent_obs, explore=False)
            elif kind in ("ddpg", "td3", "td3_matched"):
                actions = agent.select_action(obs, explore=False)
            elif kind == "ppo":
                actions, _, _ = agent.select_action(obs, explore=False)
            else:
                raise ValueError(kind)
            obs, reward, term, trunc, info = env.step(actions)
            if is_maddpg:
                per_agent_obs = env.per_agent_observations()
            ep_ret += reward
            _aggregate_info(ep_buf, info, reward)
            ep_per_user.append(info.get("per_user_rate", np.zeros(env.K)))
            steps += 1
            if term or trunc:
                break
        rets.append(ep_ret)
        srs.append(float(np.mean(ep_buf.get("sum_rate", [0.0]))))
        uqf.append(float(np.mean(ep_buf.get("user_qos_fraction", [0.0]))))
        all_ok.append(float(np.mean(ep_buf.get("all_users_qos_satisfied", [0.0]))))
        min_rates.append(float(np.mean(ep_buf.get("min_user_rate", [0.0]))))
        mean_defs.append(float(np.mean(ep_buf.get("mean_qos_deficit", [0.0]))))
        rates_common.append(float(np.mean(ep_buf.get("rate_common", [0.0]))))
        h_T_abs.append(float(np.mean(ep_buf.get("h_eff_abs_T", [0.0]))))
        phase_ent_T.append(float(np.mean(ep_buf.get("phase_entropy_T", [0.0]))))
        common_frac.append(float(np.mean(ep_buf.get("common_power_frac", [0.0]))))
        per_user_rates.append(np.mean(np.stack(ep_per_user, axis=0), axis=0))
    per_user_rates = np.stack(per_user_rates, axis=0)
    mean_rate_per_user = per_user_rates.mean(axis=0)
    c_bar = env.qos_min - mean_rate_per_user

    return {
        "return_mean": float(np.mean(rets)),
        "return_std":  float(np.std(rets)),
        "sum_rate_mean": float(np.mean(srs)),
        "sum_rate_std":  float(np.std(srs)),
        # QoS metrics -- explicit names (P0-2).
        "user_qos_fraction_mean": float(np.mean(uqf)),
        "user_qos_fraction_std": float(np.std(uqf)),
        "all_users_qos_prob": float(np.mean(all_ok)),
        "all_users_qos_prob_std": float(np.std(all_ok)),
        "min_user_rate_mean": float(np.mean(min_rates)),
        "mean_qos_deficit_mean": float(np.mean(mean_defs)),
        "c_bar_per_user": c_bar.tolist(),
        "rate_common_mean": float(np.mean(rates_common)),
        "h_eff_abs_T_mean": float(np.mean(h_T_abs)),
        "phase_entropy_T_mean": float(np.mean(phase_ent_T)),
        "common_power_frac_mean": float(np.mean(common_frac)),
        "per_user_rate_mean": mean_rate_per_user.tolist(),
        "per_episode_sum_rate": list(map(float, srs)),
        "per_episode_user_qos_fraction": list(map(float, uqf)),
        "per_episode_all_users_qos": list(map(float, all_ok)),
        "per_episode_return": list(map(float, rets)),
    }
