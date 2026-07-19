"""Post-training evaluation on ScenarioBanks + fair latency benchmarks.

Latency policy (P0-8 reviewer fix):
- benchmark_latency_cpu(): every model is forced onto the CPU with
  torch.set_num_threads(1) (restored afterwards), torch.inference_mode(),
  warm-up, >= 2000 calls; the timed region covers observation preprocessing,
  the actor forward pass and action post-processing. Classical baselines
  (analytical phases, coarse AO-grid) are timed on the SAME single CPU thread.
- benchmark_latency_gpu(): CUDA-only table with torch.cuda.synchronize()
  around the timed region.
- The two tables are reported SEPARATELY and never compared head-to-head.
"""
from __future__ import annotations
import copy
import platform
import time
import numpy as np
import torch

from experiments.train import evaluate_agent, _make_env


_KIND_ALIAS = {
    "MADDPG": "maddpg", "DDPG": "ddpg", "TD3": "td3", "PPO": "ppo",
    "TD3-Matched": "td3_matched", "TD3_MATCHED": "td3_matched",
    "FixedRIS": "maddpg", "RandomRIS": "maddpg", "NoRIS": "maddpg",
    "AnalyticalRIS": "maddpg", "AO-Grid": "maddpg", "BCD": "maddpg",
    "EqualPowerLearned": "maddpg", "EqualPowerFixed": "maddpg",
    "EqualPowerOnly": "maddpg", "MRTDirectionsOnly": "maddpg",
    "UniformCommonSplitOnly": "maddpg",
    "ClassicalMRTEqualPowerFixed": "maddpg",
    "NoQoSPenalty": "maddpg", "NoRewardNorm": "maddpg",
}


def kind_of(algo_label: str) -> str:
    return _KIND_ALIAS.get(algo_label, algo_label).lower()


def eval_on_scenarios(agent, algo_label: str, cfg: dict, scenarios: list[dict],
                      ris_mode: str = "optimized", equal_power: bool = False,
                      qos_lambda: float | None = None,
                      qos_lambda_vec=None,
                      env_overrides: dict | None = None) -> dict:
    """Evaluate one agent deterministically on a fixed scenario list.

    Every method sees the identical geometry + channel trajectory per
    scenario (playback), which makes per-scenario values pairable. The per-user
    dual multipliers (qos_lambda_vec) are preferred over the scalar mean."""
    return evaluate_agent(env_cfg=cfg, agent=agent, kind=kind_of(algo_label),
                          scenarios=scenarios, ris_mode=ris_mode,
                          equal_power=equal_power, qos_lambda=qos_lambda,
                          qos_lambda_vec=qos_lambda_vec,
                          env_overrides=env_overrides)


def scenario_rows(algo_label: str, metrics: dict, scenarios: list[dict],
                  training_seed=None, config_sha: str = "",
                  checkpoint_sha: str = "", solver_config_sha: str = "",
                  extra: dict | None = None) -> list[dict]:
    """Tidy per-scenario rows for results_raw.csv.

    Every row carries full provenance (item 8): evaluation_seed, episode_idx,
    scenario_id, config_sha and checkpoint_sha. Non-learned baselines leave
    checkpoint_sha empty but set solver_config_sha instead.
    """
    rows = []
    for j, sc in enumerate(scenarios):
        base = {
            "algorithm": algo_label,
            "training_seed": training_seed if training_seed is not None else "",
            "evaluation_seed": sc.get("evaluation_seed", ""),
            "episode_idx": sc.get("episode_idx", ""),
            "scenario_id": sc["scenario_id"],
            "config_sha": config_sha,
            "checkpoint_sha": checkpoint_sha,
            "solver_config_sha": solver_config_sha,
        }
        if extra:
            base.update(extra)
        for metric_key, per_ep_key in (
                ("sum_rate", "per_episode_sum_rate"),
                ("user_qos_fraction", "per_episode_user_qos_fraction"),
                ("all_users_qos_satisfied", "per_episode_all_users_qos"),
                ("return", "per_episode_return")):
            rows.append({**base, "metric": metric_key,
                         "value": metrics[per_ep_key][j]})
    return rows


# ------------------------------------------------------------------ latency
def _agent_torch_modules(agent) -> list[torch.nn.Module]:
    mods = []
    if hasattr(agent, "agents"):            # MADDPG
        for a in agent.agents:
            mods.extend([a.actor, a.actor_target, a.critic, a.critic_target])
    else:
        for attr in ("actor", "actor_target", "critic", "critic_target"):
            m = getattr(agent, attr, None)
            if isinstance(m, torch.nn.Module):
                mods.append(m)
    return mods


def _clone_for_inference(agent, device: str):
    """Deepcopy an agent WITHOUT its replay buffer and move it to `device`."""
    buf = getattr(agent, "buffer", None)
    rollout = getattr(agent, "rollout", None)
    try:
        if buf is not None:
            agent.buffer = None
        if rollout is not None:
            agent.rollout = None
        clone = copy.deepcopy(agent)
    finally:
        if buf is not None:
            agent.buffer = buf
        if rollout is not None:
            agent.rollout = rollout
    clone.device = torch.device(device)
    for m in _agent_torch_modules(clone):
        m.to(clone.device)
    return clone


def _summ(lats_ms: list[float]) -> dict:
    arr = np.asarray(lats_ms, dtype=np.float64)
    return {"mean_ms": float(arr.mean()), "median_ms": float(np.median(arr)),
            "std_ms": float(arr.std()), "p95_ms": float(np.percentile(arr, 95)),
            "n_calls": int(arr.size)}


def _time_policy(agent, env, is_maddpg: bool, is_ppo: bool,
                 num_calls: int, warmup: int) -> list[float]:
    """Timed region: observation preprocessing (env obs build + normalizer)
    + actor forward + action post-processing (clip/nan guard)."""
    lats = []
    with torch.inference_mode():
        for i in range(warmup + num_calls):
            t0 = time.perf_counter()
            if is_maddpg:
                per_agent_obs = env.per_agent_observations()
                agent.select_actions(per_agent_obs, explore=False)
            else:
                obs = env._build_observation()
                if is_ppo:
                    agent.select_action(obs, explore=False)
                else:
                    agent.select_action(obs, explore=False)
            dt = (time.perf_counter() - t0) * 1000.0
            if i >= warmup:
                lats.append(dt)
    return lats


def _time_baseline(env, fn, num_calls: int, warmup: int) -> list[float]:
    lats = []
    for i in range(warmup + num_calls):
        t0 = time.perf_counter()
        fn(env)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            lats.append(dt)
    return lats


def _bench_env(cfg: dict):
    env = _make_env(cfg, seed=int(cfg["seed"]))
    # Keep every AO-Grid invocation on the same pre-registered objective used
    # by the ablation and Hybrid AO references.
    if "ao_reference_lambda" in cfg.get("evaluation", {}):
        from experiments.baselines_ao import ao_reference_lambda
        env.set_qos_lambda_vec(ao_reference_lambda(cfg))
    env.reset(seed=int(cfg["seed"]))
    return env


def benchmark_latency_cpu(agents: dict, cfg: dict, num_calls: int = 2000,
                          warmup: int = 50,
                          include_baselines: bool = True) -> dict:
    """Single-thread CPU latency benchmark. Restores the previous torch
    thread setting afterwards (finally block)."""
    old_threads = torch.get_num_threads()
    results: dict = {"_meta": {
        "device": "cpu", "torch_threads": 1,
        "torch_version": torch.__version__,
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
    }}
    try:
        torch.set_num_threads(1)
        for algo, agent in agents.items():
            clone = _clone_for_inference(agent, "cpu")
            env = _bench_env(cfg)
            is_maddpg = hasattr(clone, "select_actions")
            is_ppo = (not is_maddpg) and (kind_of(algo) == "ppo")
            lats = _time_policy(clone, env, is_maddpg, is_ppo, num_calls, warmup)
            results[algo] = _summ(lats)
        if include_baselines:
            env = _bench_env(cfg)
            results["AnalyticalRIS"] = _summ(_time_baseline(
                env, lambda e: e._analytical_phases(), num_calls, warmup))
            env = _bench_env(cfg)
            # AO-grid is orders of magnitude slower; fewer calls suffice.
            n_ao = max(20, num_calls // 100)
            results["AO-Grid"] = _summ(_time_baseline(
                env, lambda e: e._coarse_ao_grid(), n_ao, warmup=2))
    finally:
        torch.set_num_threads(old_threads)
    return results


def benchmark_latency_gpu(agents: dict, cfg: dict, num_calls: int = 2000,
                          warmup: int = 50) -> dict | None:
    """CUDA latency benchmark (policies only). Returns None without CUDA.
    Reported in its OWN table; never compared directly against CPU numbers."""
    if not torch.cuda.is_available():
        return None
    results: dict = {"_meta": {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
    }}
    for algo, agent in agents.items():
        clone = _clone_for_inference(agent, "cuda")
        env = _bench_env(cfg)
        is_maddpg = hasattr(clone, "select_actions")
        is_ppo = (not is_maddpg) and (kind_of(algo) == "ppo")
        lats = []
        with torch.inference_mode():
            for i in range(warmup + num_calls):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if is_maddpg:
                    per_agent_obs = env.per_agent_observations()
                    clone.select_actions(per_agent_obs, explore=False)
                else:
                    obs = env._build_observation()
                    clone.select_action(obs, explore=False)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000.0
                if i >= warmup:
                    lats.append(dt)
        results[algo] = _summ(lats)
    return results
