"""Shared fixtures/helpers for the test suite."""
from __future__ import annotations
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Pin torch CPU threads (item 10 reviewer fix): on many-core CI/review machines
# the default oversubscription makes small MLP updates dramatically slower
# (MADDPG.learn measured 18.8 s at 56 threads vs ~0.006 s at 1 thread).
import torch  # noqa: E402
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass


def base_env_cfg(**overrides) -> dict:
    cfg = {
        "num_bs_antennas": 4,
        "num_users": 4,
        "num_users_reflection": 3,
        "num_ris_elements": 8,
        "star_mode": "ES",
        "p_max_dbm": 30.0,
        "noise_power_dbm": -90.0,
        "qos_rate_min": 0.3,
        "path_loss_exp_direct": 3.5,
        "path_loss_exp_bs_ris": 2.2,
        "path_loss_exp_ris_user": 2.5,
        "ref_path_loss_db": 30.0,
        "ref_distance": 1.0,
        "direct_block_T": True,
        "direct_block_loss_db": 25.0,
        "direct_block_R_loss_db": 0.0,
        "bs_position": [0.0, 0.0, 5.0],
        "ris_position": [50.0, 0.0, 10.0],
        "user_area_reflection": [[35.0, 50.0], [-10.0, 10.0], [1.5, 1.5]],
        "user_area_transmission": [[55.0, 75.0], [-10.0, 10.0], [1.5, 1.5]],
        "env_formulation": "dynamic_mdp",
        "channel_rho": 0.95,
        "resample_positions_on_reset": True,
        "max_steps": 10,
        "channel_block_steps": 10,
        "reward_alpha": 1.5,
        "reward_rate_reference": 20.0,
        "reward_scale": 0.1,
        "reward_clip": 50.0,
        "qos_lambda_init": 1.0,
        "augmented_penalty_weight": 1.0,
        "dual_lr": 0.01,
        "dual_ema": 0.9,
        "dual_lambda_max": 20.0,
        "enable_qos_shaping_bonus": False,
        "phase_switching_cost": 0.1,
        "power_switching_cost": 0.05,
        "beta_switching_cost": 0.05,
        "equal_power_mode": False,
        "phase_action_mode": "absolute",
        "phase_residual_scale": 0.25,
        "analytical_phase_min_direct": 1.0e-9,
        "obs_include_channel_state": True,
        "obs_include_ris_state": True,
        "local_obs_for_maddpg": True,
        "epsilon": 1.0e-12,
    }
    cfg.update(overrides)
    return cfg


def full_cfg(**env_overrides) -> dict:
    """Full training config (smoke-sized) for driver-level tests."""
    return {
        "seed": 2026,
        "device": "cpu",
        "env": base_env_cfg(**env_overrides),
        "networks": {"hidden_sizes": [32, 32], "activation": "relu",
                     "layer_norm": True, "ortho_init": True},
        "maddpg": {"num_agents": 3, "actor_lr": 1.0e-4, "critic_lr": 5.0e-4,
                   "gamma": 0.95, "tau": 0.005, "batch_size": 16,
                   "buffer_size": 1000, "warmup_steps": 20, "grad_clip": 1.0,
                   "noise_sigma_start": 0.4, "noise_sigma_end": 0.05,
                   "noise_decay_steps": 100, "policy_update_every": 1},
        "ddpg": {"actor_lr": 1.0e-4, "critic_lr": 5.0e-4, "gamma": 0.95,
                 "tau": 0.005, "batch_size": 16, "buffer_size": 1000,
                 "warmup_steps": 20, "grad_clip": 1.0,
                 "noise_sigma_start": 0.4, "noise_sigma_end": 0.05,
                 "noise_decay_steps": 100},
        "td3": {"actor_lr": 1.0e-4, "critic_lr": 5.0e-4, "gamma": 0.95,
                "tau": 0.005, "batch_size": 16, "buffer_size": 1000,
                "warmup_steps": 20, "grad_clip": 1.0,
                "noise_sigma_start": 0.4, "noise_sigma_end": 0.05,
                "noise_decay_steps": 100, "policy_noise": 0.2,
                "noise_clip": 0.5, "policy_delay": 2},
        "ppo": {"lr": 3.0e-4, "gamma": 0.95, "gae_lambda": 0.95,
                "clip_eps": 0.2, "vf_coef": 0.5, "ent_coef": 0.005,
                "epochs": 2, "minibatch_size": 16, "rollout_length": 32,
                "grad_clip": 0.5, "target_kl": 0.03},
        "training": {"total_episodes": 4, "eval_every": 1000,
                     "eval_episodes": 1, "checkpoint_every": 2,
                     "early_stop_patience": 0, "log_dir": "logs_test",
                     "ckpt_dir": "ckpt_test", "reward_smoothing_window": 5,
                     "training_seeds": [1000, 2000],
                     "model_select_constraint_tolerance": 0.0},
        "evaluation": {"num_episodes": 1, "validation_seeds": [11],
                       "test_seeds": [9101], "power_sweep_dbm": [30],
                       "ao_reference_lambda": [1.0, 1.0, 1.0, 1.0],
                       "n_sweep": [8], "k_sweep": [4],
                       "ablation_modes": [], "ao_local_search_max_n": 8,
                       "ao_scenarios_per_seed": 1,
                       # Benchmark-light integration tests (V4 review item 6).
                       "latency_num_calls": 5, "latency_warmup_calls": 1,
                       "diagnostic_steps": 4},
    }
