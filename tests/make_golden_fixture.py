"""Generate the golden regression fixture for the current MISO environment.

It records PHYSICS-ONLY outputs (channels, effective channels, RSMA rates) for
fixed seeds and a fixed action sequence. Rewards, observations and QoS penalty
terms are intentionally NOT recorded because those are algorithmic surfaces.
tests/test_legacy_regression.py replays the same seeds and actions through
env_formulation="static_block" and asserts the MISO physical layer stays stable.

Fixture metadata (source version, config, library versions, date) is stored in
the .npz and in a JSON sidecar for auditability.
"""
from __future__ import annotations
import json
import os
import platform
import sys
from datetime import date

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env import StarRisRsmaEnv  # noqa: E402


def _base_cfg(n_elements: int, num_users: int, k_r: int,
              max_steps: int, block_steps: int) -> dict:
    return {
        "num_bs_antennas": 4,
        "num_users": num_users,
        "num_users_reflection": k_r,
        "num_ris_elements": n_elements,
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
        "env_formulation": "static_block",
        "max_steps": max_steps,
        "channel_block_steps": block_steps,
        # Physics-neutral action decoding: absolute phase mapping so the fixture
        # does not depend on the analytical-prior implementation (which gains a
        # fallback in the refactor).
        "phase_action_mode": "absolute",
        "equal_power_mode": False,
        "obs_include_channel_state": True,
        "local_obs_for_maddpg": True,
        "epsilon": 1.0e-12,
        "obs_include_ris_state": True,
    }


# (name, cfg, env_seed, reset_seed, action_seed)
CASES = [
    # Channel constant for the whole episode (legacy standard setup).
    ("case_block_full", _base_cfg(8, 4, 3, max_steps=10, block_steps=10), 123, 456, 789),
    # Channel refresh mid-episode: verifies the legacy refresh schedule.
    ("case_block_3", _base_cfg(16, 4, 3, max_steps=9, block_steps=3), 321, 654, 987),
    # Different topology.
    ("case_k2", _base_cfg(8, 2, 1, max_steps=6, block_steps=6), 111, 222, 333),
]


def run_case(name: str, cfg: dict, env_seed: int, reset_seed: int, action_seed: int) -> dict:
    env = StarRisRsmaEnv(cfg, seed=env_seed, ris_mode="optimized")
    env.reset(seed=reset_seed)
    act_rng = np.random.default_rng(action_seed)

    out: dict[str, list | np.ndarray] = {
        "user_positions": env.user_positions.copy(),
        "alpha_d": env.alpha_d.copy(),
        "alpha_br": np.array([env.alpha_br]),
        "alpha_ru": env.alpha_ru.copy(),
    }
    h_d_steps, G_steps, g_steps = [], [], []
    h_eff_steps, per_user_steps, sum_rate_steps, rate_c_steps = [], [], [], []
    actions_flat = []

    for _ in range(cfg["max_steps"]):
        action = act_rng.uniform(-1.0, 1.0, size=env.act_dim_flat).astype(np.float32)
        actions_flat.append(action.copy())
        _, _, _, _, info = env.step(action)
        h_d_steps.append(env._h_d.copy())
        G_steps.append(env._G.copy())
        g_steps.append(env._g.copy())
        h_eff_steps.append(env._h_eff.copy())
        per_user_steps.append(np.asarray(info["per_user_rate"], dtype=np.float64))
        sum_rate_steps.append(float(info["sum_rate"]))
        rate_c_steps.append(float(info["rate_common"]))

    out["actions"] = np.stack(actions_flat)
    out["h_d"] = np.stack(h_d_steps)
    out["G"] = np.stack(G_steps)
    out["g"] = np.stack(g_steps)
    out["h_eff"] = np.stack(h_eff_steps)
    out["per_user_rate"] = np.stack(per_user_steps)
    out["sum_rate"] = np.asarray(sum_rate_steps)
    out["rate_common"] = np.asarray(rate_c_steps)
    return out


def main():
    fixture_dir = os.path.join(PROJECT_ROOT, "tests", "fixtures")
    os.makedirs(fixture_dir, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    meta = {
        "source": "current MISO implementation (env/star_ris_env.py)",
        "generated": str(date.today()),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cases": [],
    }
    for name, cfg, env_seed, reset_seed, action_seed in CASES:
        data = run_case(name, cfg, env_seed, reset_seed, action_seed)
        for key, val in data.items():
            arrays[f"{name}__{key}"] = np.asarray(val)
        meta["cases"].append({
            "name": name, "config": cfg,
            "env_seed": env_seed, "reset_seed": reset_seed, "action_seed": action_seed,
        })
        print(f"[fixture] {name}: sum_rate head = {data['sum_rate'][:3]}")

    npz_path = os.path.join(fixture_dir, "golden_static_block.npz")
    np.savez_compressed(npz_path, **arrays)
    json_path = os.path.join(fixture_dir, "golden_static_block.meta.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {npz_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
