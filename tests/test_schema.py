"""Observation/action schemas are the single source of truth for layouts."""
from __future__ import annotations
import json
import os
import numpy as np
import pytest

from env import StarRisRsmaEnv
from conftest import base_env_cfg


@pytest.mark.parametrize("N,K,K_r", [(4, 2, 1), (8, 4, 3), (16, 4, 2)])
def test_schema_dims_match_actual_observations(N, K, K_r):
    env = StarRisRsmaEnv(base_env_cfg(num_ris_elements=N, num_users=K,
                                      num_users_reflection=K_r), seed=1)
    env.reset(seed=2)
    obs_schema = env.observation_schema()
    act_schema = env.action_schema()

    per_agent = env.per_agent_observations()
    for i, o in enumerate(per_agent):
        fields = obs_schema["agents"][i]
        assert fields[-1]["stop"] == o.shape[0] == env.obs_dims[i]
        # Fields tile the vector contiguously without gaps.
        pos = 0
        for f in fields:
            assert f["start"] == pos
            pos = f["stop"]

    flat = env._build_observation()
    assert obs_schema["single_agent"][-1]["stop"] == flat.shape[0]

    for i, d in enumerate(env.act_dims):
        assert act_schema["agents"][i][-1]["stop"] == d


def test_action_order_beta_then_phase_for_ris_r():
    env = StarRisRsmaEnv(base_env_cfg(), seed=1)
    fields = env.action_schema()["agents"][1]
    assert fields[0]["name"] == "beta_r" and fields[0]["start"] == 0
    assert fields[1]["name"] == "phi_r" and fields[1]["start"] == env.N


def test_observation_schema_excludes_prev_reward():
    env = StarRisRsmaEnv(base_env_cfg(), seed=1)
    names = [f["name"] for f in env.observation_schema()["single_agent"]]
    assert "prev_reward" not in names
    for fields in env.observation_schema()["agents"]:
        assert "prev_reward" not in [f["name"] for f in fields]


def test_export_schema_json(tmp_path):
    env = StarRisRsmaEnv(base_env_cfg(), seed=1)
    obs_p = os.path.join(tmp_path, "observation_schema.json")
    act_p = os.path.join(tmp_path, "action_schema.json")
    env.export_schema(obs_p, act_p)
    with open(obs_p, encoding="utf-8") as f:
        obs = json.load(f)
    with open(act_p, encoding="utf-8") as f:
        act = json.load(f)
    assert obs["obs_dims"] == list(env.obs_dims)
    assert act["act_dims"] == list(env.act_dims)


def test_ris_state_in_observation_reflects_applied_phases():
    env = StarRisRsmaEnv(base_env_cfg(phase_action_mode="absolute"), seed=1)
    env.reset(seed=2)
    rng = np.random.default_rng(3)
    action = rng.uniform(-1, 1, size=env.act_dim_flat).astype(np.float32)
    env.step(action)
    per_agent = env.per_agent_observations()
    schema = env.observation_schema()["agents"][1]
    f = next(x for x in schema if x["name"] == "ris_beta_r")
    beta_obs = per_agent[1][f["start"]:f["stop"]]
    np.testing.assert_allclose(beta_obs, 2.0 * env._beta_r - 1.0, rtol=1e-5)
