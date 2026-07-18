"""ScenarioBank: auditable per-(seed, episode) scenarios with identical
geometry/channel trajectories for every method."""
from __future__ import annotations
import os
import numpy as np

from env import StarRisRsmaEnv, ScenarioBank
from conftest import base_env_cfg


def test_auditable_scenario_ids_and_provenance():
    cfg = base_env_cfg()
    bank = ScenarioBank(cfg, split="test", evaluation_seeds=[9101, 9202],
                        episodes_per_seed=3)
    assert len(bank) == 6
    ids = [sc["scenario_id"] for sc in bank.scenarios]
    assert ids[0] == "test_seed9101_ep000"
    assert ids[2] == "test_seed9101_ep002"
    assert ids[3] == "test_seed9202_ep000"
    for sc in bank.scenarios:
        assert sc["split"] == "test"
        assert sc["evaluation_seed"] in (9101, 9202)
        assert 0 <= sc["episode_idx"] < 3


def test_validation_bank_builder_never_touches_test_split(monkeypatch):
    """Training drivers call build_eval_bank(cfg, "validation") only (V4 P0-3)."""
    import env.scenario_bank as sbank
    from experiments.train import _get_validation_bank
    from conftest import full_cfg

    built = []
    orig = sbank.generate_scenario

    def spy(env_cfg, split, evaluation_seed, episode_idx):
        built.append(split)
        return orig(env_cfg, split, evaluation_seed, episode_idx)

    monkeypatch.setattr(sbank, "generate_scenario", spy)
    bank = _get_validation_bank(full_cfg())
    assert len(bank) > 0
    assert set(built) == {"validation"}


def test_build_eval_bank_rejects_unknown_split():
    import pytest
    from env import build_eval_bank
    from conftest import full_cfg
    with pytest.raises(ValueError):
        build_eval_bank(full_cfg(), "locked")


def test_bank_determinism_and_hash():
    cfg = base_env_cfg()
    b1 = ScenarioBank(cfg, split="test", evaluation_seeds=[7], episodes_per_seed=3)
    b2 = ScenarioBank(cfg, split="test", evaluation_seeds=[7], episodes_per_seed=3)
    assert b1.content_hash() == b2.content_hash()
    b3 = ScenarioBank(cfg, split="test", evaluation_seeds=[8], episodes_per_seed=3)
    assert b1.content_hash() != b3.content_hash()


def test_playback_identical_across_policies():
    """Two different action sequences on the same scenario: identical
    geometry and identical channel trajectory (innovation playback)."""
    cfg = base_env_cfg(max_steps=6)
    bank = ScenarioBank(cfg, split="val", evaluation_seeds=[7], episodes_per_seed=1)
    sc = bank[0]
    trajs, geoms = [], []
    for action_seed in (1, 2):
        env = StarRisRsmaEnv(cfg, seed=99)
        env.reset(options={"scenario": sc})
        geoms.append(env.user_positions.copy())
        rng = np.random.default_rng(action_seed)
        seq = []
        for _ in range(cfg["max_steps"]):
            a = rng.uniform(-1, 1, size=env.act_dim_flat).astype(np.float32)
            env.step(a)
            seq.append(env._h_d_small.copy())
        trajs.append(np.stack(seq))
    np.testing.assert_allclose(geoms[0], geoms[1])
    np.testing.assert_allclose(trajs[0], trajs[1], rtol=0, atol=0)


def test_playback_respects_gauss_markov_recursion():
    cfg = base_env_cfg(max_steps=4, channel_rho=0.7)
    bank = ScenarioBank(cfg, split="test", evaluation_seeds=[3], episodes_per_seed=1)
    sc = bank[0]
    env = StarRisRsmaEnv(cfg, seed=5)
    env.reset(options={"scenario": sc})
    rho = cfg["channel_rho"]
    scale = np.sqrt(1 - rho ** 2)
    h = np.asarray(sc["h_d_small0"])
    rng = np.random.default_rng(0)
    for t in range(cfg["max_steps"]):
        np.testing.assert_allclose(env._h_d_small, h, rtol=1e-12)
        a = rng.uniform(-1, 1, size=env.act_dim_flat).astype(np.float32)
        env.step(a)
        h = rho * h + scale * np.asarray(sc["innov_h_d"][t])


def test_save_load_roundtrip(tmp_path):
    cfg = base_env_cfg()
    bank = ScenarioBank(cfg, split="val", evaluation_seeds=[11, 22],
                        episodes_per_seed=2)
    jp = os.path.join(tmp_path, "bank.json")
    npz = os.path.join(tmp_path, "bank.npz")
    bank.save(jp, npz)
    loaded = ScenarioBank.load(jp, npz)
    assert loaded.content_hash() == bank.content_hash()
    assert loaded[0]["scenario_id"] == bank[0]["scenario_id"]
    assert loaded[0]["evaluation_seed"] == bank[0]["evaluation_seed"]
