"""Checkpointing: exact training resume (TrainingCheckpoint, with replay) and
inference checkpoints (weights + normalizer only)."""
from __future__ import annotations
import os
import numpy as np
import torch

from experiments.train import train_maddpg
from conftest import full_cfg


def _cfg(tmp_path, total_episodes):
    cfg = full_cfg()
    cfg["training"]["total_episodes"] = total_episodes
    cfg["training"]["checkpoint_every"] = 3
    cfg["training"]["eval_every"] = 1000          # skip validation during test
    cfg["env"]["two_stage_dual_freeze_fraction"] = None   # freeze depends on
    # total_episodes; disable it so a 3-episode run matches the first 3
    # episodes of a 6-episode run exactly.
    cfg["training"]["log_dir"] = os.path.join(str(tmp_path), "logs")
    cfg["training"]["ckpt_dir"] = os.path.join(str(tmp_path), "ckpt")
    return cfg


def _weights_equal(sd_a: dict, sd_b: dict) -> bool:
    for k in sd_a:
        if not torch.allclose(sd_a[k], sd_b[k], rtol=0, atol=0):
            return False
    return True


def test_exact_resume_matches_uninterrupted_run(tmp_path):
    # Run A: 6 episodes uninterrupted.
    cfg_a = _cfg(tmp_path / "a", total_episodes=6)
    res_a = train_maddpg(cfg_a, run_name="mad_a",
                         log_dir=cfg_a["training"]["log_dir"],
                         ckpt_dir=cfg_a["training"]["ckpt_dir"])

    # Run B phase 1: 3 episodes (TrainingCheckpoint saved at episode 2).
    cfg_b = _cfg(tmp_path / "b", total_episodes=3)
    res_b1 = train_maddpg(cfg_b, run_name="mad_b",
                          log_dir=cfg_b["training"]["log_dir"],
                          ckpt_dir=cfg_b["training"]["ckpt_dir"])
    ckpt = res_b1["latest_ckpt"]
    assert os.path.exists(ckpt)

    # Run B phase 2: resume to 6 episodes.
    cfg_b2 = _cfg(tmp_path / "b", total_episodes=6)
    res_b2 = train_maddpg(cfg_b2, run_name="mad_b",
                          log_dir=cfg_b2["training"]["log_dir"],
                          ckpt_dir=cfg_b2["training"]["ckpt_dir"],
                          resume_from=ckpt)

    # Episode returns after the resume point must match run A exactly.
    ret_a = res_a["history"]["episode_return"]
    ret_b = res_b2["history"]["episode_return"]
    assert len(ret_a) == len(ret_b) == 6
    np.testing.assert_allclose(ret_a, ret_b, rtol=0, atol=1e-10)

    # Final (in-memory) training weights identical -- compare train_agent, since
    # "agent" is now the rebuilt best.pt eval agent.
    wa = res_a["train_agent"].weights_state_dict()
    wb = res_b2["train_agent"].weights_state_dict()
    for key in wa:
        assert _weights_equal(wa[key], wb[key]), f"weights differ: {key}"

    # Dual variables identical.
    np.testing.assert_allclose(res_a["trained_qos_lambda_vec"],
                               res_b2["trained_qos_lambda_vec"],
                               rtol=0, atol=0)


def test_evaluated_agent_equals_best_checkpoint(tmp_path):
    """The returned eval agent's weights equal the on-disk best.pt (item 1/13)."""
    import torch as _torch
    from algorithms import MADDPG
    from utils import ObservationNormalizer
    from env import StarRisRsmaEnv
    cfg = _cfg(tmp_path, total_episodes=4)
    cfg["training"]["eval_every"] = 2          # exercise validation + selection
    res = train_maddpg(cfg, run_name="mad_best",
                       log_dir=cfg["training"]["log_dir"],
                       ckpt_dir=cfg["training"]["ckpt_dir"])
    # Load best.pt fresh and compare to the returned eval agent.
    env = StarRisRsmaEnv(cfg["env"], seed=1)
    fresh = MADDPG(env.spec(), hidden_sizes=cfg["networks"]["hidden_sizes"],
                   maddpg_cfg=cfg["maddpg"], net_cfg=cfg["networks"],
                   device="cpu", seed=0)
    fresh.attach_obs_normalizers(
        [ObservationNormalizer(shape=(d,)) for d in env.spec().obs_dims])
    fresh.load_inference(res["best_ckpt"])
    wa = res["agent"].weights_state_dict()
    wb = fresh.weights_state_dict()
    for key in wa:
        for pa, pb in zip(wa[key].values(), wb[key].values()):
            assert _torch.allclose(pa, pb, rtol=0, atol=0)
    # best-selection episode is recorded.
    assert res["best_selection"] is not None
    assert "episode" in res["best_selection"]


def test_resume_checkpoint_uses_only_replay_size_entries(tmp_path):
    """The replay sidecar stores only [0:size] transitions, not full capacity."""
    from experiments.checkpointing import _load_replay_sidecar
    cfg = _cfg(tmp_path, total_episodes=3)
    # Tiny episodes -> size well below the 1000 buffer capacity.
    res = train_maddpg(cfg, run_name="mad_sz",
                       log_dir=cfg["training"]["log_dir"],
                       ckpt_dir=cfg["training"]["ckpt_dir"])
    sidecar = res["latest_ckpt"] + ".replay.npz"
    assert os.path.exists(sidecar)
    replay = _load_replay_sidecar(sidecar)
    size = int(replay["size"])
    assert size < int(replay["capacity"])
    for arr in replay["obs"]:
        assert arr.shape[0] == size, "replay saved more than size entries"
    assert replay["dones"].shape[0] == size


def test_inference_checkpoint_roundtrip(tmp_path):
    cfg = _cfg(tmp_path, total_episodes=2)
    res = train_maddpg(cfg, run_name="mad_inf",
                       log_dir=cfg["training"]["log_dir"],
                       ckpt_dir=cfg["training"]["ckpt_dir"])
    agent = res["agent"]
    path = os.path.join(str(tmp_path), "inference.pt")
    agent.save_inference(path, extra_meta={"note": "unit"})

    from env import StarRisRsmaEnv
    from algorithms import MADDPG
    from utils import ObservationNormalizer
    env = StarRisRsmaEnv(cfg["env"], seed=1)
    agent2 = MADDPG(env.spec(), hidden_sizes=cfg["networks"]["hidden_sizes"],
                    maddpg_cfg=cfg["maddpg"], net_cfg=cfg["networks"],
                    device="cpu", seed=0)
    agent2.attach_obs_normalizers(
        [ObservationNormalizer(shape=(d,)) for d in env.spec().obs_dims])
    agent2.load_inference(path)

    env.reset(seed=123)
    raw = env.per_agent_observations()
    a1 = agent.select_actions(raw, explore=False)
    a2 = agent2.select_actions(raw, explore=False)
    for x, y in zip(a1, a2):
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-7)
