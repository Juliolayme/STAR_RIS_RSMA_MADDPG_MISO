"""Shard workflow integrity (V4 review P0-1/P0-2/P0-3, items 8/9):
train-only -> hash-verified manifest with source/environment/history ->
aggregate-only regenerates convergence figures from shard histories, and the
locked test ScenarioBank is never materialized during training."""
from __future__ import annotations
import glob
import json
import os

import pandas as pd
import torch
import yaml

import env.scenario_bank as sbank
from conftest import full_cfg


def _tiny_cfg_path(tmp_path):
    cfg = full_cfg()
    cfg["training"]["total_episodes"] = 3
    cfg["training"]["eval_every"] = 2
    cfg["training"]["training_seeds"] = [1000]
    cfg["evaluation"]["num_episodes"] = 1
    cfg["evaluation"]["validation_seeds"] = [101]
    cfg["evaluation"]["test_seeds"] = [303]
    cfg["evaluation"]["power_sweep_dbm"] = [30]
    cfg["evaluation"]["ao_local_search_max_n"] = 0
    path = os.path.join(str(tmp_path), "tiny.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def test_train_only_shard_then_aggregate(tmp_path, monkeypatch):
    import main as M

    # Record every scenario split that gets materialized anywhere.
    built_splits: list[str] = []
    orig_generate = sbank.generate_scenario

    def recording_generate(env_cfg, split, evaluation_seed, episode_idx):
        built_splits.append(split)
        return orig_generate(env_cfg, split, evaluation_seed, episode_idx)

    monkeypatch.setattr(sbank, "generate_scenario", recording_generate)

    cfg_path = _tiny_cfg_path(tmp_path)
    out = str(tmp_path / "out")

    # ---------------- 1) train-only shard (one algorithm-seed pair)
    M.main(["--config", cfg_path, "--train-only", "--algos", "maddpg",
            "--seeds", "1000", "--run-id", "grp", "--out", out])

    # Locked-test isolation (P0-3): the training job built ONLY validation
    # scenarios -- no test split was materialized in this process phase.
    assert built_splits, "no scenario bank was built at all"
    assert set(built_splits) == {"validation"}, \
        f"train-only materialized non-validation splits: {set(built_splits)}"

    shard_dir = os.path.join(out, "results_revised", "shards", "grp", "maddpg_seed1000")
    with open(os.path.join(shard_dir, "shard_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["status"] == "completed"

    # Source provenance (P0-1) + execution environment (item 8).
    expect_source = M.source_tree_sha256()
    assert manifest["source_sha"] == expect_source
    env_meta = manifest["environment"]
    for key in ("python_version", "torch_version", "numpy_version",
                "cuda_version", "gpu_name", "platform", "source_sha"):
        assert key in env_meta
    assert env_meta["source_sha"] == expect_source

    # Training history travels with the shard, hash-verified (P0-2).
    hist_path = os.path.join(shard_dir, manifest["history_csv"])
    assert os.path.isfile(hist_path)
    assert M.sha256_file(hist_path) == manifest["history_sha"]
    log_path = os.path.join(shard_dir, manifest["log_csv"])
    assert os.path.isfile(log_path)
    assert M.sha256_file(log_path) == manifest["log_sha"]

    # Inference checkpoint metadata carries the source sha (P0-1).
    best = os.path.join(shard_dir, manifest["best_checkpoint"])
    payload = torch.load(best, map_location="cpu", weights_only=False)
    assert payload["meta"]["source_sha"] == expect_source

    # ---------------- 2) aggregate-only from the shard manifest
    built_splits.clear()
    M.main(["--aggregate-only", "--load-shards",
            os.path.join(out, "results_revised", "shards", "grp"),
            "--allow-partial", "--run-id", "grp_agg", "--out", out])
    agg = os.path.join(out, "results_revised", "grp_agg")

    # The aggregate job (and only it) may build the test bank.
    assert "test" in set(built_splits)

    # Convergence figures are REGENERATED from the shard histories (P0-2).
    for name in ("training_convergence", "training_sum_rate",
                 "training_user_qos_fraction", "qos_lambda",
                 "reward_decomposition"):
        hits = glob.glob(os.path.join(agg, "figures", name + ".*"))
        assert hits, f"aggregate-only did not produce {name}"

    # Provenance columns in aggregate outputs (P0-1).
    completed = pd.read_csv(os.path.join(agg, "completed_runs.csv"))
    assert "source_sha" in completed.columns
    raw = pd.read_csv(os.path.join(agg, "tables", "results_raw.csv"))
    assert "source_sha" not in raw.columns
    assert (raw["training_source_sha"] == expect_source).all()
    assert (raw["evaluation_source_sha"] == expect_source).all()
    with open(os.path.join(agg, "run_meta.json"), encoding="utf-8") as f:
        run_meta = json.load(f)
    assert run_meta["source_sha"] == expect_source
    assert run_meta["training_source_sha"] == expect_source
    assert run_meta["evaluation_source_sha"] == expect_source


def test_aggregate_without_allow_partial_rejects_incomplete_matrix(tmp_path):
    import main as M
    cfg_path = _tiny_cfg_path(tmp_path)
    out = str(tmp_path / "out")
    M.main(["--config", cfg_path, "--train-only", "--algos", "maddpg",
            "--seeds", "1000", "--run-id", "grp2", "--out", out])
    import pytest
    with pytest.raises(RuntimeError, match="pre-registered"):
        M.main(["--aggregate-only", "--load-shards",
                os.path.join(out, "results_revised", "shards", "grp2"),
                "--run-id", "grp2_agg", "--out", out])


def test_drop_replay_updates_manifest_resumable_flag(tmp_path):
    import main as M
    cfg_path = _tiny_cfg_path(tmp_path)
    out = str(tmp_path / "out")
    M.main(["--config", cfg_path, "--train-only", "--algos", "maddpg",
            "--seeds", "1000", "--run-id", "grp3", "--out", out,
            "--drop-replay-after-train"])
    shard_dir = os.path.join(out, "results_revised", "shards", "grp3", "maddpg_seed1000")
    with open(os.path.join(shard_dir, "shard_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    # Replay sidecar was removed -> the shard must no longer claim exact
    # resumability (V4 review item 9).
    assert manifest["replay_dropped"] is True
    assert manifest["resumable_training"] is False
    latest = os.path.join(shard_dir, manifest["latest_checkpoint"])
    assert not os.path.exists(latest + ".replay.npz")
