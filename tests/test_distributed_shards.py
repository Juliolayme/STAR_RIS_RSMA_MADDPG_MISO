"""Safe train-only shard and manifest-driven aggregation primitives."""
from __future__ import annotations
import json
import os

import yaml

from conftest import full_cfg


def _write_tiny_config(tmp_path):
    cfg = full_cfg(max_steps=2)
    cfg["training"]["total_episodes"] = 1
    cfg["training"]["eval_every"] = 1000
    cfg["training"]["checkpoint_every"] = 1
    cfg["training"]["training_seeds"] = [1000, 2000]
    cfg["evaluation"]["validation_seeds"] = [101]
    cfg["evaluation"]["test_seeds"] = [303]
    cfg["evaluation"]["ao_local_search_max_n"] = 0
    path = tmp_path / "tiny.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def test_train_only_is_one_pair_and_writes_no_aggregate_outputs(tmp_path):
    import main as M

    cfg_path = _write_tiny_config(tmp_path)
    out = tmp_path / "out"
    M.main(["--config", str(cfg_path), "--train-only", "--algos", "maddpg",
            "--seeds", "1000", "--run-id", "unit_shards", "--out", str(out)])

    shard = out / "results_revised" / "shards" / "unit_shards" / "maddpg_seed1000"
    manifest_path = shard / M.SHARD_MANIFEST_NAME
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["status"] == "completed"
    assert manifest["algorithm"] == "maddpg"
    assert manifest["training_seed"] == 1000
    assert len(manifest["config_sha"]) == 64
    assert len(manifest["checkpoint_sha"]) == 64
    assert not (shard / "run_meta.json").exists()
    assert not (shard / "tables").exists()
    assert not (shard / "figures").exists()
    assert not (out / "results_revised" / "LATEST_RUN.txt").exists()

    records = M.scan_shard_manifests([str(out / "results_revised" / "shards")])
    assert len(records) == 1 and records[0]["_verified"]
    frame = M.completed_runs_frame(records)
    assert list(frame.columns) == M.COMPLETED_RUN_COLUMNS
    assert frame.loc[0, "status"] == "completed"


def test_shard_scanner_ignores_run_meta_and_rejects_checkpoint_tamper(tmp_path):
    import main as M

    root = tmp_path / "shard"
    root.mkdir()
    config_path = root / "effective_config.yaml"
    checkpoint_path = root / "best.pt"
    config_path.write_bytes(b"seed: 1\n")
    checkpoint_path.write_bytes(b"checkpoint-v1")
    manifest = {
        "algorithm": "td3", "training_seed": 1000, "status": "completed",
        "config_sha": M.sha256_file(str(config_path)),
        "effective_config": "effective_config.yaml",
        "best_checkpoint": "best.pt",
        "checkpoint_sha": M.sha256_file(str(checkpoint_path)),
    }
    with open(root / M.SHARD_MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with open(root / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump({"algorithm": "malicious", "training_seed": 9999}, f)

    records = M.scan_shard_manifests([str(root)])
    assert records[0]["algorithm"] == "td3"
    assert records[0]["_verified"]

    checkpoint_path.write_bytes(b"tampered")
    records = M.scan_shard_manifests([str(root)])
    assert records[0]["status"] == "invalid"
    assert not records[0]["_verified"]


def test_load_shards_is_input_run_dirs_alias():
    import main as M

    args = M.parse_args(["--aggregate-only", "--load-shards", "a", "b"])
    assert args.aggregate_only
    assert args.input_run_dirs == ["a", "b"]
    assert "td3_matched" in M.parse_args([]).algos
