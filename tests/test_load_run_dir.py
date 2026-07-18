"""End-to-end skip-train evaluation from an existing run directory (item 5):
a second process loads completed checkpoints/normalizers and reruns evaluation
without retraining."""
from __future__ import annotations
import glob
import os

import yaml

from conftest import full_cfg


def _tiny_cfg(tmp_path):
    cfg = full_cfg()
    cfg["training"]["total_episodes"] = 3
    cfg["training"]["eval_every"] = 2
    cfg["training"]["training_seeds"] = [1000]
    cfg["evaluation"]["num_episodes"] = 1
    cfg["evaluation"]["dev_seeds"] = [101]
    cfg["evaluation"]["validation_seeds"] = [101]
    cfg["evaluation"]["test_seeds"] = [303]
    cfg["evaluation"]["power_sweep_dbm"] = [30]
    cfg["evaluation"]["ao_local_search_max_n"] = 0     # keep the smoke fast
    path = os.path.join(str(tmp_path), "tiny.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def test_load_run_dir_reevaluates_without_training(tmp_path):
    import main as M

    cfg_path = _tiny_cfg(tmp_path)
    out = str(tmp_path / "out")

    # 1) Train + evaluate a tiny run.
    M.main(["--config", cfg_path, "--algos", "maddpg", "--out", out,
            "--run-id", "unit_run"])
    train_dir = os.path.join(out, "results_revised", "unit_run")
    assert os.path.exists(os.path.join(train_dir, "tables", "algorithm_comparison.csv"))
    assert os.path.exists(os.path.join(train_dir, "run_meta.json"))
    best = os.path.join(out, "ckpt_test", "unit_run", "maddpg_seed1000", "best.pt")
    assert os.path.exists(best)
    best_mtime = os.path.getmtime(best)

    # 2) Re-evaluate from the run dir WITHOUT retraining.
    M.main(["--load-run-dir", train_dir, "--algos", "maddpg", "--out", out])

    # A new eval-only output dir was created and produced result tables.
    eval_dirs = glob.glob(os.path.join(out, "results_revised", "unit_run_eval_*"))
    assert eval_dirs, "load-run-dir did not create an eval output dir"
    assert os.path.exists(os.path.join(eval_dirs[0], "tables", "algorithm_comparison.csv"))
    # The checkpoint was not retrained (unchanged mtime).
    assert os.path.getmtime(best) == best_mtime
