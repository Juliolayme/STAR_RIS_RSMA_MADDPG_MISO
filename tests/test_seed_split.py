"""The versioned seed split and its SHA are the synchronized source of truth."""
from __future__ import annotations
import os

import yaml


def test_versioned_seed_split_hash_and_documentation_are_synchronized():
    import main as M

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(root, "config", "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    spec = raw_cfg["evaluation"]["seed_split"]
    split_path = os.path.join(root, "config", spec["path"])
    actual_sha = M.sha256_file(split_path)
    assert actual_sha == spec["sha256"]

    resolved = M.load_config(config_path)
    with open(split_path, encoding="utf-8") as f:
        split = yaml.safe_load(f)
    for key, seeds in split["splits"].items():
        assert resolved["evaluation"][key] == seeds
    assert resolved["evaluation"]["seed_split"]["resolved_sha256"] == actual_sha

    for rel in ("KAGGLE_RERUN_CHECKLIST.md",
                os.path.join("latex_thesis", "chapters", "05_chuong_4.tex")):
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            assert actual_sha in f.read()

