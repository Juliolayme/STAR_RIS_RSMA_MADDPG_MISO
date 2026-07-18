"""Source-tree provenance (V4 review P0-1) and aggregate matrix checks (P0-4)."""
from __future__ import annotations
import os

import pytest

from main import source_tree_sha256, validate_shard_group, DEFAULT_ALGOS


# ------------------------------------------------------------ source hashing
def _make_tree(root, files: dict[str, str]):
    for rel, content in files.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path) or root, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


def test_source_hash_deterministic_and_scoped(tmp_path):
    root = str(tmp_path)
    _make_tree(root, {
        "main.py": "print(1)\n",
        "pkg/mod.py": "x = 2\n",
        "config/config.yaml": "a: 1\n",
        "requirements.txt": "numpy\n",
        # Generated artifacts: must NOT affect the hash.
        "results_revised/run_x/tables/foo.csv": "1,2\n",
        "logs/run/log.csv": "a\n",
        "checkpoints/best.pt": "bin\n",
        "__pycache__/mod.cpython-310.pyc": "bin\n",
        "notes.md": "docs are not source\n",
    })
    h1 = source_tree_sha256(root)
    h2 = source_tree_sha256(root)
    assert h1 == h2, "hash must be deterministic"
    # Changing generated artifacts leaves the hash unchanged.
    _make_tree(root, {"results_revised/run_x/tables/foo.csv": "3,4\n",
                      "logs/run/log.csv": "b\n"})
    assert source_tree_sha256(root) == h1
    # Changing a source file changes the hash.
    _make_tree(root, {"pkg/mod.py": "x = 3\n"})
    assert source_tree_sha256(root) != h1


def test_source_hash_covers_config_and_requirements(tmp_path):
    root = str(tmp_path)
    _make_tree(root, {"a.py": "1\n", "config/c.yaml": "k: 1\n",
                      "requirements.txt": "torch\n"})
    h0 = source_tree_sha256(root)
    _make_tree(root, {"config/c.yaml": "k: 2\n"})
    h1 = source_tree_sha256(root)
    assert h1 != h0
    _make_tree(root, {"requirements.txt": "torch==2\n"})
    assert source_tree_sha256(root) != h1


# ------------------------------------------------------------ shard-group checks
def _record(algo, seed, config_sha="cfg", source_sha="src"):
    return {"algorithm": algo, "training_seed": seed,
            "config_sha": config_sha, "source_sha": source_sha}


def _cfg(algos=None, seeds=None):
    return {"evaluation": {"required_algorithms": algos,
                           "required_training_seeds": seeds,
                           "ao_solver_frozen_on_validation": True},
            "training": {"training_seeds": seeds or [],
                         "model_select_constraint_tolerance_frozen_on_validation": True}}


def test_mixed_source_sha_rejected_even_with_same_config():
    shards = [_record("maddpg", 1000, source_sha="srcA"),
              _record("maddpg", 2000, source_sha="srcB")]
    with pytest.raises(RuntimeError, match="different source trees"):
        validate_shard_group(shards, _cfg(["maddpg"], [1000, 2000]),
                             final_paper=False, allow_partial=True)


def test_missing_seed_rejected_without_allow_partial():
    shards = [_record("maddpg", 1000)]
    with pytest.raises(RuntimeError, match="missing seeds"):
        validate_shard_group(shards, _cfg(["maddpg"], [1000, 2000]),
                             final_paper=False, allow_partial=False)


def test_missing_algorithm_rejected_without_allow_partial():
    shards = [_record("maddpg", 1000)]
    with pytest.raises(RuntimeError, match="missing algorithms"):
        validate_shard_group(shards, _cfg(["maddpg", "td3"], [1000]),
                             final_paper=False, allow_partial=False)


def test_unregistered_seed_rejected():
    shards = [_record("maddpg", 1000), _record("maddpg", 9999)]
    with pytest.raises(RuntimeError, match="unregistered seeds"):
        validate_shard_group(shards, _cfg(["maddpg"], [1000]),
                             final_paper=False, allow_partial=False)


def test_allow_partial_accepts_pilot_subset():
    shards = [_record("maddpg", 1000)]
    algos, seeds = validate_shard_group(shards, _cfg(["maddpg", "td3"], [1000, 2000]),
                                        final_paper=False, allow_partial=True)
    assert algos == ["maddpg"] and seeds == [1000]


def test_final_paper_requires_preregistered_matrix():
    shards = [_record("maddpg", 1000)]
    with pytest.raises(RuntimeError, match="pre-registered"):
        validate_shard_group(shards, {"evaluation": {}, "training": {}},
                             final_paper=True, allow_partial=False)


@pytest.mark.parametrize("missing", ["tolerance", "ao"])
def test_final_paper_requires_validation_freezes(missing):
    cfg = _cfg(["maddpg"], [1000])
    if missing == "tolerance":
        cfg["training"]["model_select_constraint_tolerance_frozen_on_validation"] = False
        message = "model-selection tolerance"
    else:
        cfg["evaluation"]["ao_solver_frozen_on_validation"] = False
        message = "AO solver hyperparameters"
    with pytest.raises(RuntimeError, match=message):
        validate_shard_group([_record("maddpg", 1000)], cfg,
                             final_paper=True, allow_partial=False)


def test_final_paper_accepts_exact_matrix():
    algos = ["maddpg", "td3"]
    seeds = [1000, 2000]
    shards = [_record(a, s) for a in algos for s in seeds]
    got_algos, got_seeds = validate_shard_group(
        shards, _cfg(algos, seeds), final_paper=True, allow_partial=False)
    assert set(got_algos) == set(algos) and got_seeds == seeds


def test_final_paper_rejects_missing_source_sha():
    shards = [_record("maddpg", 1000, source_sha="")]
    with pytest.raises(RuntimeError, match="no source_sha"):
        validate_shard_group(shards, _cfg(["maddpg"], [1000]),
                             final_paper=True, allow_partial=False)


def test_final_paper_rejects_aggregator_source_mismatch():
    shards = [_record("maddpg", 1000, source_sha="training-source")]
    with pytest.raises(
            RuntimeError,
            match="Final-paper aggregation source differs from training source"):
        validate_shard_group(
            shards, _cfg(["maddpg"], [1000]), final_paper=True,
            allow_partial=False, current_source_sha="evaluation-source")


def test_nonpaper_source_mismatch_warns_and_remains_auditable():
    shards = [_record("maddpg", 1000, source_sha="training-source")]
    with pytest.warns(RuntimeWarning, match="differs from the shards' training source"):
        validate_shard_group(
            shards, _cfg(["maddpg"], [1000]), final_paper=False,
            allow_partial=False, current_source_sha="evaluation-source")


def test_default_algo_order_is_canonical():
    algos = list(DEFAULT_ALGOS)
    seeds = [1000]
    shards = [_record(a, 1000) for a in reversed(algos)]
    got_algos, _ = validate_shard_group(shards, _cfg(algos, seeds),
                                        final_paper=False, allow_partial=False)
    assert got_algos == algos
