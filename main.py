"""End-to-end pipeline: multi-seed train, ScenarioBank evaluation, ablation,
paired statistics, latency benchmarks, plots, tables, report.

Statistical design (P0-6/P0-7/P1-8 reviewer fixes):
- Learned methods: the unit of analysis is the TRAINING-SEED-LEVEL mean
  (each independently trained seed is evaluated on the full locked test
  ScenarioBank; the CI is computed across the training seeds).
- Non-learned baselines (AO-Grid, analytical, fixed, ...) are evaluated once;
  their uncertainty comes from the independent evaluation scenarios. They are
  never duplicated across training seeds.
- Paired tests (paired t + sign-flip permutation, Holm-Bonferroni over the
  primary family {MADDPG vs TD3, vs DDPG, vs PPO} on sum-rate) use the
  matched training-seed design; every algorithm trains on the same seed list
  and is evaluated on the identical scenarios.
- All new outputs go to results_revised/ (never mixed with results_legacy/).
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
import platform
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env import StarRisRsmaEnv, ScenarioBank, build_eval_banks
from experiments.train import (
    train_maddpg, train_single_agent, train_ppo, evaluate_agent, _make_env,
    validate_formulation_config, configure_cpu_threads, _select_device,
    set_checkpoint_provenance,
)
from experiments.evaluate import (
    eval_on_scenarios, scenario_rows, benchmark_latency_cpu, benchmark_latency_gpu,
)
from experiments.ablation import ablation_study, ABLATION_CELLS
from algorithms.complexity import (
    maddpg_param_counts, td3_param_counts, ddpg_param_counts, ppo_param_counts,
    matched_td3_hidden_sizes,
)
from utils.plotting import (
    plot_training_convergence, plot_metric_vs_x, plot_bar,
    plot_reward_decomposition, plot_qos_lambda,
    plot_phase_histogram, plot_h_eff_distribution, plot_pareto,
)
from utils import (
    confidence_interval, paired_t_test_p, paired_permutation_p,
    holm_bonferroni, cohens_d_paired, paired_difference_ci,
)

LEARNED_ALGOS = {"maddpg": "MADDPG", "ddpg": "DDPG", "td3": "TD3",
                 "ppo": "PPO", "td3_matched": "TD3-Matched"}
DEFAULT_ALGOS = ["maddpg", "ddpg", "td3", "td3_matched", "ppo"]
COMPLETED_RUN_COLUMNS = ["algorithm", "training_seed", "status", "config_sha",
                         "source_sha", "best_checkpoint", "checkpoint_sha"]
SHARD_MANIFEST_NAME = "shard_manifest.json"
# Holm-corrected primary family on sum-rate. TD3-Matched is a PRE-SPECIFIED
# primary comparison (item 8): the decomposition claim (gain from multi-agent
# factorisation, not parameter count) requires MADDPG to beat a total-parameter-
# matched single-agent TD3.
PRIMARY_FAMILY = ["TD3", "TD3-Matched", "DDPG", "PPO"]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # The paper configuration keeps its seed lists in one versioned file.  A
    # copied effective_config.yaml embeds the resolved lists and provenance, so
    # it remains self-contained when a shard is moved to another machine.
    evaluation = cfg.get("evaluation", {})
    split_spec = evaluation.get("seed_split")
    if split_spec:
        required = ("dev_seeds", "validation_seeds", "legacy_eval_seeds", "test_seeds")
        split_path = split_spec.get("path")
        expected_sha = str(split_spec.get("sha256", "")).lower()
        resolved_path = (split_path if os.path.isabs(str(split_path)) else
                         os.path.join(os.path.dirname(os.path.abspath(path)), str(split_path)))
        if os.path.exists(resolved_path):
            actual_sha = sha256_file(resolved_path)
            if not expected_sha or actual_sha != expected_sha:
                raise ValueError(
                    f"Seed-split SHA-256 mismatch for {resolved_path}: "
                    f"expected {expected_sha or '<missing>'}, got {actual_sha}")
            with open(resolved_path, "r", encoding="utf-8") as f:
                split_doc = yaml.safe_load(f)
            splits = split_doc.get("splits", {})
            missing = [k for k in required if k not in splits]
            if missing:
                raise ValueError(f"Seed-split file is missing keys: {missing}")
            for key in required:
                evaluation[key] = [int(s) for s in splits[key]]
            split_spec["version"] = int(split_doc["schema_version"])
            split_spec["resolved_sha256"] = actual_sha
        else:
            embedded = all(k in evaluation for k in required)
            if not (embedded and split_spec.get("resolved_sha256") == expected_sha):
                raise FileNotFoundError(f"Seed-split source not found: {resolved_path}")
    return cfg


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_SOURCE_EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".git", ".ipynb_checkpoints",
                         "results_legacy", "results_revised", "scalability_runs"}
_SOURCE_EXCLUDED_PREFIXES = ("results", "logs", "checkpoints", "ckpt")


def source_tree_sha256(root: str = PROJECT_ROOT) -> str:
    """Deterministic SHA-256 of the SOURCE tree (V4 review P0-1).

    Hashes, in sorted relative-path order: every *.py under `root`,
    config/*.yaml and requirements.txt. Never hashes results/, logs*/,
    checkpoints*/, __pycache__/ or other generated artifacts, so the value is
    stable across runs of identical code. Two shards trained with different
    implementations therefore carry different source_sha even when their
    config_sha is identical, and the aggregator refuses to mix them.
    """
    root = os.path.abspath(root)
    entries: list[str] = []
    for cur, dirs, files in os.walk(root):
        rel_dir = os.path.relpath(cur, root)
        dirs[:] = sorted(
            d for d in dirs
            if d not in _SOURCE_EXCLUDED_DIRS
            and not d.startswith(_SOURCE_EXCLUDED_PREFIXES))
        for fname in sorted(files):
            rel = os.path.normpath(os.path.join(rel_dir, fname)).replace(os.sep, "/")
            if rel.startswith("./"):
                rel = rel[2:]
            if fname.endswith(".py"):
                entries.append(rel)
            elif rel == "requirements.txt":
                entries.append(rel)
            elif rel.startswith("config/") and (fname.endswith(".yaml") or
                                                fname.endswith(".yml")):
                entries.append(rel)
    h = hashlib.sha256()
    for rel in sorted(entries):
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(os.path.join(root, rel)).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def runtime_environment_metadata(source_sha: str) -> dict:
    """Execution-environment provenance stored in every shard manifest
    (V4 review item 8)."""
    cuda_available = torch.cuda.is_available()
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_version": (torch.version.cuda or "") if cuda_available else "",
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else "",
        "platform": platform.platform(),
        "hostname": platform.node(),
        "source_sha": source_sha,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="STAR-RIS RSMA MADDPG end-to-end")
    p.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config", "config.yaml"))
    p.add_argument("--episodes", type=int, default=None,
                   help="Override training episodes per algorithm.")
    p.add_argument("--algos", nargs="+",
                   default=list(DEFAULT_ALGOS),
                   help="Algorithms to train (maddpg ddpg td3 ppo td3_matched).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true",
                      help="Train exactly one algorithm/seed into an isolated shard; "
                           "do not create aggregate metadata, tables, or figures.")
    mode.add_argument("--aggregate-only", action="store_true",
                      help="Discover and verify completed shard manifests, then evaluate "
                           "and aggregate them without training.")
    p.add_argument("--input-run-dirs", "--load-shards", dest="input_run_dirs",
                   nargs="+", default=None,
                   help="Shard directories, parent directories, or manifest files to scan "
                        "for --aggregate-only (the two option names are aliases).")
    p.add_argument("--skip-train", action="store_true",
                   help="Load best.pt inference checkpoints instead of training.")
    p.add_argument("--resume", default=None,
                   help="Path to a TrainingCheckpoint to resume MADDPG training from.")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: few episodes, tiny evaluation (uses dev seeds).")
    # --- distributed / separable jobs (item 7) ---
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Subset of training seeds to run (distributed training jobs).")
    p.add_argument("--run-id", default=None,
                   help="Fixed run id so parallel --seeds jobs share one run "
                        "directory (otherwise a config-hash+timestamp id is used).")
    p.add_argument("--load-run-dir", default=None,
                   help="Evaluate/aggregate an existing run: load its trained "
                        "checkpoints + normalizers and rerun eval/sweep/ablation/"
                        "plots WITHOUT retraining (item 5). Path to a "
                        "results_revised/run_* directory.")
    p.add_argument("--drop-replay-after-train", action="store_true",
                   help="Delete resumable replay sidecars after each successful "
                        "training run (best inference checkpoints are kept; the "
                        "shard manifest is updated to resumable_training=false).")
    strictness = p.add_mutually_exclusive_group()
    strictness.add_argument("--final-paper", action="store_true",
                            help="Aggregate-only strict mode: require the exact "
                                 "pre-registered algorithm x seed matrix "
                                 "(evaluation.required_algorithms x "
                                 "required_training_seeds) with uniform "
                                 "source/config SHAs; fail otherwise.")
    strictness.add_argument("--allow-partial", action="store_true",
                            help="Aggregate-only: explicitly allow an incomplete "
                                 "algorithm/seed matrix (smoke and pilot only).")
    p.add_argument("--out", default=PROJECT_ROOT)
    return p.parse_args(argv)


def validate_shard_group(verified_shards: list[dict], cfg: dict | None,
                         final_paper: bool, allow_partial: bool,
                         current_source_sha: str = "") -> tuple[list[str], list[int]]:
    """Cross-shard integrity checks for --aggregate-only (V4 review P0-1/P0-4).

    Always enforced: uniform config_sha, uniform source_sha (shards trained by
    different implementations must never be mixed even when configs match),
    no duplicate algorithm-seed pair, matched seed sets across algorithms.

    Completeness: unless --allow-partial, the shard set must equal the
    pre-registered matrix (evaluation.required_algorithms x
    required_training_seeds, falling back to DEFAULT_ALGOS x
    training.training_seeds). --final-paper additionally refuses to run
    without an explicit pre-registered matrix in the config.

    Returns (algorithms present in canonical order, sorted seed list).
    """
    config_shas = {r["config_sha"] for r in verified_shards}
    if len(config_shas) != 1:
        raise RuntimeError(f"Cannot aggregate shards with different config SHAs: {config_shas}")
    source_shas = {str(r.get("source_sha", "")) for r in verified_shards}
    if len(source_shas) != 1:
        raise RuntimeError(
            "Cannot aggregate shards trained by different source trees "
            f"(source_sha values: {sorted(source_shas)}). Retrain the divergent "
            "shards with one frozen implementation.")
    shard_source = next(iter(source_shas))
    if not shard_source:
        msg = ("Shard manifests carry no source_sha; provenance cannot be "
               "verified across shards.")
        if final_paper:
            raise RuntimeError(msg + " Final-paper aggregation refuses to proceed.")
        warnings.warn(msg, RuntimeWarning)
    elif current_source_sha and shard_source != current_source_sha:
        if final_paper:
            raise RuntimeError(
                "Final-paper aggregation source differs from training source.")
        warnings.warn(
            f"Aggregator source tree ({current_source_sha[:12]}) differs from the "
            f"shards' training source ({shard_source[:12]}); evaluation code is "
            "newer/older than training code.", RuntimeWarning)

    seen_pairs: set[tuple[str, int]] = set()
    seed_sets: dict[str, set[int]] = {}
    for record in verified_shards:
        pair = (record["algorithm"], int(record["training_seed"]))
        if pair in seen_pairs:
            raise RuntimeError(f"Duplicate completed shard for {pair}; choose one retry explicitly")
        seen_pairs.add(pair)
        seed_sets.setdefault(pair[0], set()).add(pair[1])
    if len({tuple(sorted(v)) for v in seed_sets.values()}) > 1:
        raise RuntimeError(f"Algorithms do not have the same matched training seeds: {seed_sets}")

    evaluation = (cfg or {}).get("evaluation", {})
    req_algos = evaluation.get("required_algorithms")
    req_seeds = evaluation.get("required_training_seeds")
    if final_paper and (not req_algos or not req_seeds):
        raise RuntimeError(
            "--final-paper requires evaluation.required_algorithms and "
            "evaluation.required_training_seeds to be pre-registered in the config.")
    if final_paper:
        training_cfg = (cfg or {}).get("training", {})
        if training_cfg.get("model_select_constraint_tolerance_frozen_on_validation") is not True:
            raise RuntimeError(
                "--final-paper requires model-selection tolerance to be frozen on validation.")
        if evaluation.get("ao_solver_frozen_on_validation") is not True:
            raise RuntimeError(
                "--final-paper requires AO solver hyperparameters to be frozen on validation.")
    req_algos = [str(a) for a in (req_algos or DEFAULT_ALGOS)]
    req_seeds = [int(s) for s in
                 (req_seeds or (cfg or {}).get("training", {}).get("training_seeds", []))]
    if not allow_partial:
        have_algos = set(seed_sets)
        want_algos = set(req_algos)
        have_seeds = next(iter(seed_sets.values())) if seed_sets else set()
        want_seeds = set(req_seeds)
        problems = []
        if want_algos - have_algos:
            problems.append(f"missing algorithms: {sorted(want_algos - have_algos)}")
        if have_algos - want_algos:
            problems.append(f"unregistered algorithms: {sorted(have_algos - want_algos)}")
        if want_seeds - have_seeds:
            problems.append(f"missing seeds: {sorted(want_seeds - have_seeds)}")
        if have_seeds - want_seeds:
            problems.append(f"unregistered seeds: {sorted(have_seeds - want_seeds)}")
        if problems:
            raise RuntimeError(
                "Aggregate matrix does not match the pre-registered experiment "
                f"({'; '.join(problems)}). Pass --allow-partial ONLY for "
                "smoke/pilot aggregation; final paper tables require the full matrix.")

    algos = [a for a in DEFAULT_ALGOS if a in seed_sets]
    algos.extend(sorted(set(seed_sets) - set(algos)))
    seeds = sorted(next(iter(seed_sets.values()))) if seed_sets else []
    return algos, seeds


def _config_bytes_and_sha(cfg: dict) -> tuple[bytes, str]:
    data = yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True).encode("utf-8")
    return data, hashlib.sha256(data).hexdigest()


def _atomic_write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + f".tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _manifest_path_value(shard_dir: str, value: str | None) -> str:
    if not value:
        return ""
    return os.path.abspath(value if os.path.isabs(value) else os.path.join(shard_dir, value))


def scan_shard_manifests(input_paths: list[str]) -> list[dict]:
    """Discover shard manifests without consulting run_meta.json.

    Completed entries are cryptographically verified against both their
    effective config and best checkpoint.  Incomplete/corrupt entries remain
    visible in completed_runs.csv with a non-completed status, but are never
    loaded for evaluation.
    """
    manifest_paths: set[str] = set()
    for raw in input_paths:
        path = os.path.abspath(raw)
        if os.path.isfile(path):
            if os.path.basename(path) == SHARD_MANIFEST_NAME:
                manifest_paths.add(path)
            continue
        if not os.path.isdir(path):
            warnings.warn(f"Shard input does not exist and was skipped: {path}", RuntimeWarning)
            continue
        direct = os.path.join(path, SHARD_MANIFEST_NAME)
        if os.path.isfile(direct):
            manifest_paths.add(direct)
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache")]
            if SHARD_MANIFEST_NAME in files:
                manifest_paths.add(os.path.join(root, SHARD_MANIFEST_NAME))

    records: list[dict] = []
    for manifest_path in sorted(manifest_paths):
        shard_dir = os.path.dirname(manifest_path)
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError) as exc:
            records.append({
                "algorithm": "", "training_seed": "", "status": "invalid_manifest",
                "config_sha": "", "best_checkpoint": "", "checkpoint_sha": "",
                "_manifest_path": manifest_path, "_verified": False,
                "_error": str(exc),
            })
            continue

        best_path = _manifest_path_value(shard_dir, meta.get("best_checkpoint"))
        cfg_path = _manifest_path_value(shard_dir, meta.get("effective_config"))
        declared_status = str(meta.get("status", "unknown"))
        status = declared_status
        actual_ck_sha = sha256_file(best_path) if best_path and os.path.isfile(best_path) else ""
        actual_cfg_sha = sha256_file(cfg_path) if cfg_path and os.path.isfile(cfg_path) else ""
        verified = declared_status == "completed"
        error = ""
        if verified and not actual_ck_sha:
            verified, status, error = False, "invalid", "best checkpoint is missing"
        elif verified and actual_ck_sha != str(meta.get("checkpoint_sha", "")):
            verified, status, error = False, "invalid", "checkpoint SHA-256 mismatch"
        elif verified and not actual_cfg_sha:
            verified, status, error = False, "invalid", "effective config is missing"
        elif verified and actual_cfg_sha != str(meta.get("config_sha", "")):
            verified, status, error = False, "invalid", "config SHA-256 mismatch"

        records.append({
            **meta,
            "algorithm": str(meta.get("algorithm", "")),
            "training_seed": meta.get("training_seed", ""),
            "status": status,
            "config_sha": str(meta.get("config_sha", "")),
            "best_checkpoint": best_path,
            "checkpoint_sha": actual_ck_sha or str(meta.get("checkpoint_sha", "")),
            "_effective_config": cfg_path,
            "_manifest_path": manifest_path,
            "_verified": verified,
            "_error": error,
        })
    return records


def completed_runs_frame(records: list[dict]) -> pd.DataFrame:
    rows = [{k: record.get(k, "") for k in COMPLETED_RUN_COLUMNS}
            for record in records]
    return pd.DataFrame(rows, columns=COMPLETED_RUN_COLUMNS)


def _save_history_csv(path: str, history: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = {}
    for k, v in history.items():
        if k == "qos_lambda_per_user" and len(v):
            lam = np.stack(v, axis=0)
            for u in range(lam.shape[1]):
                cols[f"qos_lambda_{u}"] = lam[:, u]
        elif hasattr(v, "__len__"):
            cols[k] = v
    pd.DataFrame(cols).to_csv(path, index=False)


def _history_from_csv(path: str) -> dict:
    """Inverse of _save_history_csv: per-episode history columns as lists."""
    df = pd.read_csv(path)
    return {col: df[col].tolist() for col in df.columns}


def _load_shard_history(record: dict) -> tuple[dict, str]:
    """Load and hash-verify a shard's training history (V4 review P0-2).

    Returns (history dict, absolute log.csv path or ""). Raises when the
    manifest lacks the history or the on-disk file does not match its
    registered SHA-256 -- aggregate-only must not silently drop the
    convergence figures.
    """
    shard_dir = os.path.dirname(record["_manifest_path"])
    hist_path = _manifest_path_value(shard_dir, record.get("history_csv"))
    hist_sha = str(record.get("history_sha", ""))
    if not hist_path or not os.path.isfile(hist_path):
        raise RuntimeError(
            f"Shard {record.get('shard_id', shard_dir)} has no training history; "
            "re-run the shard with the current pipeline (manifest must register "
            "history_csv/history_sha).")
    actual = sha256_file(hist_path)
    if actual != hist_sha:
        raise RuntimeError(
            f"history.csv SHA-256 mismatch for shard {record.get('shard_id', shard_dir)}: "
            f"manifest {hist_sha}, file {actual}")
    log_path = _manifest_path_value(shard_dir, record.get("log_csv"))
    if log_path and os.path.isfile(log_path):
        log_sha = str(record.get("log_sha", ""))
        actual_log = sha256_file(log_path)
        if log_sha and actual_log != log_sha:
            raise RuntimeError(
                f"log.csv SHA-256 mismatch for shard {record.get('shard_id', shard_dir)}: "
                f"manifest {log_sha}, file {actual_log}")
    else:
        log_path = ""
    return _history_from_csv(hist_path), log_path


def _train_algorithm(cfg: dict, algo_key: str, seed: int, log_dir: str,
                     ckpt_dir: str, resume_from: str | None = None) -> dict:
    """Train one algorithm/seed pair using deterministic, pair-owned paths."""
    run_name = f"{algo_key}_seed{seed}"
    if algo_key == "maddpg":
        return train_maddpg(cfg, log_dir=log_dir, ckpt_dir=ckpt_dir,
                            seed_override=seed, run_name=run_name,
                            resume_from=resume_from)
    if algo_key == "ddpg":
        return train_single_agent(cfg, kind="ddpg", log_dir=log_dir,
                                  ckpt_dir=ckpt_dir, seed_override=seed,
                                  run_name=run_name)
    if algo_key in ("td3", "td3_matched"):
        hidden = None
        if algo_key == "td3_matched":
            env = _make_env(cfg, seed)
            target = maddpg_param_counts(env.spec(), cfg["networks"]["hidden_sizes"],
                                         cfg["networks"])["total_params"]
            hidden = matched_td3_hidden_sizes(env.observation_space.shape[0],
                                              env.action_space.shape[0], target,
                                              cfg["networks"])
        return train_single_agent(cfg, kind=algo_key, log_dir=log_dir,
                                  ckpt_dir=ckpt_dir, seed_override=seed,
                                  run_name=run_name,
                                  hidden_sizes_override=hidden)
    if algo_key == "ppo":
        return train_ppo(cfg, log_dir=log_dir, ckpt_dir=ckpt_dir,
                         seed_override=seed, run_name=run_name)
    raise ValueError(f"Unknown algorithm: {algo_key}")


def _run_train_only(args, cfg: dict, config_bytes: bytes, config_sha: str,
                    source_sha: str) -> None:
    """Run one interruption-bounded Kaggle job and emit one shard manifest."""
    selected_seeds = list(args.seeds or [])
    if len(args.algos) != 1 or len(selected_seeds) != 1:
        raise SystemExit(
            "--train-only enforces one algorithm-seed per job. Pass exactly one "
            "--algos value and one --seeds value; aggregate jobs later with "
            "--aggregate-only --input-run-dirs/--load-shards.")
    algo_key = args.algos[0]
    seed = int(selected_seeds[0])
    if algo_key not in LEARNED_ALGOS:
        raise SystemExit(f"Unsupported --train-only algorithm: {algo_key}")
    if args.resume and algo_key != "maddpg":
        raise SystemExit("--resume is currently supported only for MADDPG; the one-pair "
                         "job rule bounds interruption loss for DDPG/TD3/PPO.")

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    group_id = args.run_id or f"shards_{config_sha[:8]}_{stamp}"
    shard_root = os.path.join(args.out, "results_revised", "shards", group_id,
                              f"{algo_key}_seed{seed}")
    manifest_path = os.path.join(shard_root, SHARD_MANIFEST_NAME)
    if os.path.exists(shard_root):
        raise FileExistsError(
            f"Shard output already exists and will not be overwritten: {shard_root}. "
            "Use a new --run-id for a retry.")
    os.makedirs(shard_root, exist_ok=False)
    effective_path = os.path.join(shard_root, "effective_config.yaml")
    with open(effective_path, "wb") as f:
        f.write(config_bytes)

    manifest = {
        "schema_version": 2,
        "shard_id": f"{group_id}:{algo_key}:seed{seed}",
        "run_group": group_id,
        "algorithm": algo_key,
        "algorithm_label": LEARNED_ALGOS[algo_key],
        "training_seed": seed,
        "status": "running",
        "config_sha": config_sha,
        # Source provenance (V4 review P0-1): aggregation refuses to mix shards
        # whose source trees differ, even under identical configs.
        "source_sha": source_sha,
        # Execution environment (V4 review item 8).
        "environment": runtime_environment_metadata(source_sha),
        "effective_config": "effective_config.yaml",
        "best_checkpoint": "",
        "latest_checkpoint": "",
        "checkpoint_sha": "",
        "history_csv": "",
        "history_sha": "",
        "log_csv": "",
        "log_sha": "",
        "resumable_training": algo_key == "maddpg",
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    _atomic_write_json(manifest_path, manifest)
    log_dir = os.path.join(shard_root, "logs")
    ckpt_dir = os.path.join(shard_root, "checkpoints")
    run_name = f"{algo_key}_seed{seed}"
    print(f"Training-only shard: {manifest['shard_id']}")
    print("Only the validation ScenarioBank may be used by this job; no test bank is built.")
    set_checkpoint_provenance({"source_sha": source_sha, "config_sha": config_sha})
    try:
        info = _train_algorithm(cfg, algo_key, seed, log_dir, ckpt_dir,
                                resume_from=args.resume)
        history_path = os.path.join(log_dir, run_name, "history.csv")
        _save_history_csv(history_path, info["history"])
        log_path = os.path.join(log_dir, run_name, "log.csv")
        resumable = algo_key == "maddpg"
        replay_dropped = False
        if args.drop_replay_after_train:
            sidecar = info.get("latest_ckpt", "") + ".replay.npz"
            if sidecar and os.path.exists(sidecar):
                os.remove(sidecar)
                replay_dropped = True
                # Without its replay sidecar the checkpoint is no longer
                # exactly resumable (V4 review item 9).
                resumable = False
        best_path = os.path.abspath(info["best_ckpt"])
        latest_path = os.path.abspath(info["latest_ckpt"])
        manifest.update({
            "status": "completed",
            "best_checkpoint": os.path.relpath(best_path, shard_root),
            "latest_checkpoint": os.path.relpath(latest_path, shard_root),
            "checkpoint_sha": sha256_file(best_path),
            # Training histories travel with the shard so aggregate-only can
            # rebuild convergence figures (V4 review P0-2).
            "history_csv": os.path.relpath(os.path.abspath(history_path), shard_root),
            "history_sha": sha256_file(history_path),
            "log_csv": (os.path.relpath(os.path.abspath(log_path), shard_root)
                        if os.path.exists(log_path) else ""),
            "log_sha": sha256_file(log_path) if os.path.exists(log_path) else "",
            "resumable_training": resumable,
            "replay_dropped": replay_dropped,
            "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        _atomic_write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest.update({
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        _atomic_write_json(manifest_path, manifest)
        raise
    print(f"Completed shard manifest: {manifest_path}")


def _write_tex_table(path: str, df: pd.DataFrame, caption: str, label: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tex = df.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.3f}")
    tex = (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
           + tex + "\n\\end{table}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)


def _seed_matrix(runs: list[dict], metric: str) -> np.ndarray:
    mat = [np.asarray(info["history"][metric], dtype=float) for info in runs]
    min_len = min(len(v) for v in mat)
    return np.stack([v[:min_len] for v in mat], axis=0)


def _collect_phase_and_heff_samples(runs, cfg, n_steps=200):
    """Aggregate phases + |h_eff_T| across ALL training seeds per RIS mode."""
    out_phase: dict[str, list] = {}
    out_heff: dict[str, list] = {}
    for label, ris_mode in [("Learned", "optimized"), ("AnalyticalRIS", "analytical"),
                            ("FixedRIS", "fixed"), ("RandomRIS", "random")]:
        phases, heff_T = [], []
        for run in runs:
            agent = run["agent"]
            env = _make_env(cfg, seed=int(cfg["seed"]) + 7777, ris_mode=ris_mode)
            env.reset(seed=int(cfg["seed"]) + 7777)
            steps_per_run = max(1, n_steps // max(len(runs), 1))
            for s in range(steps_per_run):
                per_agent_obs = env.per_agent_observations()
                actions = agent.select_actions(per_agent_obs, explore=False)
                _, _, term, trunc, info = env.step(actions)
                # Item 6: read applied-block phases and |h_eff| from info (same
                # coherence block as the action), NOT the post-transition
                # private env state which reflects the already-evolved channel.
                phases.append(np.concatenate([info["phi_r_applied"],
                                              info["phi_t_applied"]]))
                if env.K_t > 0:
                    heff_T.append(info["h_eff_applied_abs"][env.K_r:])
                if term or trunc:
                    env.reset(seed=int(cfg["seed"]) + 7777 + s)
        out_phase[label] = np.concatenate(phases) if phases else np.array([])
        out_heff[label] = np.concatenate(heff_T) if heff_T else np.array([])
    return out_phase, out_heff


def _rebuild_agent_for_eval(cfg, algo_key: str, seed: int, ckpt_dir: str | None = None,
                            run_name: str | None = None, device: str = "cpu",
                            best_path: str | None = None):
    """Reconstruct an agent and load its best.pt inference checkpoint."""
    from algorithms import MADDPG, DDPGAgent, TD3Agent, PPOAgent
    from utils import ObservationNormalizer
    best = (os.path.abspath(best_path) if best_path else
            os.path.join(str(ckpt_dir), str(run_name), "best.pt"))
    if not os.path.exists(best):
        return None
    env = _make_env(cfg, seed)
    if algo_key == "maddpg":
        agent = MADDPG(env.spec(), hidden_sizes=cfg["networks"]["hidden_sizes"],
                       maddpg_cfg=cfg["maddpg"], net_cfg=cfg["networks"],
                       device=device, seed=seed)
        agent.attach_obs_normalizers(
            [ObservationNormalizer(shape=(d,)) for d in env.spec().obs_dims])
    else:
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        hidden = cfg["networks"]["hidden_sizes"]
        if algo_key == "td3_matched":
            target = maddpg_param_counts(env.spec(), hidden, cfg["networks"])["total_params"]
            hidden = matched_td3_hidden_sizes(obs_dim, act_dim, target, cfg["networks"])
        if algo_key == "ddpg":
            agent = DDPGAgent(obs_dim, act_dim, hidden, cfg["ddpg"], cfg["networks"],
                              device=device, seed=seed)
        elif algo_key in ("td3", "td3_matched"):
            agent = TD3Agent(obs_dim, act_dim, hidden, cfg["td3"], cfg["networks"],
                             device=device, seed=seed)
        elif algo_key == "ppo":
            agent = PPOAgent(obs_dim, act_dim, hidden, cfg["ppo"], cfg["networks"],
                             device=device, seed=seed)
        else:
            return None
        from utils import ObservationNormalizer as _ON
        agent.attach_obs_normalizer(_ON(shape=(obs_dim,)))
    agent.load_inference(best)
    # Recover the per-user lambda vector selected at the best episode from the
    # checkpoint metadata (item 9): evaluation must reuse it, not a scalar mean.
    payload = torch.load(best, map_location="cpu", weights_only=False)
    lam_vec = payload.get("meta", {}).get("lambda_vec")
    lam_vec = None if lam_vec is None else np.asarray(lam_vec, dtype=np.float64)
    return {"agent": agent, "seed": seed, "best_ckpt": best,
            "trained_qos_lambda_vec": lam_vec,
            "trained_qos_lambda": None if lam_vec is None else float(np.mean(lam_vec)),
            "history": {k: [] for k in ("ma_return",)}}


def main(argv=None):
    args = parse_args(argv)

    if args.load_run_dir and (args.train_only or args.aggregate_only):
        raise SystemExit("--load-run-dir cannot be combined with --train-only/--aggregate-only")
    if args.input_run_dirs and not args.aggregate_only:
        raise SystemExit("--input-run-dirs/--load-shards requires --aggregate-only")
    if args.seeds is not None and not args.train_only:
        raise SystemExit("--seeds is reserved for isolated --train-only jobs; normal and "
                         "aggregate runs derive their seed matrix from config/manifests.")
    if (args.final_paper or args.allow_partial) and not args.aggregate_only:
        raise SystemExit("--final-paper/--allow-partial apply to --aggregate-only")

    # Deterministic source provenance (V4 review P0-1): stamped into every
    # manifest, checkpoint, run_meta and raw-result row.
    source_sha = source_tree_sha256()
    evaluation_source_sha = source_sha
    training_source_sha = source_sha

    # ------------------------------------------------ config / shard discovery
    # Aggregate-only never reads run_meta.json.  It discovers immutable pair
    # manifests and verifies their effective configs and best checkpoints.
    load_meta = None
    aggregate_records: list[dict] = []
    verified_shards: list[dict] = []
    aggregate_training_seeds: list[int] | None = None
    if args.aggregate_only:
        if args.quick or args.episodes is not None or args.resume or args.skip_train:
            raise SystemExit("--aggregate-only does not accept training/quick/resume overrides")
        roots = args.input_run_dirs or [os.path.join(args.out, "results_revised", "shards")]
        aggregate_records = scan_shard_manifests(roots)
        verified_shards = [r for r in aggregate_records if r.get("_verified")]
        if not verified_shards:
            raise RuntimeError("No verified completed shard manifests were found")
        cfg = load_config(verified_shards[0]["_effective_config"])
        args.algos, aggregate_training_seeds = validate_shard_group(
            verified_shards, cfg, final_paper=args.final_paper,
            allow_partial=args.allow_partial, current_source_sha=source_sha)
        training_source_sha = str(verified_shards[0]["source_sha"])
        args.skip_train = True
    elif args.load_run_dir:
        # Legacy single-directory re-evaluation remains available, but the new
        # distributed aggregator intentionally does not use this metadata.
        with open(os.path.join(args.load_run_dir, "run_meta.json"), encoding="utf-8") as f:
            load_meta = json.load(f)
        training_source_sha = str(load_meta.get(
            "training_source_sha", load_meta.get("source_sha", source_sha)))
        cfg = load_config(os.path.join(args.load_run_dir, "effective_config.yaml"))
        args.skip_train = True
    else:
        cfg = load_config(args.config)
    validate_formulation_config(cfg)

    if args.quick:
        cfg["training"]["total_episodes"] = 10
        cfg["training"]["eval_every"] = 5
        cfg["training"]["eval_episodes"] = 2
        cfg["evaluation"]["num_episodes"] = 2
        # Quick/smoke uses development seeds only -- never validation/test (item 4).
        dev = cfg["evaluation"].get("dev_seeds", [101, 202])
        cfg["evaluation"]["validation_seeds"] = dev[:2]
        cfg["evaluation"]["test_seeds"] = dev[:2]
        cfg["training"]["training_seeds"] = cfg["training"]["training_seeds"][:2]
    if args.episodes is not None:
        cfg["training"]["total_episodes"] = int(args.episodes)

    config_bytes, config_sha = _config_bytes_and_sha(cfg)
    if args.aggregate_only and config_sha != verified_shards[0]["config_sha"]:
        raise RuntimeError("Resolved effective config does not match the shard config SHA")
    if args.train_only:
        if _select_device(cfg.get("device", "auto")) == "cpu":
            configure_cpu_threads(1, 1)
        _run_train_only(args, cfg, config_bytes, config_sha, source_sha)
        return
    set_checkpoint_provenance({"source_sha": source_sha, "config_sha": config_sha})

    is_paper_run = bool(cfg.get("evaluation", {}).get("seed_split"))
    if is_paper_run and "td3_matched" not in args.algos:
        warnings.warn(
            "Paper run omits td3_matched, the primary comparison required for the "
            "multi-agent-decomposition claim.", RuntimeWarning)

    # On CPU, pin thread counts so small MLP updates are not oversubscribed
    # (item 10). No effect on CUDA runs.
    if _select_device(cfg.get("device", "auto")) == "cpu":
        configure_cpu_threads(1, 1)

    # ------------------------------------------------ run-specific output dir
    # Results go into a directory keyed by config hash + timestamp (item 14) so
    # generated artifacts never overwrite a previous run or leak into the source
    # tree / release ZIP.
    if args.aggregate_only:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = args.run_id or f"aggregate_{config_sha[:8]}_{stamp}"
        results_root = os.path.join(args.out, "results_revised")
        out_root = os.path.join(results_root, run_id)
        log_dir = ""
        ckpt_dir = ""
    elif load_meta is not None:
        # Reuse the trained run's checkpoints/logs; write eval outputs into a
        # new sibling directory so the original run is not overwritten.
        run_id = load_meta["run_id"]
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_root = os.path.join(args.out, "results_revised")
        out_root = os.path.join(results_root, f"{run_id}_eval_{stamp}")
        log_dir = load_meta["log_dir"]
        ckpt_dir = load_meta["ckpt_dir"]
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = args.run_id or f"run_{config_sha[:8]}_{stamp}"
        results_root = os.path.join(args.out, "results_revised")
        out_root = os.path.join(results_root, run_id)
        log_dir = os.path.join(args.out, cfg["training"]["log_dir"], run_id)
        ckpt_dir = os.path.join(args.out, cfg["training"]["ckpt_dir"], run_id)
    fig_dir = os.path.join(out_root, "figures")
    tab_dir = os.path.join(out_root, "tables")
    for d in (out_root, fig_dir, tab_dir):
        os.makedirs(d, exist_ok=True)

    # ------------------------------------------------ effective config + run meta
    eff_cfg_path = os.path.join(out_root, "effective_config.yaml")
    with open(eff_cfg_path, "wb") as f:
        f.write(config_bytes)
    meta_payload = {"run_id": run_id, "config_sha": config_sha,
                    # `source_sha` is retained as a backward-compatible alias
                    # for the evaluation implementation.  The two explicit
                    # fields prevent train-with-A/evaluate-with-B ambiguity.
                    "source_sha": evaluation_source_sha,
                    "training_source_sha": training_source_sha,
                    "evaluation_source_sha": evaluation_source_sha,
                    "environment": runtime_environment_metadata(source_sha),
                    "log_dir": log_dir, "ckpt_dir": ckpt_dir,
                    "training_seeds": (aggregate_training_seeds or
                                       list(cfg["training"]["training_seeds"])),
                    "algos": args.algos}
    if args.aggregate_only:
        meta_payload["source"] = "verified_shard_manifests"
        meta_payload["source_manifests"] = [r["_manifest_path"] for r in verified_shards]
    _atomic_write_json(os.path.join(out_root, "run_meta.json"), meta_payload)
    if args.aggregate_only:
        completed_runs_frame(aggregate_records).to_csv(
            os.path.join(out_root, "completed_runs.csv"), index=False)
    with open(os.path.join(results_root, "LATEST_RUN.txt"), "w", encoding="utf-8") as f:
        f.write(os.path.basename(out_root) + "\n")
    print(f"Run id: {run_id}  (config sha256={config_sha[:12]})")
    print(f"Output dir: {out_root}")
    if load_meta is not None:
        print(f"Loading trained checkpoints from: {ckpt_dir}")
    if args.aggregate_only:
        print(f"Verified {len(verified_shards)} completed shard manifests; run_meta.json was not read.")

    # ------------------------------------------------ scenario banks
    print("\n========== Building ScenarioBanks (validation + locked test) ==========")
    banks = build_eval_banks(cfg)
    for split, bank in banks.items():
        bank.save(os.path.join(out_root, f"scenario_bank_{split}.json"),
                  os.path.join(out_root, f"scenario_bank_{split}.npz"))
        print(f"  {split}: {len(bank)} scenarios, sha256={bank.content_hash()[:12]}")
    test_scenarios = banks["test"].scenarios

    # ------------------------------------------------ multi-seed training
    training_seeds = (aggregate_training_seeds or
                      list(cfg["training"].get("training_seeds", [int(cfg["seed"])])))
    trained: dict[str, list[dict]] = {}
    if not args.skip_train:
        for algo_key in args.algos:
            label = LEARNED_ALGOS.get(algo_key)
            if label is None:
                continue
            trained[label] = []
            for s in training_seeds:
                print(f"\n========== Training {label} (seed={s}) ==========")
                resume = args.resume if algo_key == "maddpg" and s == training_seeds[0] else None
                info = _train_algorithm(cfg, algo_key, s, log_dir, ckpt_dir,
                                        resume_from=resume)
                info["seed"] = s
                info["log_csv"] = os.path.join(log_dir, f"{algo_key}_seed{s}", "log.csv")
                _save_history_csv(os.path.join(log_dir, f"{algo_key}_seed{s}", "history.csv"),
                                  info["history"])
                # Archive the (large) resumable replay sidecar after a
                # successful run; the best inference checkpoint is kept (item 7).
                if args.drop_replay_after_train:
                    for sidecar in (info.get("latest_ckpt", "") + ".replay.npz",):
                        if sidecar and os.path.exists(sidecar):
                            os.remove(sidecar)
                trained[label].append(info)
    elif args.aggregate_only:
        for algo_key in args.algos:
            label = LEARNED_ALGOS.get(algo_key)
            if label is None:
                continue
            runs = []
            records = sorted((r for r in verified_shards if r["algorithm"] == algo_key),
                             key=lambda r: int(r["training_seed"]))
            for record in records:
                s = int(record["training_seed"])
                info = _rebuild_agent_for_eval(cfg, algo_key, s,
                                               best_path=record["best_checkpoint"])
                if info is None:
                    raise RuntimeError(f"Verified checkpoint could not be loaded: {record}")
                info["shard_manifest"] = record["_manifest_path"]
                # Rebuild convergence inputs from the shard's hash-verified
                # history (V4 review P0-2): aggregate-only must regenerate the
                # training figures, not silently omit them.
                info["history"], info["log_csv"] = _load_shard_history(record)
                runs.append(info)
            if runs:
                trained[label] = runs
    else:
        for algo_key in args.algos:
            label = LEARNED_ALGOS.get(algo_key)
            if label is None:
                continue
            runs = []
            for s in training_seeds:
                info = _rebuild_agent_for_eval(cfg, algo_key, s, ckpt_dir,
                                               f"{algo_key}_seed{s}")
                if info is not None:
                    runs.append(info)
            if runs:
                trained[label] = runs
    if not trained:
        print("No trained agents."); return

    if not args.aggregate_only:
        completed = []
        label_to_key = {v: k for k, v in LEARNED_ALGOS.items()}
        for label, runs in trained.items():
            for run in runs:
                best = os.path.abspath(run.get("best_ckpt", ""))
                completed.append({
                    "algorithm": label_to_key[label],
                    "training_seed": int(run["seed"]),
                    "status": "completed",
                    "config_sha": config_sha,
                    "source_sha": source_sha,
                    "best_checkpoint": best,
                    "checkpoint_sha": sha256_file(best) if os.path.isfile(best) else "",
                })
        completed_runs_frame(completed).to_csv(
            os.path.join(out_root, "completed_runs.csv"), index=False)

    # ------------------------------------------------ training curves (true CI)
    print("\n========== Plotting training convergence (CI across seeds) ==========")
    have_histories = all(len(r["history"].get("ma_return", [])) > 0
                         for runs in trained.values() for r in runs)
    if args.aggregate_only and not have_histories:
        raise RuntimeError(
            "Aggregate-only loaded shards without training histories; the "
            "convergence figures would be silently omitted (V4 review P0-2).")
    if have_histories:
        for metric, fname, ylab in [
                ("ma_return", "training_convergence", "Episode return (MA)"),
                ("sum_rate", "training_sum_rate", "Avg. sum-rate (b/s/Hz)"),
                ("user_qos_fraction", "training_user_qos_fraction",
                 "User QoS fraction (1/K) sum 1[R_k >= R_min]"),
                ("all_users_qos_satisfied", "training_all_users_qos",
                 "P(all K users satisfy QoS)"),
                ("common_power_frac", "training_common_power_frac", "P_c / P_max")]:
            plot_training_convergence(
                {algo: _seed_matrix(runs, metric) for algo, runs in trained.items()},
                out_dir=fig_dir, name=fname, ylabel=ylab)
        if "MADDPG" in trained:
            plot_qos_lambda(
                {"qos_lambda": _seed_matrix(trained["MADDPG"], "qos_lambda_mean").mean(axis=0)},
                out_dir=fig_dir, name="qos_lambda")
            # Reward decomposition from the first seed's episode log. The path
            # comes from the run info (set by both the local-train and the
            # hash-verified shard-aggregate loaders), so aggregate-only also
            # regenerates this figure (V4 review P0-2).
            log_csv = trained["MADDPG"][0].get("log_csv", "")
            if log_csv and os.path.exists(log_csv):
                df_log = pd.read_csv(log_csv)
                plot_reward_decomposition(
                    {k: df_log[k].values for k in
                     ("reward_sr_mean", "reward_dual_mean",
                      "reward_aug_mean", "reward_switch_mean")
                     if k in df_log.columns},
                    out_dir=fig_dir, name="reward_decomposition")

    # ------------------------------------------------ locked-test evaluation
    print("\n========== Test-bank evaluation (unit = training seed) ==========")
    raw_rows: list[dict] = []
    common_extra = {"Pmax_dbm": cfg["env"]["p_max_dbm"], "N": cfg["env"]["num_ris_elements"],
                    "K": cfg["env"]["num_users"]}
    # Cache each run's best.pt sha so ablation/sweep rows are consistent.
    run_ckpt_sha = {id(run): (sha256_file(run["best_ckpt"])
                              if run.get("best_ckpt") and os.path.exists(run["best_ckpt"]) else "")
                    for runs in trained.values() for run in runs}
    rows = []
    pareto_points = {}
    seed_level: dict[str, dict[str, list[float]]] = {}
    for algo, runs in trained.items():
        per_seed = {"sum_rate": [], "user_qos_fraction": [], "all_users_qos": [],
                    "return": [], "rate_common": [], "h_eff_T": [], "pent": [],
                    "cfrac": [], "min_rate": [], "cbar_max": [], "cbar_meanviol": []}
        for run in runs:
            ck_sha = run_ckpt_sha[id(run)]
            # Evaluate the SELECTED best.pt agent (run["agent"] is now the
            # rebuilt best checkpoint) with the per-user lambda from that
            # checkpoint -- results are therefore consistent with ck_sha.
            m = eval_on_scenarios(run["agent"], algo, cfg, test_scenarios,
                                  qos_lambda_vec=run.get("trained_qos_lambda_vec"))
            per_seed["sum_rate"].append(m["sum_rate_mean"])
            per_seed["user_qos_fraction"].append(m["user_qos_fraction_mean"])
            per_seed["all_users_qos"].append(m["all_users_qos_prob"])
            per_seed["return"].append(m["return_mean"])
            per_seed["rate_common"].append(m["rate_common_mean"])
            per_seed["h_eff_T"].append(m["h_eff_abs_T_mean"])
            per_seed["pent"].append(m["phase_entropy_T_mean"])
            per_seed["cfrac"].append(m["common_power_frac_mean"])
            per_seed["min_rate"].append(m["min_user_rate_mean"])
            # Expected constraint violation c_bar_k = R_min - mean(R_k) (item 9):
            # a physical, cross-algorithm-comparable measure of QoS shortfall.
            c_bar = np.asarray(m["c_bar_per_user"], dtype=np.float64)
            per_seed["cbar_max"].append(float(c_bar.max()))
            per_seed["cbar_meanviol"].append(float(np.maximum(c_bar, 0.0).mean()))
            raw_rows.extend(scenario_rows(algo, m, test_scenarios,
                                          training_seed=run.get("seed"),
                                          config_sha=config_sha, checkpoint_sha=ck_sha,
                                          extra={**common_extra, "scenario": "test_bank"}))
        seed_level[algo] = per_seed
        sr_m, sr_ci, _ = confidence_interval(np.array(per_seed["sum_rate"]))
        uq_m, uq_ci, _ = confidence_interval(np.array(per_seed["user_qos_fraction"]))
        aq_m, aq_ci, _ = confidence_interval(np.array(per_seed["all_users_qos"]))
        mv_m, mv_ci, _ = confidence_interval(np.array(per_seed["cbar_max"]))
        # Primary metrics are PHYSICAL (sum-rate, QoS fractions) and the expected
        # constraint violation. Return is intentionally NOT reported as a primary
        # cross-algorithm metric because the selected lambda vectors differ
        # between algorithms, making Return non-comparable (item 9).
        rows.append({"Algorithm": algo,
                     "SumRate": sr_m, "SumRate_CI": sr_ci,
                     "UserQoSFraction": uq_m, "UserQoSFraction_CI": uq_ci,
                     "AllUsersQoSProb": aq_m, "AllUsersQoSProb_CI": aq_ci,
                     "MaxExpViolation": mv_m, "MaxExpViolation_CI": mv_ci,
                     "MeanExpViolation": float(np.mean(per_seed["cbar_meanviol"])),
                     "MinUserRate": float(np.mean(per_seed["min_rate"])),
                     "RateCommon": float(np.mean(per_seed["rate_common"])),
                     "P_c/Pmax": float(np.mean(per_seed["cfrac"])),
                     "N_train_seeds": len(runs)})
        pareto_points[algo] = {"sum_rate_mean": sr_m, "sum_rate_ci": sr_ci,
                               "qos_mean": uq_m, "qos_ci": uq_ci}
    df_cmp = pd.DataFrame(rows)
    df_cmp.to_csv(os.path.join(tab_dir, "algorithm_comparison.csv"), index=False)
    _write_tex_table(
        os.path.join(tab_dir, "algorithm_comparison.tex"), df_cmp,
        caption=("Deterministic-policy evaluation on the locked test ScenarioBank "
                 f"({len(training_seeds)} training seeds; unit of analysis = training-seed mean; "
                 "Student-t 95\\% CI across training seeds). Primary metrics are physical: "
                 "'UserQoSFraction' = $(1/K)\\sum_k \\mathbf{1}[R_k \\ge R_{\\min}]$; "
                 "'AllUsersQoSProb' = $\\Pr[\\min_k R_k \\ge R_{\\min}]$; 'MaxExpViolation' = "
                 "$\\max_k (R_{\\min} - \\mathbb{E}[R_k])$. Episode Return is omitted as a "
                 "cross-algorithm metric because selected $\\lambda$ vectors differ."),
        label="tab:algorithm_comparison")
    print(df_cmp.to_string(index=False))

    # ------------------------------------------------ paired statistics
    print("\n========== Paired statistics (Holm-corrected primary family) ==========")
    if "MADDPG" in seed_level:
        m_sr = np.array(seed_level["MADDPG"]["sum_rate"])
        fam_rows, fam_ps = [], []
        for other in PRIMARY_FAMILY:
            if other not in seed_level:
                continue
            o_sr = np.array(seed_level[other]["sum_rate"])
            if o_sr.size != m_sr.size:
                continue
            p_t = paired_t_test_p(m_sr, o_sr)
            p_perm = paired_permutation_p(m_sr, o_sr, seed=123)
            d_eff = cohens_d_paired(m_sr, o_sr)
            diff, diff_ci = paired_difference_ci(m_sr, o_sr)
            fam_rows.append({"Comparison": f"MADDPG vs {other}", "Metric": "sum_rate",
                             "diff_mean": diff, "diff_CI95": diff_ci,
                             "p_paired_t": p_t, "p_permutation": p_perm,
                             "cohens_d_paired": d_eff,
                             "n_seeds": int(m_sr.size)})
            fam_ps.append(p_t)
        if fam_rows:
            adj = holm_bonferroni(fam_ps)
            for r, ap in zip(fam_rows, adj):
                r["p_holm"] = ap
                r["significant_5pct_holm"] = ap < 0.05
        df_sig = pd.DataFrame(fam_rows)
        df_sig.to_csv(os.path.join(tab_dir, "significance.csv"), index=False)
        _write_tex_table(
            os.path.join(tab_dir, "significance.tex"), df_sig,
            caption=("Paired tests on training-seed-level sum-rate means (same seed list, "
                     "identical test scenarios). Holm-Bonferroni corrected within the "
                     "pre-specified primary family \\{MADDPG vs TD3, TD3-Matched, DDPG, PPO\\}. "
                     "TD3-Matched (total-parameter-matched single agent) is a primary "
                     "comparison: the multi-agent-decomposition claim requires MADDPG to beat "
                     "it, not merely the smaller single agents."),
            label="tab:significance")
        print(df_sig.to_string(index=False))

    print("\n========== Pareto plot (SR vs user QoS fraction) ==========")
    plot_pareto(pareto_points, out_dir=fig_dir, name="pareto_sr_vs_qos",
                ylabel="User QoS fraction")

    # ------------------------------------------------ power sweep (all seeds)
    print("\n========== Sweep: sum-rate vs Pmax (all training seeds) ==========")
    p_list = cfg["evaluation"]["power_sweep_dbm"]
    sweep = {algo: {"mean": [], "ci": [], "qos_mean": [], "qos_ci": []}
             for algo in trained}
    for p_dbm in p_list:
        cfg_p = json.loads(json.dumps(cfg))
        cfg_p["env"]["p_max_dbm"] = float(p_dbm)
        for algo, runs in trained.items():
            srs, uqs = [], []
            for run in runs:
                m = eval_on_scenarios(run["agent"], algo, cfg_p, test_scenarios,
                                      qos_lambda_vec=run.get("trained_qos_lambda_vec"))
                srs.append(m["sum_rate_mean"])
                uqs.append(m["user_qos_fraction_mean"])
                raw_rows.extend(scenario_rows(algo, m, test_scenarios,
                                              training_seed=run.get("seed"),
                                              config_sha=config_sha,
                                              checkpoint_sha=run_ckpt_sha[id(run)],
                                              extra={**common_extra,
                                                     "Pmax_dbm": float(p_dbm),
                                                     "scenario": "power_sweep"}))
            sr_m, sr_ci, _ = confidence_interval(np.array(srs))
            uq_m, uq_ci, _ = confidence_interval(np.array(uqs))
            sweep[algo]["mean"].append(sr_m); sweep[algo]["ci"].append(sr_ci)
            sweep[algo]["qos_mean"].append(uq_m); sweep[algo]["qos_ci"].append(uq_ci)
    plot_metric_vs_x(p_list, sweep, xlabel="$P_{\\max}$ (dBm)",
                     ylabel="Avg. sum-rate (b/s/Hz)",
                     out_dir=fig_dir, name="sumrate_vs_power")
    plot_metric_vs_x(p_list,
                     {a: {"mean": sweep[a]["qos_mean"], "ci": sweep[a]["qos_ci"]}
                      for a in sweep},
                     xlabel="$P_{\\max}$ (dBm)", ylabel="User QoS fraction",
                     out_dir=fig_dir, name="qos_vs_power")
    pd.DataFrame({"Pmax_dBm": p_list,
                  **{f"{a}_sr_mean": sweep[a]["mean"] for a in sweep},
                  **{f"{a}_sr_ci": sweep[a]["ci"] for a in sweep},
                  **{f"{a}_uqf_mean": sweep[a]["qos_mean"] for a in sweep}}
                 ).to_csv(os.path.join(tab_dir, "sumrate_vs_power.csv"), index=False)
    _write_tex_table(os.path.join(tab_dir, "sumrate_vs_power.tex"),
                     pd.DataFrame({"Pmax_dBm": p_list,
                                   **{a: sweep[a]["mean"] for a in sweep}}),
                     caption=(f"Average sum-rate (b/s/Hz) vs $P_{{\\max}}$ — mean across "
                              f"{len(training_seeds)} training seeds on the locked test bank."),
                     label="tab:sumrate_vs_power")

    # ------------------------------------------------ ablation (all seeds)
    abl = None
    if "MADDPG" in trained:
        print("\n========== Ablation (agent cells: all training seeds) ==========")
        abl = ablation_study(trained["MADDPG"], cfg, test_scenarios, raw_rows=raw_rows,
                             config_sha=config_sha,
                             run_checkpoint_shas=[run_ckpt_sha[id(r)]
                                                  for r in trained["MADDPG"]])
        labels = list(abl.keys())
        plot_bar(labels, {k: abl[k]["sum_rate_mean"] for k in labels},
                 out_dir=fig_dir, name="ablation",
                 ylabel="Avg. sum-rate (b/s/Hz)",
                 ci={k: abl[k]["sum_rate_ci"] for k in labels})
        plot_bar(labels, {k: abl[k]["user_qos_fraction_mean"] for k in labels},
                 out_dir=fig_dir, name="ablation_qos",
                 ylabel="User QoS fraction",
                 ci={k: abl[k]["user_qos_fraction_ci"] for k in labels})
        df_abl = pd.DataFrame({
            "Cell": labels,
            "SumRate_mean": [abl[k]["sum_rate_mean"] for k in labels],
            "SumRate_CI95": [abl[k]["sum_rate_ci"] for k in labels],
            "UserQoSFraction": [abl[k]["user_qos_fraction_mean"] for k in labels],
            "AllUsersQoSProb": [abl[k]["all_users_qos_prob"] for k in labels],
            "CI_unit": [abl[k]["ci_unit"] for k in labels],
            "N_units": [abl[k]["n_units"] for k in labels],
            "RateCommon": [abl[k]["rate_common"] for k in labels],
            "|h_eff_T|": [abl[k]["h_eff_abs_T"] for k in labels],
            "P_c/Pmax": [abl[k]["common_power_frac"] for k in labels],
        })
        df_abl.to_csv(os.path.join(tab_dir, "ablation.csv"), index=False)
        _write_tex_table(os.path.join(tab_dir, "ablation.tex"), df_abl,
                         caption=("Ablation across RIS modes and BS power policies. "
                                  "'AO-Grid' is a coarse alternating-optimization grid "
                                  "heuristic (NOT an upper bound). CI units: training seeds "
                                  "for agent-dependent cells, independent scenarios for "
                                  "policy-independent cells."),
                         label="tab:ablation")

        print("\n========== Phase histogram + |h_eff| dist (all seeds) ==========")
        diag_steps = int(cfg["evaluation"].get("diagnostic_steps", 240))
        phase_samples, heff_samples = _collect_phase_and_heff_samples(
            trained["MADDPG"], cfg, n_steps=diag_steps)
        plot_phase_histogram(phase_samples, out_dir=fig_dir, name="phase_histogram")
        plot_h_eff_distribution(heff_samples, out_dir=fig_dir, name="h_eff_distribution")

    # ------------------------------------------------ AO local-search reference
    ao_max_n = int(cfg["evaluation"].get("ao_local_search_max_n", 32))
    if ao_max_n > 0 and cfg["env"]["num_ris_elements"] <= ao_max_n:
        print("\n========== Hybrid AO Local Search reference (N small) ==========")
        try:
            from experiments.baselines_ao import (
                AOHybridLocalSearch, ao_reference_lambda, stratified_ao_scenarios,
                solver_params_from_config,
            )
            reference_lambda = ao_reference_lambda(cfg)
            per_seed = int(cfg["evaluation"].get("ao_scenarios_per_seed", 1))
            ao_scenarios = stratified_ao_scenarios(test_scenarios, per_seed=per_seed)
            # Solver hyperparameters are pre-registered in the config (frozen
            # on validation evidence, V4 review item 7) and folded into the
            # solver_config_sha so any post-hoc change is visible.
            ao_solver_params = solver_params_from_config(cfg)
            solver_cfg_sha = hashlib.sha256(
                json.dumps({"solver": "AOHybridLocalSearch",
                            "env": cfg["env"],
                            "ao_reference_lambda": reference_lambda.tolist(),
                            "ao_solver_params": ao_solver_params,
                            "scenarios_per_evaluation_seed": per_seed},
                           sort_keys=True).encode()).hexdigest()
            n_blocks = 3      # sequential AO decisions per scenario (correlated blocks)
            # Statistical unit (item 6): average the correlated AO blocks WITHIN
            # each scenario first, then compute the CI across scenario means.
            per_scenario_sr, per_scenario_uqf = [], []
            scenario_ao_rows = []
            for sc in ao_scenarios:
                env = _make_env(cfg, seed=int(cfg["seed"]))
                env.set_qos_lambda_vec(reference_lambda)
                env.reset(options={"scenario": sc})
                solver = AOHybridLocalSearch(
                    env, seed=int(cfg["seed"]) + int(sc["evaluation_seed"]),
                    **ao_solver_params)
                prev = None
                block_records = []
                for _b in range(n_blocks):
                    sol = solver.solve(prev_applied=prev)
                    rates = np.asarray(sol["per_user_rate"], dtype=np.float64)
                    violation = np.maximum(env.qos_min - rates, 0.0)
                    block_records.append({
                        "sum_rate": float(sol["sum_rate"]),
                        "user_qos_fraction": float(np.mean(rates >= env.qos_min)),
                        "all_users_qos": float(np.all(rates >= env.qos_min)),
                        "min_user_rate": float(np.min(rates)),
                        "mean_qos_deficit": float(np.mean(violation)),
                        "max_constraint_violation": float(np.max(violation)),
                        "objective": float(sol["objective"]),
                        "converged": bool(sol["converged"]),
                        "n_evals": int(sol["n_evals"]),
                        # Wall-clock per solve (V4 review item 7).
                        "solve_time_ms": float(sol["solve_time_ms"]),
                    })
                    prev = {"phi_r": sol["phi_r"], "phi_t": sol["phi_t"],
                            "beta_r": sol["beta_r"], "power_weights": sol["power_weights"]}
                    # Public API advances step_count so innovation index goes
                    # 0, 1, 2 (item 5) -- never reuse innovation 0.
                    env.advance_to_next_block()
                scalar_metrics = ("sum_rate", "user_qos_fraction", "all_users_qos",
                                  "min_user_rate", "mean_qos_deficit",
                                  "max_constraint_violation", "objective",
                                  "solve_time_ms")
                scenario_row = {
                    "evaluation_seed": int(sc["evaluation_seed"]),
                    "episode_idx": int(sc["episode_idx"]),
                    "scenario_id": sc["scenario_id"],
                    "config_sha": config_sha,
                    "solver_config_sha": solver_cfg_sha,
                    **{key: float(np.mean([b[key] for b in block_records]))
                       for key in scalar_metrics},
                    "converged": bool(all(b["converged"] for b in block_records)),
                    "n_evals": int(sum(b["n_evals"] for b in block_records)),
                    "wall_time_ms": float(sum(b["solve_time_ms"] for b in block_records)),
                    "blocks": n_blocks,
                }
                scenario_ao_rows.append(scenario_row)
                per_scenario_sr.append(scenario_row["sum_rate"])
                per_scenario_uqf.append(scenario_row["user_qos_fraction"])
                for b, block in enumerate(block_records):
                    for metric in (*scalar_metrics, "converged", "n_evals"):
                        raw_rows.append({
                            "algorithm": "AO-LocalSearch", "training_seed": "",
                            "evaluation_seed": sc.get("evaluation_seed", ""),
                            "episode_idx": sc.get("episode_idx", ""),
                            "scenario_id": sc["scenario_id"], "config_sha": config_sha,
                            "checkpoint_sha": "", "solver_config_sha": solver_cfg_sha,
                            "scenario": "ao_local_search", "block_idx": b,
                            "metric": metric, "value": block[metric]})
            pd.DataFrame(scenario_ao_rows).to_csv(
                os.path.join(tab_dir, "ao_local_search_scenarios.csv"), index=False)
            ao_m, ao_ci, _ = confidence_interval(np.array(per_scenario_sr))
            uq_m, uq_ci, _ = confidence_interval(np.array(per_scenario_uqf))
            pd.DataFrame([{"Method": "Hybrid AO Local Search (SLSQP + projected gradient)",
                           "SumRate_mean": ao_m, "SumRate_CI95": ao_ci,
                           "UserQoSFraction": uq_m, "UserQoSFraction_CI95": uq_ci,
                           "n_scenarios": len(per_scenario_sr),
                           "n_evaluation_seeds": len({r["evaluation_seed"]
                                                      for r in scenario_ao_rows}),
                           "scenarios_per_evaluation_seed": per_seed,
                           "blocks_per_scenario": n_blocks,
                           "ConvergenceRate": float(np.mean(
                               [r["converged"] for r in scenario_ao_rows])),
                           "TotalNEvals": int(sum(r["n_evals"] for r in scenario_ao_rows)),
                           "MeanSolveTimeMs": float(np.mean(
                               [r["solve_time_ms"] for r in scenario_ao_rows])),
                           "TotalWallTimeMs": float(sum(
                               r["wall_time_ms"] for r in scenario_ao_rows)),
                           "ci_unit": "scenario_mean",
                           "note": "local-search reference, NOT an upper bound"}]
                         ).to_csv(os.path.join(tab_dir, "ao_local_search.csv"), index=False)
            print(f"  AO local search: SR={ao_m:.3f} +/- {ao_ci:.3f} "
                  f"(CI over {len(per_scenario_sr)} scenario means)")
        except ImportError as e:
            print(f"  Skipped (scipy missing): {e}")

    # ------------------------------------------------ latency (separate tables)
    print("\n========== Latency benchmarks (CPU and GPU tables SEPARATE) ==========")
    bench_agents = {algo: runs[0]["agent"] for algo, runs in trained.items()}
    # Benchmark size is configurable so integration tests stay light (V4
    # review item 6); the paper run keeps the 2000-call default.
    n_calls = 200 if args.quick else int(cfg["evaluation"].get("latency_num_calls", 2000))
    n_warmup = int(cfg["evaluation"].get("latency_warmup_calls", 50))
    lat_cpu = benchmark_latency_cpu(bench_agents, cfg, num_calls=n_calls,
                                    warmup=n_warmup)
    meta_cpu = lat_cpu.pop("_meta")
    df_lat_cpu = pd.DataFrame([{"Method": k, **v} for k, v in lat_cpu.items()])
    df_lat_cpu.to_csv(os.path.join(tab_dir, "latency_cpu.csv"), index=False)
    _write_tex_table(os.path.join(tab_dir, "latency_cpu.tex"), df_lat_cpu,
                     caption=("Inference latency on CPU (single thread, "
                              "torch.inference\\_mode, warm-up, batch 1; timed region = "
                              "observation preprocessing + forward + post-processing). "
                              f"Hardware: {meta_cpu['cpu']}."),
                     label="tab:latency_cpu")
    with open(os.path.join(tab_dir, "latency_cpu_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_cpu, f, indent=2)
    plot_bar(list(lat_cpu.keys()), {k: v["median_ms"] for k, v in lat_cpu.items()},
             out_dir=fig_dir, name="latency", ylabel="Median CPU latency (ms/action)")

    lat_gpu = benchmark_latency_gpu(bench_agents, cfg, num_calls=n_calls,
                                    warmup=n_warmup)
    if lat_gpu is not None:
        meta_gpu = lat_gpu.pop("_meta")
        df_lat_gpu = pd.DataFrame([{"Method": k, **v} for k, v in lat_gpu.items()])
        df_lat_gpu.to_csv(os.path.join(tab_dir, "latency_gpu.csv"), index=False)
        _write_tex_table(os.path.join(tab_dir, "latency_gpu.tex"), df_lat_gpu,
                         caption=(f"Inference latency on GPU ({meta_gpu['device']}, "
                                  "torch.cuda.synchronize around the timed region). "
                                  "Reported separately from the CPU table; the two are "
                                  "not directly comparable."),
                         label="tab:latency_gpu")
        with open(os.path.join(tab_dir, "latency_gpu_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_gpu, f, indent=2)

    # ------------------------------------------------ model complexity
    print("\n========== Model complexity (params / FLOPs) ==========")
    env0 = _make_env(cfg, int(cfg["seed"]))
    comp_rows = []
    hidden = cfg["networks"]["hidden_sizes"]
    obs_dim = env0.observation_space.shape[0]
    act_dim = env0.action_space.shape[0]
    comp_rows.append({"Algorithm": "MADDPG", "hidden": str(hidden),
                      **maddpg_param_counts(env0.spec(), hidden, cfg["networks"])})
    comp_rows.append({"Algorithm": "TD3", "hidden": str(hidden),
                      **td3_param_counts(obs_dim, act_dim, hidden, cfg["networks"])})
    comp_rows.append({"Algorithm": "DDPG", "hidden": str(hidden),
                      **ddpg_param_counts(obs_dim, act_dim, hidden, cfg["networks"])})
    comp_rows.append({"Algorithm": "PPO", "hidden": str(hidden),
                      **ppo_param_counts(obs_dim, act_dim, hidden, cfg["networks"])})
    try:
        target = comp_rows[0]["total_params"]
        h_m = matched_td3_hidden_sizes(obs_dim, act_dim, target, cfg["networks"])
        comp_rows.append({"Algorithm": "TD3-Matched", "hidden": str(h_m),
                          **td3_param_counts(obs_dim, act_dim, h_m, cfg["networks"])})
    except ValueError as e:
        print(f"  TD3-Matched sizing failed: {e}")
    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(os.path.join(tab_dir, "model_complexity.csv"), index=False)
    _write_tex_table(os.path.join(tab_dir, "model_complexity.tex"), df_comp,
                     caption=("Trainable parameters (main networks; target copies excluded) "
                              "and estimated forward FLOPs. TD3-Matched is sized to match "
                              "the MADDPG total within 5\\%."),
                     label="tab:model_complexity")
    print(df_comp.to_string(index=False))

    # ------------------------------------------------ raw tidy CSV
    df_raw = pd.DataFrame(raw_rows)
    # Keep training and evaluation provenance distinct. In --final-paper mode
    # validate_shard_group guarantees equality; non-paper exploratory aggregate
    # runs remain auditable if evaluation code differs from training code.
    df_raw["training_source_sha"] = training_source_sha
    df_raw["evaluation_source_sha"] = evaluation_source_sha
    df_raw.to_csv(os.path.join(tab_dir, "results_raw.csv"), index=False)
    print(f"\nRaw tidy results: {os.path.join(tab_dir, 'results_raw.csv')} "
          f"({len(df_raw)} rows)")

    # ------------------------------------------------ simulation parameters
    e = cfg["env"]
    sim_params = {
        "Parameter": ["Formulation", "channel_rho", "K", "K_r", "N", "M",
                      "P_max (dBm)", "Noise (dBm)", "QoS min (b/s/Hz)",
                      "T-blockage (dB)", "PL exp direct", "PL exp BS-RIS",
                      "PL exp RIS-User", "Reward alpha", "R_ref",
                      "augmented_penalty_weight",
                      "dual_lr", "dual_ema", "dual_lambda_max",
                      "two_stage_dual_freeze_fraction",
                      "phase/power/beta switching cost",
                      "Episodes per algo", "Warmup steps",
                      "obs_norm_freeze_after_env_steps",
                      "Training seeds", "Validation seeds", "Test seeds (locked)"],
        "Value": [e.get("env_formulation", "dynamic_mdp"), e.get("channel_rho", ""),
                  e["num_users"], e["num_users_reflection"], e["num_ris_elements"],
                  e["num_bs_antennas"], e["p_max_dbm"], e["noise_power_dbm"],
                  e["qos_rate_min"], e["direct_block_loss_db"],
                  e["path_loss_exp_direct"], e["path_loss_exp_bs_ris"],
                  e["path_loss_exp_ris_user"], e["reward_alpha"],
                  e.get("reward_rate_reference", ""),
                  e.get("augmented_penalty_weight", 1.0),
                  e.get("dual_lr", ""), e.get("dual_ema", ""),
                  e.get("dual_lambda_max", ""),
                  e.get("two_stage_dual_freeze_fraction", ""),
                  f"{e.get('phase_switching_cost', 0)}/{e.get('power_switching_cost', 0)}/{e.get('beta_switching_cost', 0)}",
                  cfg["training"]["total_episodes"], cfg["maddpg"]["warmup_steps"],
                  e.get("obs_norm_freeze_after_env_steps", ""),
                  str(training_seeds), str(cfg["evaluation"]["validation_seeds"]),
                  str(cfg["evaluation"]["test_seeds"])],
    }
    df_sim = pd.DataFrame(sim_params)
    df_sim.to_csv(os.path.join(tab_dir, "simulation_parameters.csv"), index=False)
    _write_tex_table(os.path.join(tab_dir, "simulation_parameters.tex"),
                     df_sim, caption="Simulation parameters.",
                     label="tab:simulation_parameters")

    # ------------------------------------------------ report
    print("\n========== Generating results_summary.md ==========")
    report_path = os.path.join(out_root, "results_summary.md")
    _write_report(report_path, cfg, df_cmp, sweep, lat_cpu, abl, config_sha)
    print(f"Report: {report_path}")
    print("\nAll done.")


def _write_report(path, cfg, df_cmp, sweep, lat_cpu, abl, config_sha):
    e = cfg["env"]
    lines = []
    lines.append("# Results Summary - DRL Resource Allocation in STAR-RIS Assisted RSMA Networks\n\n")
    lines.append(f"Effective-config sha256: `{config_sha}`\n\n")
    lines.append("## 1. System Setup\n")
    lines.append(
        f"- Formulation: {e.get('env_formulation', 'dynamic_mdp')} "
        f"(Gauss-Markov rho = {e.get('channel_rho', 'n/a')}), MISO downlink "
        f"(M = {e['num_bs_antennas']}), "
        f"K = {e['num_users']} (K_R = {e['num_users_reflection']}), "
        f"N = {e['num_ris_elements']} STAR-RIS elements (ideal independent-phase ES).\n"
        f"- P_max = {e['p_max_dbm']} dBm, noise = {e['noise_power_dbm']} dBm, "
        f"R_min = {e['qos_rate_min']} b/s/Hz, T-blockage = {e['direct_block_loss_db']} dB.\n"
        "- Reward: penalty/Lagrangian surrogate of P0 with per-user projected "
        "dual updates (expected-rate constraint E[R_k] >= R_min). No per-step "
        "QoS guarantee is claimed.\n"
    )
    lines.append("## 2. Algorithm Comparison (locked test bank; CI across training seeds)\n")
    lines.append("```\n" + df_cmp.to_string(index=False) + "\n```\n")
    lines.append("## 3. Sum-rate vs P_max\n```\n")
    p_list = cfg["evaluation"]["power_sweep_dbm"]
    lines.append("Pmax(dBm) | " + " | ".join(sweep.keys()) + "\n")
    for i, p in enumerate(p_list):
        lines.append(f"{p:9.1f} | " + " | ".join(f"{sweep[a]['mean'][i]:.3f}" for a in sweep) + "\n")
    lines.append("```\n")
    lines.append("## 4. CPU Inference Latency (single thread; GPU table separate)\n```\n")
    lines.append("\n".join(f"  {k:22s}: median {v['median_ms']:.3f} ms  "
                           f"(mean {v['mean_ms']:.3f}, p95 {v['p95_ms']:.3f})"
                           for k, v in lat_cpu.items()) + "\n```\n")
    if abl is not None:
        lines.append("## 5. Ablation (AO-Grid = coarse heuristic, NOT an upper bound)\n```\n")
        for k, v in abl.items():
            lines.append(
                f"  {k:22s} sr={v['sum_rate_mean']:.3f} +/- {v['sum_rate_ci']:.3f}   "
                f"UQF={v['user_qos_fraction_mean']:.3f}   "
                f"AllQoS={v['all_users_qos_prob']:.3f}   "
                f"[{v['ci_unit']} x {v['n_units']}]\n")
        lines.append("```\n")
    lines.append("## 6. Metric definitions\n")
    lines.append(
        "- `UserQoSFraction` = (1/K) sum_k 1[R_k >= R_min] (fraction of users meeting QoS).\n"
        "- `AllUsersQoSProb` = P[min_k R_k >= R_min] (all users simultaneously).\n"
        "These are DIFFERENT quantities; tables always state which one is used.\n")
    lines.append("## 7. Limitations\n")
    lines.append(
        "- Penalty-surrogate training: expected-rate constraint, no hard per-user guarantee.\n"
        "- Ideal independent-phase ES STAR-RIS (continuous phases, perfect CSI, no coupling).\n"
        f"- M = {e['num_bs_antennas']} BS antennas (MISO, joint power projection).\n"
        "- AO local search is a local reference, not an optimum.\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
