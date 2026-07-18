"""Expanded ablation across RIS modes and BS power policies.

Statistical units (P0-7 reviewer fix):
- Agent-dependent cells (anything whose power allocation or RIS phases come
  from a trained policy) are evaluated for EVERY training seed; the CI is
  computed across training-seed-level means (independent runs).
- Policy-independent cells (AO-Grid, EqualPower+Fixed) are evaluated ONCE on
  the evaluation ScenarioBank; their uncertainty is computed across the
  independent evaluation scenarios. They are never duplicated across training
  seeds.

All cells see the identical scenarios (ScenarioBank playback).
"""
from __future__ import annotations
import numpy as np

from utils.metrics import confidence_interval
from experiments.evaluate import eval_on_scenarios, scenario_rows
from experiments.baselines_ao import ao_reference_lambda


# (label, ris_mode, equal_power, agent_dependent)
# "AO-Grid" is the coarse alternating-optimization grid heuristic
# (env._coarse_ao_grid). It is a heuristic reference, NOT an upper bound.
ABLATION_CELLS = [
    ("Learned",            "optimized",  False, True),
    ("AO-Grid",            "ao_grid",    False, False),
    ("AnalyticalRIS",      "analytical", False, True),
    ("FixedRIS",           "fixed",      False, True),
    ("RandomRIS",          "random",     False, True),
    ("NoRIS",              "none",       False, True),
    ("EqualPower+Learned", "optimized",  True,  True),
    ("EqualPower+Fixed",   "fixed",      True,  False),
]


def _cell_summary_from_values(sr_vals, uqf_vals, allq_vals, aux: dict,
                              n_units: int, unit: str) -> dict:
    sr_m, sr_ci, sr_std = confidence_interval(np.asarray(sr_vals))
    uq_m, uq_ci, uq_std = confidence_interval(np.asarray(uqf_vals))
    aq_m, aq_ci, aq_std = confidence_interval(np.asarray(allq_vals))
    return {
        "sum_rate_mean": sr_m, "sum_rate_ci": sr_ci, "sum_rate_std": sr_std,
        "user_qos_fraction_mean": uq_m, "user_qos_fraction_ci": uq_ci,
        "all_users_qos_prob": aq_m, "all_users_qos_prob_ci": aq_ci,
        "n_units": n_units, "ci_unit": unit,
        **aux,
    }


def ablation_study(runs: list[dict], cfg: dict, scenarios: list[dict],
                   raw_rows: list[dict] | None = None,
                   config_sha: str = "", run_checkpoint_shas=None) -> dict:
    """Run the ablation.

    runs: list of per-training-seed MADDPG run-info dicts (from train_maddpg),
          each with keys "agent" and "trained_qos_lambda_vec".
    scenarios: evaluation ScenarioBank scenarios (identical for every cell).
    raw_rows: optional list collecting tidy per-scenario rows.
    config_sha / run_checkpoint_shas: provenance for the raw rows; the latter is
          a list aligned with `runs` giving each run's best.pt sha.
    """
    run_checkpoint_shas = run_checkpoint_shas or ["" for _ in runs]
    out = {}
    for label, ris_mode, equal_power, agent_dependent in ABLATION_CELLS:
        aux_acc = {"rate_common": [], "h_eff_abs_T": [],
                   "phase_entropy_T": [], "common_power_frac": []}
        if agent_dependent:
            # One evaluation per training seed; CI across training seeds.
            srs, uqfs, allqs = [], [], []
            for run, ck_sha in zip(runs, run_checkpoint_shas):
                lam_vec = run.get("trained_qos_lambda_vec")
                m = eval_on_scenarios(run["agent"], "MADDPG", cfg, scenarios,
                                      ris_mode=ris_mode, equal_power=equal_power,
                                      qos_lambda_vec=lam_vec)
                srs.append(m["sum_rate_mean"])
                uqfs.append(m["user_qos_fraction_mean"])
                allqs.append(m["all_users_qos_prob"])
                aux_acc["rate_common"].append(m["rate_common_mean"])
                aux_acc["h_eff_abs_T"].append(m["h_eff_abs_T_mean"])
                aux_acc["phase_entropy_T"].append(m["phase_entropy_T_mean"])
                aux_acc["common_power_frac"].append(m["common_power_frac_mean"])
                if raw_rows is not None:
                    raw_rows.extend(scenario_rows(
                        f"ablation:{label}", m, scenarios,
                        training_seed=run.get("seed"), config_sha=config_sha,
                        checkpoint_sha=ck_sha, extra={"scenario": "ablation"}))
            aux = {k: float(np.mean(v)) for k, v in aux_acc.items()}
            out[label] = _cell_summary_from_values(
                srs, uqfs, allqs, aux, n_units=len(runs), unit="training_seed")
        else:
            # Policy-independent: single evaluation; CI across scenarios.  Use
            # the one pre-registered AO reference vector, never runs[0]'s
            # trained dual variables (which would make AO-Grid seed-dependent).
            run = runs[0]
            reference_lambda = ao_reference_lambda(cfg)
            m = eval_on_scenarios(run["agent"], "MADDPG", cfg, scenarios,
                                  ris_mode=ris_mode, equal_power=equal_power,
                                  qos_lambda_vec=reference_lambda)
            aux = {k: float(m[f"{k}_mean"]) for k in aux_acc}
            out[label] = _cell_summary_from_values(
                m["per_episode_sum_rate"],
                m["per_episode_user_qos_fraction"],
                m["per_episode_all_users_qos"],
                aux, n_units=len(scenarios), unit="scenario")
            if raw_rows is not None:
                raw_rows.extend(scenario_rows(
                    f"ablation:{label}", m, scenarios, training_seed=None,
                    config_sha=config_sha, solver_config_sha=config_sha,
                    extra={"scenario": "ablation"}))
    return out
