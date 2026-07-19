"""Clean one-factor ablations for the structured BS/RIS controller.

Statistical units
-----------------
- Agent-dependent cells are evaluated for every independent MADDPG training
  seed; confidence intervals use training-seed-level means.
- Policy-independent cells (AO-Grid and the fully classical MRT/equal-power
  baseline) are evaluated once on the shared ScenarioBank; uncertainty uses
  independent scenarios.

The former ``EqualPower+Learned`` cell changed powers, beam directions and the
common split simultaneously. It is intentionally removed because it could not
identify which BS component caused the gain.
"""
from __future__ import annotations
import numpy as np

from utils.metrics import confidence_interval
from experiments.evaluate import eval_on_scenarios, scenario_rows
from experiments.baselines_ao import ao_reference_lambda


# (label, ris_mode, explicit env overrides, agent_dependent)
# Every learned one-factor cell changes exactly one decoder component.
ABLATION_CELLS = [
    ("Learned", "optimized", {}, True),
    ("AO-Grid", "ao_grid", {}, False),
    ("AnalyticalRIS", "analytical", {}, True),
    ("FixedRIS", "fixed", {}, True),
    ("RandomRIS", "random", {}, True),
    ("NoRIS", "none", {}, True),
    ("EqualPowerOnly", "optimized",
     {"force_equal_stream_power": True}, True),
    ("MRTDirectionsOnly", "optimized",
     {"force_mrt_directions": True}, True),
    ("UniformCommonSplitOnly", "optimized",
     {"force_uniform_common_split": True}, True),
    ("UniformCommonBeamOnly", "optimized",
     {"force_uniform_common_beam": True}, True),
    ("ClassicalMRTEqualPowerFixedRIS", "fixed", {
        "force_equal_stream_power": True,
        "force_mrt_directions": True,
        "force_uniform_common_split": True,
        "force_uniform_common_beam": True,
    }, False),
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
    """Evaluate clean structured-controller ablations on one ScenarioBank."""
    run_checkpoint_shas = run_checkpoint_shas or ["" for _ in runs]
    out = {}
    for label, ris_mode, env_overrides, agent_dependent in ABLATION_CELLS:
        aux_acc = {"rate_common": [], "h_eff_abs_T": [],
                   "phase_entropy_T": [], "common_power_frac": []}
        if agent_dependent:
            srs, uqfs, allqs = [], [], []
            for run, ck_sha in zip(runs, run_checkpoint_shas):
                lam_vec = run.get("trained_qos_lambda_vec")
                m = eval_on_scenarios(
                    run["agent"], "MADDPG", cfg, scenarios,
                    ris_mode=ris_mode, qos_lambda_vec=lam_vec,
                    env_overrides=env_overrides)
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
                        checkpoint_sha=ck_sha,
                        extra={"scenario": "ablation",
                               "env_overrides": str(env_overrides)}))
            aux = {k: float(np.mean(v)) for k, v in aux_acc.items()}
            out[label] = _cell_summary_from_values(
                srs, uqfs, allqs, aux, n_units=len(runs),
                unit="training_seed")
        else:
            run = runs[0]
            reference_lambda = ao_reference_lambda(cfg)
            m = eval_on_scenarios(
                run["agent"], "MADDPG", cfg, scenarios,
                ris_mode=ris_mode, qos_lambda_vec=reference_lambda,
                env_overrides=env_overrides)
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
                    extra={"scenario": "ablation",
                           "env_overrides": str(env_overrides)}))
    return out
