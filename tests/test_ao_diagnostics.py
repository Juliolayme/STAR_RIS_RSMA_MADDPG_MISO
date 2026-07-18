"""Validation-only Hybrid AO diagnostic coverage."""
from __future__ import annotations

import pytest

from conftest import full_cfg
from experiments.ao_diagnostics import run_ao_diagnostics


def _tiny_cfg():
    cfg = full_cfg(num_ris_elements=4, num_users=2, num_users_reflection=1,
                   max_steps=2)
    cfg["evaluation"]["validation_seeds"] = [101]
    cfg["evaluation"]["num_episodes"] = 1
    cfg["evaluation"]["ao_reference_lambda"] = [1.0, 1.0]
    cfg["evaluation"]["ao_solver"] = {
        "n_starts": 2, "max_outer": 2, "tol": 1e-4,
        "pg_steps": 4, "pg_lr": 0.3,
    }
    return cfg


def test_validation_ao_diagnostics_reports_freeze_metrics():
    df = run_ao_diagnostics(_tiny_cfg(), split="validation", scenarios_per_seed=1)
    assert len(df) == 1
    for column in ("sum_rate", "max_qos_violation", "objective",
                   "objective_gain", "converged", "n_evals", "solve_time_ms"):
        assert column in df.columns


def test_ao_diagnostics_refuses_locked_test_split():
    with pytest.raises(ValueError, match="locked test"):
        run_ao_diagnostics(_tiny_cfg(), split="test")
