"""Validation-only R_min feasibility study (V4 review item 10)."""
from __future__ import annotations
import numpy as np
import pytest

pytest.importorskip("scipy")

from experiments.feasibility import run_feasibility
from conftest import full_cfg


def _tiny_cfg():
    cfg = full_cfg(num_ris_elements=4, num_users=2, num_users_reflection=1)
    cfg["evaluation"]["validation_seeds"] = [101]
    cfg["evaluation"]["num_episodes"] = 1
    cfg["evaluation"]["ao_reference_lambda"] = [1.0, 1.0]
    cfg["evaluation"]["ao_solver"] = {"n_starts": 2, "max_outer": 2,
                                      "tol": 1e-4, "pg_steps": 4, "pg_lr": 0.3}
    return cfg


def test_feasibility_runs_on_validation_and_reports_maxmin():
    df = run_feasibility(_tiny_cfg(), split="validation", scenarios_per_seed=1)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["scenario_id"].startswith("validation_seed101")
    assert np.isfinite(row["maxmin_rate"])
    assert row["feasible"] == (row["maxmin_rate"] >= row["R_min"])
    assert row["solve_time_ms"] > 0


def test_feasibility_refuses_locked_test_split():
    with pytest.raises(ValueError, match="locked test"):
        run_feasibility(_tiny_cfg(), split="test")
