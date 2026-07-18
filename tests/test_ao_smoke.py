"""Tiny AO local-search smoke test (item 12): the full AO reference is disabled
in config/smoke.yaml so the smoke run finishes quickly; this fast test still
exercises the sequential-block AO playback path (advance_to_next_block +
switching-cost warm start) end to end on a minimal problem."""
from __future__ import annotations
import numpy as np
import pytest

pytest.importorskip("scipy")

from env import StarRisRsmaEnv, ScenarioBank
from experiments.baselines_ao import AOHybridLocalSearch
from conftest import base_env_cfg


def test_ao_sequential_blocks_smoke():
    cfg = base_env_cfg(num_ris_elements=4, num_users=2, num_users_reflection=1,
                       max_steps=3, channel_rho=0.8)
    bank = ScenarioBank(cfg, split="test", evaluation_seeds=[9101], episodes_per_seed=1)
    env = StarRisRsmaEnv(cfg, seed=1)
    env.reset(options={"scenario": bank[0]})
    solver = AOHybridLocalSearch(env, n_starts=2, max_outer=2, pg_steps=4, seed=1)
    prev = None
    sr = []
    for b in range(3):
        assert env.current_step() == b       # innovation index advances 0,1,2
        sol = solver.solve(prev_applied=prev)
        sr.append(float(sol["sum_rate"]))
        assert np.all(sol["beta_r"] >= 1e-4) and np.all(sol["beta_r"] <= 1 - 1e-4)
        prev = {"phi_r": sol["phi_r"], "phi_t": sol["phi_t"],
                "beta_r": sol["beta_r"], "power_weights": sol["power_weights"]}
        env.advance_to_next_block()
    assert len(sr) == 3 and all(np.isfinite(sr))
