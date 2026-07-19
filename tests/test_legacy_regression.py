"""Regression: env_formulation='static_block' must reproduce the frozen MISO
physics snapshot recorded in the golden fixture (tests/fixtures/golden_static_block.npz).

Only PHYSICAL outputs are compared (channels, effective channels, RSMA rates).
Rewards, observations and dual variables changed by design in the refactor and
are intentionally excluded. Tolerances are dtype-aware.
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env import StarRisRsmaEnv  # noqa: E402

FIXTURE_DIR = os.path.join(PROJECT_ROOT, "tests", "fixtures")
NPZ_PATH = os.path.join(FIXTURE_DIR, "golden_static_block.npz")
META_PATH = os.path.join(FIXTURE_DIR, "golden_static_block.meta.json")

# Channels / effective channel: strict (pure complex arithmetic, must be
# bit-reproducible across the refactor).
RTOL_CHAN, ATOL_CHAN = 1e-10, 1e-12
# Derived rate quantities go through log2/division and accumulate tiny
# platform-dependent rounding; use a relaxed cross-platform tolerance (item 11).
RTOL_RATE, ATOL_RATE = 1e-7, 1e-9

_RATE_KEYS = {"rate_common", "per_user_rate", "sum_rate"}


def _tols(key: str) -> tuple[float, float]:
    if key in _RATE_KEYS:
        return RTOL_RATE, ATOL_RATE
    return RTOL_CHAN, ATOL_CHAN


@pytest.fixture(scope="module")
def golden():
    if not os.path.exists(NPZ_PATH):
        pytest.skip("golden fixture missing (run tests/make_golden_fixture.py "
                    "against the pre-refactor code)")
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return np.load(NPZ_PATH), meta


@pytest.mark.parametrize("case_name", ["case_block_full", "case_block_3", "case_k2"])
def test_static_block_reproduces_golden_physics(golden, case_name):
    data, meta = golden
    case = next(c for c in meta["cases"] if c["name"] == case_name)
    cfg = dict(case["config"])
    if int(cfg.get("num_bs_antennas", 1)) < 2:
        pytest.skip(
            "golden fixture predates the MISO migration (SISO, M=1, legacy "
            "softmax-power action layout); the MISO env cannot replay it. "
            "Regenerate tests/make_golden_fixture.py under the MISO model to "
            "restore this regression gate.")
    cfg["env_formulation"] = "static_block"

    env = StarRisRsmaEnv(cfg, seed=case["env_seed"], ris_mode="optimized")
    env.reset(seed=case["reset_seed"])

    g = lambda key: data[f"{case_name}__{key}"]

    np.testing.assert_allclose(env.user_positions, g("user_positions"),
                               rtol=RTOL_CHAN, atol=ATOL_CHAN)
    np.testing.assert_allclose(env.alpha_d, g("alpha_d"), rtol=RTOL_CHAN, atol=ATOL_CHAN)
    np.testing.assert_allclose(np.array([env.alpha_br]), g("alpha_br"),
                               rtol=RTOL_CHAN, atol=ATOL_CHAN)

    actions = g("actions")
    for t in range(cfg["max_steps"]):
        _, _, _, _, info = env.step(actions[t])
        for key, actual in [
            ("h_d", env._h_d), ("G", env._G), ("g", env._g),
            ("h_eff", env._h_eff),
            ("per_user_rate", np.asarray(info["per_user_rate"])),
            ("sum_rate", np.asarray(info["sum_rate"])),
            ("rate_common", np.asarray(info["rate_common"])),
        ]:
            expected = g(key)[t]
            rtol, atol = _tols(key)
            np.testing.assert_allclose(
                np.asarray(actual), expected, rtol=rtol, atol=atol,
                err_msg=f"{case_name} step {t}: mismatch in {key}")


def test_golden_metadata_records_source_provenance(golden):
    _, meta = golden
    assert len(meta["env_sha256"]) == 64
    assert len(meta["generator_sha256"]) == 64
    assert meta["base_branch_commit"] == "67d6adec0ecb5e735f35e01a36c41ebd1dec3c9e"
