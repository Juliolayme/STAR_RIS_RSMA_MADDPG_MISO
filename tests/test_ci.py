"""Confidence-interval computation across training seeds (P0-6)."""
from __future__ import annotations
import numpy as np

from utils.metrics import confidence_interval, student_t_crit_95
from utils.plotting import seed_mean_and_ci


def test_confidence_interval_students_t():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, half, std = confidence_interval(x)
    assert abs(mean - 3.0) < 1e-12
    expected = student_t_crit_95(4) * x.std(ddof=1) / np.sqrt(5)
    assert abs(half - expected) < 1e-9


def test_seed_mean_and_ci_on_constant_seeds():
    """Rows are constant curves: mean = mean of constants, CI matches the
    Student-t formula across seeds -- and is NOT a rolling temporal std."""
    consts = np.array([1.0, 2.0, 3.0, 4.0])
    n_eps = 50
    mat = np.tile(consts[:, None], (1, n_eps))
    x, mean, ci = seed_mean_and_ci(mat, window=10)
    np.testing.assert_allclose(mean, consts.mean())
    expected = student_t_crit_95(3) * consts.std(ddof=1) / np.sqrt(4)
    np.testing.assert_allclose(ci, expected, rtol=1e-9)
    # A rolling TEMPORAL std of the mean curve would be zero here; the
    # between-seed CI must not be.
    assert np.all(ci > 0.1)


def test_seed_mean_and_ci_single_seed_has_zero_band():
    mat = np.random.default_rng(0).normal(size=(1, 40))
    _, mean, ci = seed_mean_and_ci(mat, window=5)
    np.testing.assert_allclose(ci, 0.0)


def test_seed_mean_and_ci_smooths_each_seed_before_aggregation():
    # Two seeds with opposite alternating noise: per-seed smoothing (window 2)
    # cancels the alternation, so the CI shrinks close to zero.
    n = 40
    base = np.linspace(0, 1, n)
    alt = np.array([0.5 if i % 2 == 0 else -0.5 for i in range(n)])
    mat = np.stack([base + alt, base - alt])
    _, _, ci_smoothed = seed_mean_and_ci(mat, window=2)
    _, _, ci_raw = seed_mean_and_ci(mat, window=1)
    assert ci_smoothed.mean() < ci_raw.mean() * 0.2
