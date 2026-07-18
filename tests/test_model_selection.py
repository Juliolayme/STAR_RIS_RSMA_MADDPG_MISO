"""Lexicographic best-checkpoint selection matched to the expected-rate
constraint (validation-only, never test seeds)."""
from __future__ import annotations

from experiments.train import BestCheckpointSelector


def _metrics(c_bar, sr):
    return {"c_bar_per_user": c_bar, "sum_rate_mean": sr}


def test_feasible_beats_infeasible_regardless_of_sum_rate():
    sel = BestCheckpointSelector(tolerance=0.0)
    infeasible_high_sr = _metrics([0.05, -0.1], sr=9.0)
    feasible_low_sr = _metrics([-0.01, -0.2], sr=3.0)
    assert sel.key(feasible_low_sr) < sel.key(infeasible_high_sr)


def test_among_feasible_highest_sum_rate_wins():
    sel = BestCheckpointSelector(tolerance=0.0)
    a = _metrics([-0.01, -0.02], sr=3.0)
    b = _metrics([-0.30, -0.40], sr=2.5)
    assert sel.key(a) < sel.key(b)


def test_among_infeasible_smallest_max_violation_wins():
    sel = BestCheckpointSelector(tolerance=0.0)
    a = _metrics([0.10, 0.00], sr=1.0)   # max violation 0.10
    b = _metrics([0.20, -0.5], sr=9.0)   # max violation 0.20
    assert sel.key(a) < sel.key(b)


def test_tie_break_mean_violation_then_sum_rate():
    sel = BestCheckpointSelector(tolerance=0.0)
    a = _metrics([0.10, 0.00], sr=1.0)   # mean violation 0.05
    b = _metrics([0.10, 0.10], sr=9.0)   # mean violation 0.10
    assert sel.key(a) < sel.key(b)
    c = _metrics([0.10, 0.00], sr=2.0)   # same violations, higher SR
    assert sel.key(c) < sel.key(a)


def test_tolerance_shifts_feasibility():
    sel = BestCheckpointSelector(tolerance=0.05)
    slightly_violating = _metrics([0.04, -0.1], sr=5.0)
    assert sel.key(slightly_violating)[0] == 0    # feasible under tolerance


def test_consider_tracks_best_deterministically():
    sel = BestCheckpointSelector(tolerance=0.0)
    assert sel.consider(_metrics([0.2, 0.1], sr=2.0), episode=10)
    assert sel.consider(_metrics([-0.1, -0.1], sr=1.0), episode=20)   # feasible
    assert not sel.consider(_metrics([0.0, 0.0], sr=0.5), episode=30)
    assert sel.best_info["episode"] == 20
