"""Common numerical helpers for the wireless / RL stack."""
from __future__ import annotations
import numpy as np


def db_to_lin(x_db: float | np.ndarray) -> float | np.ndarray:
    return 10.0 ** (np.asarray(x_db) / 10.0)


def dbm_to_watt(x_dbm: float | np.ndarray) -> float | np.ndarray:
    return 10.0 ** ((np.asarray(x_dbm) - 30.0) / 10.0)


def watt_to_dbm(x_watt: float | np.ndarray) -> float | np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(x_watt), 1e-30)) + 30.0


def safe_log2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log2(np.maximum(x, eps))


def free_space_path_loss_db(distance_m: np.ndarray,
                            ref_pl_db: float,
                            ref_d: float,
                            exponent: float) -> np.ndarray:
    d = np.maximum(np.asarray(distance_m), ref_d)
    return ref_pl_db + 10.0 * exponent * np.log10(d / ref_d)


def _student_t_sf(t: float, df: float) -> float:
    """Survival function of Student-t (i.e., 1 - CDF) for |t|, df > 0.

    Uses the relationship with the regularized incomplete beta function:
        P(T > |t|) = 0.5 * I_x(df/2, 1/2),  where x = df / (df + t^2)
    Implemented via a numerically stable continued-fraction for I_x (Lentz's method).
    Pure numpy / math — no scipy dependency. R3 reviewer fix.
    """
    import math
    t_abs = abs(float(t))
    df = max(float(df), 1.0)
    x = df / (df + t_abs * t_abs)
    a, b = df / 2.0, 0.5
    # log B(a, b) via lgamma.
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    # Edge cases of the regularized incomplete beta I_x(a, b):
    #   x = 0  → I_x = 0  → sf = 0   (|t| = ∞)
    #   x = 1  → I_x = 1  → sf = 0.5 (|t| = 0, no evidence of difference)
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 0.5
    log_front = a * math.log(x) + b * math.log(1.0 - x) - log_beta - math.log(a)
    front = math.exp(log_front)
    # Continued-fraction expansion for I_x(a, b) — Numerical Recipes 6.4 (betacf).
    eps = 3.0e-7
    fpmin = 1.0e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    incbeta = front * h
    # P(T > |t|) = 0.5 * I_x(df/2, 1/2)
    return 0.5 * float(incbeta)


def welch_ttest_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Welch's t-test (unequal variances). Returns two-sided p-value.

    Uses Student-t CDF (R3 reviewer fix) — the previous normal-approximation grossly
    overstated significance for small df (e.g., n=3 → df=2 has critical t≈4.3, not 1.96).
    """
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return 1.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if va == 0 and vb == 0:
        return 1.0
    se2 = va / na + vb / nb
    t = (a.mean() - b.mean()) / np.sqrt(max(se2, 1e-30))
    df = (se2 ** 2) / ((va / na) ** 2 / max(na - 1, 1) + (vb / nb) ** 2 / max(nb - 1, 1) + 1e-30)
    # Two-sided p-value via Student-t survival function.
    return 2.0 * _student_t_sf(t, df)


def student_t_crit_95(df: int) -> float:
    """Two-sided 95% Student-t critical value (tabulated, no scipy)."""
    _t95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
            7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131,
            20: 2.086, 25: 2.060, 29: 2.045}
    df = max(int(df), 1)
    if df >= 30:
        return 1.96
    keys = sorted(_t95.keys())
    chosen = keys[0]
    for k in keys:
        if k <= df:
            chosen = k
    return _t95[chosen]


def paired_t_test_p(a: np.ndarray, b: np.ndarray) -> float:
    """Paired two-sided t-test on matched samples (e.g. same training seeds).

    Returns the two-sided p-value computed from the Student-t survival
    function of the mean paired difference.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.size == b.size, "paired test requires matched samples"
    d = a - b
    n = d.size
    if n < 2:
        return 1.0
    sd = d.std(ddof=1)
    if sd == 0:
        return 1.0 if abs(d.mean()) < 1e-15 else 0.0
    t = d.mean() / (sd / np.sqrt(n))
    return 2.0 * _student_t_sf(t, n - 1)


def paired_permutation_p(a: np.ndarray, b: np.ndarray,
                         n_perm: int = 10000, seed: int = 0) -> float:
    """Exact-style sign-flip permutation test on paired differences.

    Two-sided p-value: fraction of sign-flipped mean differences whose
    absolute value is at least the observed one (add-one smoothing).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.size == b.size, "paired test requires matched samples"
    d = a - b
    n = d.size
    if n < 2:
        return 1.0
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    if n <= 20:
        # Enumerate all 2^n sign patterns when feasible.
        count = 0
        total = 2 ** n
        for mask in range(total):
            signs = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(n)])
            if abs((d * signs).mean()) >= obs - 1e-15:
                count += 1
        return count / total
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = np.abs((signs * d[None, :]).mean(axis=1))
    return float((np.sum(perm_means >= obs - 1e-15) + 1) / (n_perm + 1))


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in the
    original order (monotone, clipped to 1)."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p_values[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Paired-samples effect size: mean(d) / std(d)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    if d.size < 2 or d.std(ddof=1) == 0:
        return 0.0
    return float(d.mean() / d.std(ddof=1))


def paired_difference_ci(a: np.ndarray, b: np.ndarray, conf: float = 0.95
                         ) -> tuple[float, float]:
    """(mean difference, 95% CI half-width) of paired samples."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    n = d.size
    if n < 2:
        return float(d.mean()) if n else 0.0, 0.0
    crit = student_t_crit_95(n - 1)
    return float(d.mean()), float(crit * d.std(ddof=1) / np.sqrt(n))


def confidence_interval(samples: np.ndarray, conf: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, half-width-CI, std). Uses Student-t distribution for small n (n<30),
    normal approximation for large n. M5 reviewer fix — for n=5, t_crit≈2.78 vs z=1.96.
    """
    samples = np.asarray(samples, dtype=np.float64)
    n = samples.size
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(samples.mean())
    std = float(samples.std(ddof=1)) if n > 1 else 0.0
    if n >= 30:
        crit = 1.96 if abs(conf - 0.95) < 1e-6 else 1.645
    else:
        # Pre-tabulated Student-t critical values at 95% (two-sided), df = n-1.
        _t95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
                7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131,
                20: 2.086, 25: 2.060, 29: 2.045}
        df = max(n - 1, 1)
        # Find closest df ≤ given df in the table.
        keys = sorted(_t95.keys())
        chosen = keys[0]
        for k in keys:
            if k <= df:
                chosen = k
        crit = _t95[chosen] if abs(conf - 0.95) < 1e-6 else 1.645
    half = crit * std / np.sqrt(max(n, 1))
    return mean, half, std


def moving_average(x: np.ndarray, w: int = 20) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if w <= 1 or x.size <= 1:
        return x.copy()
    w = min(w, x.size)
    cs = np.cumsum(np.insert(x, 0, 0.0))
    ma = (cs[w:] - cs[:-w]) / w
    pad = np.full(w - 1, ma[0] if ma.size else 0.0)
    return np.concatenate([pad, ma])
