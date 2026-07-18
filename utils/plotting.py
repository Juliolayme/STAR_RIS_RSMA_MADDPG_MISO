"""IEEE-quality figure utilities with CI bands + diagnostic plots."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colorblind-safe palette (Wong, 2011).
PALETTE = {
    "MADDPG":            "#0072B2",
    "DDPG":              "#D55E00",
    "TD3":               "#009E73",
    "TD3-Matched":       "#66CCEE",
    "PPO":               "#CC79A7",
    "FixedRIS":          "#E69F00",
    "RandomRIS":         "#56B4E9",
    "NoRIS":             "#999999",
    "AnalyticalRIS":     "#000000",
    "MaxMinAlignedRIS":  "#000000",   # honest name for the max-min single-user heuristic
    "AO-Grid":           "#4B0082",   # coarse AO-grid heuristic (NOT an upper bound)
    "AO-LocalSearch":    "#7B3FA0",   # Hybrid AO Local Search reference
    "BCD":               "#4B0082",   # deprecated alias of AO-Grid (legacy figures)
    "Learned":           "#0072B2",
    "EqualPower+Learned":"#882255",
    "EqualPower+Fixed":  "#AA4499",
}
MARKERS = {
    "MADDPG": "o", "DDPG": "s", "TD3": "D", "TD3-Matched": "d", "PPO": "^",
    "FixedRIS": "x", "RandomRIS": "v", "NoRIS": "P",
    "AnalyticalRIS": "*", "MaxMinAlignedRIS": "*",
    "AO-Grid": "h", "AO-LocalSearch": "H", "BCD": "h", "Learned": "o",
    "EqualPower+Learned": "P", "EqualPower+Fixed": "X",
}


def setup_ieee_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _save_figure(fig, out_dir: str, name: str, formats=("png", "pdf", "svg")):
    os.makedirs(out_dir, exist_ok=True)
    for ext in formats:
        path = os.path.join(out_dir, f"{name}.{ext}")
        try:
            fig.savefig(path, format=ext)
        except Exception as e:
            print(f"[plot] Failed to save {path}: {e}")
    plt.close(fig)


def _ma(y: np.ndarray, w: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size == 0 or w <= 1:
        return y.copy()
    w = min(w, y.size)
    k = np.ones(w) / w
    return np.convolve(y, k, mode="valid")


def seed_mean_and_ci(seed_matrix: np.ndarray, window: int = 30
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-episode mean and Student-t 95% CI ACROSS TRAINING SEEDS.

    seed_matrix: [n_seeds, n_episodes]. Each seed's curve is smoothed
    individually (moving average) BEFORE aggregating, then at every episode
    the mean and the Student-t 95% half-width across seeds are computed.
    This is a true between-seed confidence interval -- NOT a rolling temporal
    standard deviation (P0-6 reviewer fix).

    Returns (x_indices, mean_curve, ci_halfwidth).
    """
    from utils.metrics import student_t_crit_95
    m = np.asarray(seed_matrix, dtype=float)
    assert m.ndim == 2, "seed_mean_and_ci expects [n_seeds, n_episodes]"
    n_seeds, n_eps = m.shape
    w = min(window, max(1, n_eps // 5))
    if w > 1:
        smoothed = np.stack([_ma(m[i], w) for i in range(n_seeds)], axis=0)
        x = np.arange(w, n_eps + 1)
    else:
        smoothed = m
        x = np.arange(1, n_eps + 1)
    mean = smoothed.mean(axis=0)
    if n_seeds > 1:
        sd = smoothed.std(axis=0, ddof=1)
        ci = student_t_crit_95(n_seeds - 1) * sd / np.sqrt(n_seeds)
    else:
        ci = np.zeros_like(mean)
    return x, mean, ci


def plot_training_convergence(curves: dict[str, np.ndarray], out_dir: str,
                              name: str = "training_convergence",
                              ylabel: str = "Episode return",
                              window: int = 30):
    """Training curves with correct between-seed confidence intervals.

    curves[algo] is either:
      - a 2-D array [n_training_seeds, n_episodes]: each seed is smoothed
        individually, then mean and Student-t 95% CI across seeds are drawn
        (shaded band = TRUE between-seed CI);
      - a 1-D array: plotted as a plain line WITHOUT any band (a single run
        has no between-seed uncertainty; rolling std is never presented as a
        confidence interval).
    """
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    n_seeds_seen = []
    for algo, y in curves.items():
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            continue
        if y.ndim == 2:
            x, mean, ci = seed_mean_and_ci(y, window=window)
            n_seeds_seen.append(y.shape[0])
            ax.fill_between(x, mean - ci, mean + ci,
                            color=PALETTE.get(algo, "gray"), alpha=0.18, linewidth=0)
            ax.plot(x, mean, color=PALETTE.get(algo, None),
                    marker=MARKERS.get(algo, None),
                    markevery=max(1, mean.size // 12), label=algo)
        else:
            x = np.arange(1, y.size + 1)
            w = min(window, max(1, y.size // 5))
            y_s = _ma(y, w) if w > 1 else y
            x_s = x[w - 1:] if w > 1 else x
            ax.plot(x_s, y_s, color=PALETTE.get(algo, None),
                    marker=MARKERS.get(algo, None),
                    markevery=max(1, max(y_s.size, 1) // 12), label=algo)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    if n_seeds_seen:
        ax.set_title(f"Shaded: Student-t 95% CI over {max(n_seeds_seen)} training seeds",
                     fontsize=9)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    _save_figure(fig, out_dir, name)


def plot_metric_vs_x(x_values, results: dict[str, dict],
                     xlabel: str, ylabel: str,
                     out_dir: str, name: str):
    """results[algo] = {"mean": [...], "ci": [...]} — plotted with error bars."""
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    x = np.asarray(x_values, dtype=float)
    for algo, data in results.items():
        mean = np.asarray(data["mean"], dtype=float)
        ci = np.asarray(data.get("ci", np.zeros_like(mean)), dtype=float)
        ax.errorbar(x, mean, yerr=ci,
                    color=PALETTE.get(algo, None),
                    marker=MARKERS.get(algo, "o"),
                    capsize=3, label=algo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    _save_figure(fig, out_dir, name)


def plot_bar(x_labels, results: dict[str, float], out_dir: str, name: str,
             ylabel: str = "Value", ci: dict[str, float] | None = None):
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    keys = list(results.keys())
    vals = [results[k] for k in keys]
    colors = [PALETTE.get(k, "#666666") for k in keys]
    yerr = [ci[k] for k in keys] if ci else None
    positions = np.arange(len(keys))
    ax.bar(positions, vals, color=colors, edgecolor="black", linewidth=0.7,
           yerr=yerr, capsize=4)
    ax.set_xticks(positions)
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    _save_figure(fig, out_dir, name)


def plot_reward_decomposition(history: dict, out_dir: str, name: str = "reward_decomposition"):
    """Breakdown of the Lagrangian-surrogate reward terms over episodes."""
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    series = [
        ("reward_sr_mean", "#0072B2", "Sum-rate term"),
        ("reward_dual_mean", "#D55E00", "Dual term (-sum lambda_k c_k)"),
        ("reward_aug_mean", "#009E73", "Augmented penalty"),
        ("reward_switch_mean", "#CC79A7", "Switching cost"),
    ]
    plotted = False
    for key, color, label in series:
        y = np.asarray(history.get(key, []), dtype=float)
        if y.size == 0:
            continue
        x = np.arange(1, y.size + 1)
        ax.plot(x, y, color=color, label=label)
        plotted = True
    if not plotted:
        plt.close(fig); return
    ax.set_xlabel("Episode")
    ax.set_ylabel("Per-step reward (mean)")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    _save_figure(fig, out_dir, name)


def plot_qos_lambda(history: dict, out_dir: str, name: str = "qos_lambda"):
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    lam = np.asarray(history.get("qos_lambda", []), dtype=float)
    if lam.size == 0:
        plt.close(fig); return
    x = np.arange(1, lam.size + 1)
    ax.plot(x, lam, color="#882255")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Adaptive λ (QoS penalty)")
    fig.tight_layout()
    _save_figure(fig, out_dir, name)


def plot_phase_histogram(phase_samples: dict[str, np.ndarray],
                         out_dir: str, name: str = "phase_histogram"):
    """Histogram of RIS phases under each mode (should differ visibly across modes)."""
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for label, phi in phase_samples.items():
        phi = np.asarray(phi, dtype=float)
        if phi.size == 0:
            continue
        phi = np.mod(phi, 2 * np.pi)
        ax.hist(phi, bins=24, range=(0.0, 2 * np.pi), alpha=0.55,
                color=PALETTE.get(label, None), label=label, density=True,
                edgecolor="black", linewidth=0.4)
    ax.set_xlabel("RIS phase (rad)")
    ax.set_ylabel("Density")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    _save_figure(fig, out_dir, name)


def plot_pareto(points: dict[str, dict], out_dir: str, name: str = "pareto",
                xlabel: str = "Avg. sum-rate (b/s/Hz)",
                ylabel: str = "QoS satisfaction probability"):
    """Scatter of (sum-rate, QoS) per algorithm with horizontal+vertical CI bars."""
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for algo, p in points.items():
        x = float(p["sum_rate_mean"])
        y = float(p["qos_mean"])
        xerr = float(p.get("sum_rate_ci", 0.0))
        yerr = float(p.get("qos_ci", 0.0))
        ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                    fmt=MARKERS.get(algo, "o"),
                    color=PALETTE.get(algo, "#444444"),
                    markersize=10, capsize=4, label=algo)
        ax.annotate(algo, (x, y), xytext=(8, 6), textcoords="offset points",
                    fontsize=9, color=PALETTE.get(algo, "#444444"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    fig.tight_layout()
    _save_figure(fig, out_dir, name)


def plot_h_eff_distribution(h_eff_samples: dict[str, np.ndarray],
                            out_dir: str, name: str = "h_eff_distribution"):
    """Distribution of |h_eff| under each RIS mode (for T-region users)."""
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for label, h in h_eff_samples.items():
        h = np.asarray(h, dtype=float).reshape(-1)
        if h.size == 0:
            continue
        ax.hist(h, bins=40, alpha=0.55, label=label, density=True,
                color=PALETTE.get(label, None), edgecolor="black", linewidth=0.4)
    ax.set_xlabel("|h_eff|  (T-region users)")
    ax.set_ylabel("Density")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    _save_figure(fig, out_dir, name)
