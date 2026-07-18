"""Hybrid AO Local Search (SLSQP + projected gradient) baseline.

An alternating-optimization LOCAL-SEARCH reference for the P0 surrogate:

  Block 1 (RIS fixed): SLSQP over the full power simplex [P_c, P_1..P_K]
          and the common-split simplex [c_1..c_K]  (scipy.optimize, equality
          constraints + bounds).
  Block 2 (power fixed): PROJECTED GRADIENT ascent (finite-difference
          gradients, backtracking step) over the FULL per-element variable
          space: beta_r in [1e-4, 1-1e-4]^N, phi_r in R^N, phi_t in R^N
          (phases wrap mod 2pi). This block intentionally does NOT use SLSQP,
          hence the method name "Hybrid AO Local Search".

The objective is the SAME penalty/Lagrangian surrogate the DRL agents
optimize, INCLUDING the reconfiguration switching costs relative to a given
previous action:

  J = alpha * R_sum / R_ref - sum_k lambda_k * c_k
      - 0.5 * w * sum_k max(c_k, 0)^2
      - eta_phi * C_phase - eta_P * C_power - eta_beta * C_beta

with c_k = R_min - R_k evaluated on the CURRENT channel realization.

Properties:
  - multi-start (analytical prior + uniform + random restarts, dedicated
    solver RNG stream);
  - blocks only accept improving moves, so the objective trace is
    non-decreasing (asserted in tests);
  - returns feasibility/convergence flags and evaluation counts.

This is a LOCAL-search reference. It is NOT an upper bound and is never
described as one. Intended operating range: N <= 32 (cost grows with N due to
finite-difference gradients).
"""
from __future__ import annotations
import math
import time

import numpy as np

try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except ImportError:                                    # pragma: no cover
    _HAS_SCIPY = False


def ao_reference_lambda(cfg: dict) -> np.ndarray:
    """Return the pre-registered, policy-independent AO objective vector.

    This value must be selected from configuration/development evidence before
    final-test evaluation.  Falling back to a trained policy's dual variables
    would make a supposedly policy-independent baseline depend on an arbitrary
    training seed, so missing or malformed registrations are hard errors.
    """
    values = cfg.get("evaluation", {}).get("ao_reference_lambda")
    if values is None:
        raise ValueError("evaluation.ao_reference_lambda must be pre-registered")
    lam = np.asarray(values, dtype=np.float64).reshape(-1)
    n_users = int(cfg["env"]["num_users"])
    if lam.shape != (n_users,) or not np.all(np.isfinite(lam)) or np.any(lam < 0.0):
        raise ValueError(
            f"evaluation.ao_reference_lambda must contain {n_users} finite, nonnegative values")
    return lam.copy()


def stratified_ao_scenarios(scenarios: list[dict], per_seed: int = 1) -> list[dict]:
    """Select the first ``per_seed`` scenarios from every evaluation seed."""
    per_seed = int(per_seed)
    if per_seed < 1:
        raise ValueError("AO scenarios per seed must be at least one")
    selected: list[dict] = []
    counts: dict[int, int] = {}
    for scenario in scenarios:
        seed = int(scenario["evaluation_seed"])
        if counts.get(seed, 0) < per_seed:
            selected.append(scenario)
            counts[seed] = counts.get(seed, 0) + 1
    available = {int(s["evaluation_seed"]) for s in scenarios}
    if set(counts) != available:
        raise RuntimeError("Stratified AO selection failed to cover every evaluation seed")
    return selected


def solver_params_from_config(cfg: dict) -> dict:
    """Pre-registered AO solver hyperparameters (V4 review item 7).

    They must be FROZEN using validation evidence only, before the locked test
    bank is opened, and they participate in the solver_config_sha so any later
    change is visible. Defaults match the constructor.
    """
    raw = dict((cfg.get("evaluation", {}) or {}).get("ao_solver", {}) or {})
    allowed = {"n_starts", "max_outer", "tol", "pg_steps", "pg_lr", "fd_eps"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown evaluation.ao_solver keys: {sorted(unknown)}")
    return raw


class AOHybridLocalSearch:
    name = "Hybrid AO Local Search (SLSQP + projected gradient)"

    def __init__(self, env, n_starts: int = 3, max_outer: int = 8,
                 tol: float = 1e-5, pg_steps: int = 40, pg_lr: float = 0.3,
                 fd_eps: float = 1e-4, seed: int = 0,
                 objective: str = "penalty"):
        if not _HAS_SCIPY:
            raise ImportError("AOHybridLocalSearch requires scipy (pip install scipy)")
        if objective not in ("penalty", "maxmin"):
            raise ValueError(f"Unknown AO objective: {objective!r}")
        self.env = env
        self.n_starts = int(n_starts)
        self.max_outer = int(max_outer)
        self.tol = float(tol)
        self.pg_steps = int(pg_steps)
        self.pg_lr = float(pg_lr)
        self.fd_eps = float(fd_eps)
        # "penalty": the same Lagrangian surrogate the DRL agents optimize.
        # "maxmin": J = min_k R_k -- used by the R_min feasibility study
        # (V4 review item 10), NOT for the baseline comparison.
        self.objective_kind = objective
        # Dedicated solver RNG stream (separate from env/policy RNGs).
        self.solver_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
        self.n_evals = 0

    # ------------------------------------------------------------ objective
    def _objective(self, powers_split: np.ndarray, beta: np.ndarray,
                   phi_r: np.ndarray, phi_t: np.ndarray,
                   prev_applied: dict | None) -> float:
        """Objective to MAXIMIZE. "penalty" matches the env reward (without
        global reward_scale and the optional shaping bonus); "maxmin" is the
        worst-user rate for the feasibility study."""
        env = self.env
        K = env.K
        pw = powers_split[: K + 1]
        cs = powers_split[K + 1:]
        P_c = float(pw[0] * env.p_max)
        P_k = (pw[1:] * env.p_max).astype(np.float64)
        h_eff = env._effective_channels(beta, phi_r, phi_t)
        rs = env._rsma_rates(h_eff, P_c, P_k, cs)
        self.n_evals += 1
        if self.objective_kind == "maxmin":
            return float(np.min(rs["per_user"]))
        c_signed = env.qos_min - rs["per_user"]
        deficit = np.maximum(c_signed, 0.0)
        J = env.r_alpha * rs["sum_rate"] / max(env.r_ref, 1e-12)
        J -= float(np.dot(env.qos_lambda_vec, c_signed))
        J -= 0.5 * env.augmented_penalty_weight * float((deficit ** 2).sum())
        if prev_applied is not None:
            dphi = np.concatenate([phi_r - prev_applied["phi_r"],
                                   phi_t - prev_applied["phi_t"]])
            phase_cost = float(np.mean(1.0 - np.cos(dphi))) if dphi.size else 0.0
            power_cost = float(np.abs(pw - prev_applied["power_weights"]).sum() / (K + 1))
            beta_cost = float(np.mean(np.abs(beta - prev_applied["beta_r"])))
            J -= (env.eta_phase * phase_cost + env.eta_power * power_cost
                  + env.eta_beta * beta_cost)
        return float(J)

    # ------------------------------------------------------------ block 1: SLSQP
    def _optimize_power_split(self, x0: np.ndarray, beta, phi_r, phi_t,
                              prev_applied) -> tuple[np.ndarray, float]:
        K = self.env.K
        n = 2 * K + 1

        def neg_obj(x):
            return -self._objective(x, beta, phi_r, phi_t, prev_applied)

        constraints = [
            {"type": "eq", "fun": lambda x: float(x[: K + 1].sum() - 1.0)},
            {"type": "eq", "fun": lambda x: float(x[K + 1:].sum() - 1.0)},
        ]
        bounds = [(1e-6, 1.0)] * n
        res = minimize(neg_obj, x0, method="SLSQP", bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 60, "ftol": 1e-8})
        x = np.clip(res.x, 1e-6, 1.0)
        x[: K + 1] = x[: K + 1] / x[: K + 1].sum()
        x[K + 1:] = x[K + 1:] / x[K + 1:].sum()
        return x, self._objective(x, beta, phi_r, phi_t, prev_applied)

    # ------------------------------------------------------------ block 2: projected gradient
    def _optimize_ris(self, powers_split: np.ndarray, beta0, phi_r0, phi_t0,
                      prev_applied) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        env = self.env
        N = env.N

        def pack(beta, phi_r, phi_t):
            return np.concatenate([beta, phi_r, phi_t])

        def unpack(z):
            return z[:N], z[N:2 * N], z[2 * N:]

        def obj(z):
            b, pr, pt = unpack(z)
            return self._objective(powers_split, b, pr, pt, prev_applied)

        def project(z):
            b, pr, pt = unpack(z)
            b = np.clip(b, 1e-4, 1.0 - 1e-4)
            pr = np.mod(pr, 2 * math.pi)
            pt = np.mod(pt, 2 * math.pi)
            return pack(b, pr, pt)

        z = project(pack(np.asarray(beta0, dtype=np.float64).copy(),
                         np.asarray(phi_r0, dtype=np.float64).copy(),
                         np.asarray(phi_t0, dtype=np.float64).copy()))
        f = obj(z)
        lr = self.pg_lr
        for _ in range(self.pg_steps):
            # Central finite-difference gradient.
            grad = np.zeros_like(z)
            for i in range(z.size):
                zp = z.copy(); zp[i] += self.fd_eps
                zm = z.copy(); zm[i] -= self.fd_eps
                grad[i] = (obj(project(zp)) - obj(project(zm))) / (2 * self.fd_eps)
            gnorm = float(np.linalg.norm(grad))
            if gnorm < 1e-10:
                break
            # Backtracking: only accept improving steps (keeps the trace
            # non-decreasing).
            step = lr
            improved = False
            for _bt in range(6):
                z_new = project(z + step * grad / max(gnorm, 1e-12))
                f_new = obj(z_new)
                if f_new > f + 1e-12:
                    z, f = z_new, f_new
                    improved = True
                    break
                step *= 0.5
            if not improved:
                break
        b, pr, pt = unpack(z)
        return b, pr, pt, f

    # ------------------------------------------------------------ solve
    def _initial_points(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        env = self.env
        N = env.N
        prior_r, prior_t = env._analytical_phases()
        starts = [
            (0.5 * np.ones(N), prior_r.copy(), prior_t.copy()),
            (0.5 * np.ones(N), np.zeros(N), np.zeros(N)),
        ]
        while len(starts) < self.n_starts:
            starts.append((
                self.solver_rng.uniform(0.1, 0.9, size=N),
                self.solver_rng.uniform(0.0, 2 * math.pi, size=N),
                self.solver_rng.uniform(0.0, 2 * math.pi, size=N),
            ))
        return starts[: max(self.n_starts, 2)]

    def solve(self, prev_applied: dict | None = None) -> dict:
        """Run multi-start AO on the env's CURRENT channel realization.

        prev_applied: previously applied physical action dict (phi_r, phi_t,
        beta_r, power_weights) for the switching-cost terms; None for the
        first decision of an episode (costs are zero then, matching the env).
        """
        t_start = time.perf_counter()
        env = self.env
        K = env.K
        self.n_evals = 0
        best = None

        for beta0, phi_r0, phi_t0 in self._initial_points():
            x = np.concatenate([np.full(K + 1, 1.0 / (K + 1)),
                                np.full(K, 1.0 / K)])
            beta, phi_r, phi_t = beta0.copy(), phi_r0.copy(), phi_t0.copy()
            f_prev = self._objective(x, beta, phi_r, phi_t, prev_applied)
            trace = [f_prev]
            converged = False
            for _ in range(self.max_outer):
                x_new, f1 = self._optimize_power_split(x, beta, phi_r, phi_t, prev_applied)
                if f1 >= trace[-1] - 1e-12:
                    x = x_new
                    trace.append(max(f1, trace[-1]))
                else:
                    trace.append(trace[-1])
                b_new, pr_new, pt_new, f2 = self._optimize_ris(x, beta, phi_r, phi_t, prev_applied)
                if f2 >= trace[-1] - 1e-12:
                    beta, phi_r, phi_t = b_new, pr_new, pt_new
                    trace.append(max(f2, trace[-1]))
                else:
                    trace.append(trace[-1])
                if abs(trace[-1] - f_prev) < self.tol:
                    converged = True
                    break
                f_prev = trace[-1]

            final_f = trace[-1]
            if best is None or final_f > best["objective"]:
                pw = x[: K + 1]
                cs = x[K + 1:]
                P_c = float(pw[0] * env.p_max)
                P_k = (pw[1:] * env.p_max).astype(np.float64)
                h_eff = env._effective_channels(beta, phi_r, phi_t)
                rs = env._rsma_rates(h_eff, P_c, P_k, cs)
                c_signed = env.qos_min - rs["per_user"]
                best = {
                    "P_c": P_c, "P_k": P_k,
                    "power_weights": pw.copy(), "common_split": cs.copy(),
                    "beta_r": beta.copy(), "phi_r": phi_r.copy(), "phi_t": phi_t.copy(),
                    "objective": final_f,
                    "objective_trace": list(map(float, trace)),
                    "sum_rate": rs["sum_rate"],
                    "per_user_rate": rs["per_user"].copy(),
                    "qos_feasible": bool(np.all(c_signed <= 1e-9)),
                    "converged": converged,
                }
        best["n_evals"] = self.n_evals
        # Wall-clock per solve (V4 review item 7): required latency evidence
        # for the AO reference alongside n_evals.
        best["solve_time_ms"] = (time.perf_counter() - t_start) * 1000.0
        return best
