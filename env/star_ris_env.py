"""STAR-RIS assisted RSMA MISO downlink Gymnasium environment.

System model
------------
- One BS with M >= 2 antennas, one STAR-RIS with N elements (ES mode), K users
  partitioned into a reflection (R) region and a transmission (T) region.
- All channels: Rayleigh small-scale + log-distance large-scale path loss.
- Direct link h_dk in C^{M}: BS -> user k.
- BS->RIS:  G in C^{N x M}.
- RIS->user_k: g_k in C^{N}.
- STAR-RIS coefficients per element n (ideal independent-phase ES model):
     beta_n^r, beta_n^t in [0, 1] with beta_n^r + beta_n^t = 1
     phi_n^r, phi_n^t in [0, 2pi)
- Effective channel is stored as h_eff,k in C^M such that the scalar received
  channel is h_eff,k^H w = h_dk^H w + g_k^H Phi_k G w. Equivalently,
  h_eff,k = h_dk + G^H Phi_k^H g_k.

Environment formulations (config key `env_formulation`)
--------------------------------------------------------
- "dynamic_mdp" (default): each control step corresponds to one channel
  coherence block. Small-scale fading evolves via a first-order Gauss-Markov
  process  h_{t+1} = rho * h_t + sqrt(1 - rho^2) * eps_t  (config `channel_rho`).
  Transition order per step: the agent observes h_t, selects a_t, the reward is
  computed on h_t (including reconfiguration switching costs relative to the
  previously applied action), and only THEN the small-scale channel evolves to
  h_{t+1}; next_obs contains h_{t+1}. Geometry and large-scale fading stay
  fixed within one episode (resampled at reset when
  `resample_positions_on_reset` is true).
- "static_block": legacy v14 behaviour kept for regression against the golden
  fixture: the channel is sampled at reset() and only refreshed every
  `channel_block_steps` steps (before the action is applied). Not a dynamic
  MDP; retained only for reproducing pre-refactor results.
- "contextual_bandit": max_steps forced to 1; a fresh scenario per reset. The
  training configuration must use gamma = 0 (validated by the training driver).

RSMA MISO
---------
Transmit signal:
    x_BS = w_c s_c + sum_k w_k s_k,   sum ||w||^2 <= P_max (joint projection).
SINR (common, decoded first by every user, treat private as noise):
    gamma_c,k = |h_k^H w_c|^2 / (sum_j |h_k^H w_j|^2 + sigma^2)
Common rate: R_c = min_k log2(1 + gamma_c,k); user share c_k, sum_k c_k = 1.
Private (after SIC of common):
    gamma_k = |h_k^H w_k|^2 / (sum_{j!=k} |h_k^H w_j|^2 + sigma^2)
Per-user rate: R_k = c_k * R_c + log2(1 + gamma_k);  R_sum = R_c + sum_k log2(1+gamma_k).

Constrained-RL reward (penalty/Lagrangian surrogate of P0)
----------------------------------------------------------
With the signed per-user constraint function c_k = R_min - R_k (expected-rate
constraint E[c_k] <= 0, i.e. E[R_k] >= R_min):

    r_t = alpha * R_sum / R_ref
          - sum_k lambda_k * c_k
          - 0.5 * w * sum_k max(c_k, 0)^2
          - eta_phi * C_phase - eta_P * C_power - eta_beta * C_beta

where lambda_k >= 0 are per-user dual variables updated by the training driver
(projected dual gradient on the SAME signed c_k), w = `augmented_penalty_weight`
is the augmented quadratic penalty weight, and C_* are dimension-normalised
switching costs (zero at t = 0 by definition). This reward does NOT guarantee
per-step per-user QoS; it is a surrogate of the hard-constrained problem P0.

QoS metrics reported in `info` (explicit names, no ambiguous aliases):
    user_qos_fraction        = (1/K) sum_k 1[R_k >= R_min]
    all_users_qos_satisfied  = 1[min_k R_k >= R_min]
    min_user_rate, mean_qos_deficit, max_qos_deficit,
    per_user_qos_satisfied, per_user_rate

Action mapping (per agent, networks output in [-1, 1])
------------------------------------------------------
Agent 0 (structured BS controller, default): size = 3K + 2
    - K+1 bounded logits for common/private stream powers
    - K bounded logits for the common-rate split c_k
    - K bounded logits for the common-beam user weights
    - 1 bounded residual controlling the MRT/RZF private-direction mixture
  Beam directions are generated from the effective channel after applying the
  current STAR-RIS action. This keeps the policy close to a strong physical
  baseline and removes raw complex beamformer entries from the action space.
  Legacy `raw_complex` mode remains available only for reproducing old results.
Agent 1 (STAR-RIS reflection):    size = 2N   -- ORDER: [beta_r (N), phi_r (N)]
Agent 2 (STAR-RIS transmission):  size = N    -- [phi_t (N)]
Use `action_schema()` / `observation_schema()` for the authoritative layout.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
import warnings
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from utils.metrics import dbm_to_watt, safe_log2, free_space_path_loss_db


VALID_FORMULATIONS = ("dynamic_mdp", "static_block", "contextual_bandit")


@dataclass
class EnvSpec:
    obs_dims: list[int]
    act_dims: list[int]
    global_state_dim: int
    n_agents: int


@dataclass
class SchemaField:
    """One contiguous slice of an observation or action vector."""
    name: str
    start: int
    stop: int
    description: str

    def to_dict(self) -> dict:
        return {"name": self.name, "start": self.start, "stop": self.stop,
                "dim": self.stop - self.start, "description": self.description}


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), 1e-12)


class StarRisRsmaEnv(gym.Env):
    """Gymnasium-compatible multi-agent friendly env for STAR-RIS RSMA MISO."""
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, cfg: dict, seed: int | None = None,
                 ris_mode: str = "optimized"):
        """
        ris_mode:
          - "optimized":  RIS phases/amplitudes set by agent action.
          - "fixed":      RIS phases all 0, amplitudes 50/50.
          - "random":     RIS phases & amplitudes drawn uniformly at random each step.
          - "none":       RIS contribution disabled (effective channel = h_dk only).
          - "analytical": closed-form single-user constructive alignment heuristic.
          - "ao_grid":    coarse alternating-optimization grid heuristic; it is
                          NOT an upper bound (see _coarse_ao_grid).
                          "bcd" is a deprecated alias.
        """
        super().__init__()
        self.cfg = cfg
        if ris_mode == "bcd":
            warnings.warn(
                "ris_mode='bcd' is deprecated; the baseline is a coarse AO-grid "
                "heuristic, not BCD and not an upper bound. Use ris_mode='ao_grid'.",
                DeprecationWarning, stacklevel=2)
            ris_mode = "ao_grid"
        self.ris_mode = ris_mode

        # ----------- Formulation -----------
        self.formulation = str(cfg.get("env_formulation", "dynamic_mdp")).lower()
        if self.formulation not in VALID_FORMULATIONS:
            raise ValueError(
                f"env_formulation must be one of {VALID_FORMULATIONS}, "
                f"got {self.formulation!r}")
        self.channel_rho = float(cfg.get("channel_rho", 0.95))
        if not (0.0 <= self.channel_rho <= 1.0):
            raise ValueError(f"channel_rho must be in [0, 1], got {self.channel_rho}")

        # ----------- Topology -----------
        self.M = int(cfg["num_bs_antennas"])
        if self.M < 2:
            raise ValueError("MISO formulation requires num_bs_antennas >= 2")
        self.K = int(cfg["num_users"])
        self.K_r = int(cfg["num_users_reflection"])
        assert 0 <= self.K_r <= self.K
        self.K_t = self.K - self.K_r
        self.N = int(cfg["num_ris_elements"])
        self.star_mode = cfg.get("star_mode", "ES")
        assert self.star_mode == "ES", "Only ES mode implemented."
        # Documentation label only: ideal independent-phase ES STAR-RIS
        # (continuous phases, no coupling, no insertion loss, perfect CSI).
        self.hardware_model = str(cfg.get("star_ris_hardware_model",
                                          "ideal_independent_phase_es"))

        # ----------- Power / noise -----------
        self.p_max = float(dbm_to_watt(cfg["p_max_dbm"]))
        self.sigma2 = float(dbm_to_watt(cfg["noise_power_dbm"]))
        self.qos_min = float(cfg["qos_rate_min"])

        # ----------- Path-loss -----------
        self.pl_exp_d = float(cfg["path_loss_exp_direct"])
        self.pl_exp_br = float(cfg["path_loss_exp_bs_ris"])
        self.pl_exp_ru = float(cfg["path_loss_exp_ris_user"])
        self.ref_pl_db = float(cfg["ref_path_loss_db"])
        self.ref_d = float(cfg["ref_distance"])

        # ----------- Geometry -----------
        self.bs_pos = np.array(cfg["bs_position"], dtype=np.float64)
        self.ris_pos = np.array(cfg["ris_position"], dtype=np.float64)
        self.area_r = np.array(cfg["user_area_reflection"], dtype=np.float64)
        self.area_t = np.array(cfg["user_area_transmission"], dtype=np.float64)

        # ----------- Episode -----------
        self.max_steps = int(cfg["max_steps"])
        if self.formulation == "contextual_bandit":
            self.max_steps = 1
        self.channel_block_steps = int(cfg.get("channel_block_steps", 1))
        default_resample = self.formulation != "static_block"
        self.resample_positions = bool(cfg.get("resample_positions_on_reset",
                                               default_resample))

        # ----------- Reward (penalty surrogate of P0) -----------
        self.r_alpha = float(cfg.get("reward_alpha", 1.0))
        # Reference rate for sum-rate normalisation. Default preserves the
        # legacy K*5 normalisation when the key is absent.
        self.r_ref = float(cfg.get("reward_rate_reference", self.K * 5.0))
        self.r_scale = float(cfg.get("reward_scale", 0.1))
        self.r_clip = float(cfg.get("reward_clip", 50.0))
        self.eps = float(cfg.get("epsilon", 1e-12))
        # Per-user dual variables (projected dual gradient updates them via the
        # training driver; the env only evaluates the Lagrangian reward).
        lam0 = float(cfg.get("qos_lambda_init", 1.0))
        self.dual_lambda_max = float(cfg.get("dual_lambda_max", 20.0))
        self.qos_lambda_vec = np.full(self.K, lam0, dtype=np.float64)
        # Augmented quadratic penalty weight w (name chosen to avoid confusion
        # with channel_rho).
        self.augmented_penalty_weight = float(cfg.get("augmented_penalty_weight", 1.0))
        # Optional shaping bonus, default OFF (kept only for ablation).
        self.enable_qos_shaping_bonus = bool(cfg.get("enable_qos_shaping_bonus", False))
        self.r_qos_bonus = float(cfg.get("reward_qos_bonus", 0.5))
        # Switching-cost weights (dimension-normalised costs, zero at t = 0).
        self.eta_phase = float(cfg.get("phase_switching_cost", 0.0))
        self.eta_power = float(cfg.get("power_switching_cost", 0.0))
        self.eta_beta = float(cfg.get("beta_switching_cost", 0.0))

        # ----------- Structured BS action -----------
        # Missing key intentionally defaults to raw_complex so historical
        # static-block golden fixtures remain reproducible. Publication configs
        # explicitly select structured_rzf.
        self.bs_action_mode = str(cfg.get("bs_action_mode", "raw_complex")).lower()
        if self.bs_action_mode not in ("raw_complex", "structured_rzf"):
            raise ValueError(
                "bs_action_mode must be 'raw_complex' or 'structured_rzf', "
                f"got {self.bs_action_mode!r}")
        self.bs_power_logit_scale = float(cfg.get("bs_power_logit_scale", 1.5))
        self.bs_power_action_clip = float(cfg.get("bs_power_action_clip", 0.9))
        self.bs_min_stream_power_fraction = float(
            cfg.get("bs_min_stream_power_fraction", 0.02))
        self.bs_common_beam_logit_scale = float(
            cfg.get("bs_common_beam_logit_scale", 1.0))
        self.bs_common_beam_action_clip = float(
            cfg.get("bs_common_beam_action_clip", 0.9))
        self.bs_rzf_regularization = float(cfg.get("bs_rzf_regularization", 0.05))
        self.bs_rzf_mix_prior = float(cfg.get("bs_rzf_mix_prior", 0.85))
        self.bs_rzf_mix_span = float(cfg.get("bs_rzf_mix_span", 0.15))
        if not (0.0 < self.bs_power_action_clip < 1.0):
            raise ValueError("bs_power_action_clip must be in (0, 1)")
        if not (0.0 <= self.bs_min_stream_power_fraction < 1.0 / (self.K + 1)):
            raise ValueError(
                "bs_min_stream_power_fraction must be in [0, 1/(K+1))")
        if not (0.0 < self.bs_common_beam_action_clip < 1.0):
            raise ValueError("bs_common_beam_action_clip must be in (0, 1)")
        if self.bs_rzf_regularization <= 0:
            raise ValueError("bs_rzf_regularization must be positive")
        if not (0.0 <= self.bs_rzf_mix_prior <= 1.0):
            raise ValueError("bs_rzf_mix_prior must be in [0, 1]")
        if self.bs_rzf_mix_span < 0:
            raise ValueError("bs_rzf_mix_span must be non-negative")

        # Clean one-factor ablations. The old equal_power_mode is retained as a
        # deprecated composite compatibility switch (equal powers + MRT +
        # uniform common split/beam); new studies must use the explicit flags.
        self.force_equal_stream_power = bool(
            cfg.get("force_equal_stream_power", False))
        self.force_mrt_directions = bool(cfg.get("force_mrt_directions", False))
        self.force_uniform_common_split = bool(
            cfg.get("force_uniform_common_split", False))
        self.force_uniform_common_beam = bool(
            cfg.get("force_uniform_common_beam", False))
        self.equal_power_mode = bool(cfg.get("equal_power_mode", False))
        if self.equal_power_mode:
            self.force_equal_stream_power = True
            self.force_mrt_directions = True
            self.force_uniform_common_split = True
            self.force_uniform_common_beam = True

        # Physics-informed phase action: "absolute" or "residual".
        self.phase_action_mode = str(cfg.get("phase_action_mode", "absolute")).lower()
        self.phase_residual_scale = float(cfg.get("phase_residual_scale", 0.3))
        self.common_split_logit_scale = float(cfg.get("common_split_logit_scale", 1.0))
        # Below this direct-link amplitude the analytical prior aligns the
        # cascaded terms to a zero-phase reference instead of the (numerically
        # meaningless) phase of a vanishing h_d.
        self.analytical_min_direct = float(cfg.get("analytical_phase_min_direct", 1e-9))

        # ----------- RNG streams -----------
        # static_block keeps the single legacy stream so the golden fixture
        # (pre-refactor physics) is reproduced bit-exactly. dynamic_mdp and
        # contextual_bandit use separate streams so the channel trajectory is
        # independent of policy/exploration randomness (required for paired
        # comparisons and for the ScenarioBank).
        self._init_seed = seed
        self._setup_rngs(seed)

        # ----------- User positions -----------
        self.user_positions = self._sample_user_positions()
        self._compute_path_losses()

        # ----------- State variables -----------
        self._prev_power_weights = None   # length K+1 (softmax weights)
        self._prev_common_split = None    # length K
        self._prev_reward = 0.0
        self._step_count = 0
        self._h_eff = None
        self._h_ris = None
        # Small-scale (unit-variance) and composed channels.
        self._h_d_small = None            # (K, M)
        self._G_small = None              # (N, M)
        self._g_small = None              # (K, N)
        self._h_d = None
        self._G = None
        self._g = None
        # RIS state currently applied.
        self._beta_r = None               # (N,)
        self._phi_r = None                # (N,)
        self._phi_t = None                # (N,)
        # Previously APPLIED physical action (switching-cost reference).
        # None at t = 0 -> all switching costs are zero by definition.
        self._prev_applied = None
        # Scenario playback (set via reset(options={"scenario": ...})).
        self._scenario = None

        # ----------- Spaces -----------
        self.n_agents = 3
        if self.bs_action_mode == "structured_rzf":
            # [power logits (K+1), common-split logits (K), common-beam
            # weights (K), private MRT/RZF mix residual (1)].
            bs_act_dim = 3 * self.K + 2
        else:
            bs_act_dim = 2 * self.M * (self.K + 1) + self.K
        self.act_dims = [
            bs_act_dim,
            2 * self.N,              # RIS reflection: [beta_r, phi_r]
            self.N,                  # RIS transmission: [phi_t]
        ]

        self.obs_include_channel = bool(cfg.get("obs_include_channel_state", True))
        # Include the currently applied RIS state in observations. Needed for
        # the Markov property under dynamic_mdp with switching costs.
        self.obs_include_ris_state = bool(cfg.get("obs_include_ris_state", True))
        self.local_obs = bool(cfg.get("local_obs_for_maddpg", True))

        self._build_schemas()
        self.obs_dims = [sch[-1].stop for sch in self._obs_schema]
        self.obs_dim_per_agent = max(self.obs_dims)
        self.single_agent_obs_dim = self._single_schema[-1].stop
        # Canonical centralized-critic state: every physical feature appears
        # exactly once. Concatenating local observations duplicates the shared
        # h_eff/base block and the BS->RIS channel G.
        self.global_state_dim = int(self.single_agent_obs_dim)

        self.act_dim_flat = int(sum(self.act_dims))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.single_agent_obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.act_dim_flat,), dtype=np.float32,
        )
        self.agent_action_spaces = [
            spaces.Box(low=-1.0, high=1.0, shape=(d,), dtype=np.float32) for d in self.act_dims
        ]
        self.agent_observation_spaces = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(d,), dtype=np.float32) for d in self.obs_dims
        ]

    # ------------------------------------------------------------------ RNG
    def _setup_rngs(self, seed: int | None):
        if self.formulation == "static_block":
            # Legacy single-stream path (geometry, channels and random-RIS all
            # share self.rng, in the original draw order).
            self.rng = np.random.default_rng(seed)
            self.geometry_rng = self.rng
            self.channel_rng = self.rng
            self.misc_rng = self.rng
        else:
            ss = np.random.SeedSequence(seed)
            geo_ss, chan_ss, misc_ss = ss.spawn(3)
            self.geometry_rng = np.random.default_rng(geo_ss)
            self.channel_rng = np.random.default_rng(chan_ss)
            self.misc_rng = np.random.default_rng(misc_ss)   # random-RIS mode only
            self.rng = self.misc_rng  # back-compat alias

    # ------------------------------------------------------------------ utils
    def spec(self) -> EnvSpec:
        return EnvSpec(
            obs_dims=list(self.obs_dims),
            act_dims=list(self.act_dims),
            global_state_dim=self.global_state_dim,
            n_agents=self.n_agents,
        )

    def seed(self, seed: int | None = None):
        self._setup_rngs(seed)
        return [seed]

    # ---------------------------------------------------------- geometry
    def _sample_user_positions(self) -> np.ndarray:
        positions = np.zeros((self.K, 3), dtype=np.float64)
        for k in range(self.K_r):
            positions[k] = self._sample_in_box(self.area_r)
        for k in range(self.K_r, self.K):
            positions[k] = self._sample_in_box(self.area_t)
        return positions

    def _sample_in_box(self, box: np.ndarray) -> np.ndarray:
        rng = self.geometry_rng
        return np.array([
            rng.uniform(box[0, 0], box[0, 1]),
            rng.uniform(box[1, 0], box[1, 1]),
            rng.uniform(box[2, 0], box[2, 1]),
        ])

    def _compute_path_losses(self):
        d_bs_user = np.linalg.norm(self.user_positions - self.bs_pos[None, :], axis=1)
        d_bs_ris = float(np.linalg.norm(self.ris_pos - self.bs_pos))
        d_ris_user = np.linalg.norm(self.user_positions - self.ris_pos[None, :], axis=1)

        pl_d_db = free_space_path_loss_db(d_bs_user, self.ref_pl_db, self.ref_d, self.pl_exp_d)
        pl_br_db = free_space_path_loss_db(np.array([d_bs_ris]), self.ref_pl_db, self.ref_d, self.pl_exp_br)[0]
        pl_ru_db = free_space_path_loss_db(d_ris_user, self.ref_pl_db, self.ref_d, self.pl_exp_ru)

        block_db = float(self.cfg.get("direct_block_loss_db", 0.0))
        if bool(self.cfg.get("direct_block_T", False)) and self.K_t > 0 and block_db > 0:
            pl_d_db[self.K_r:] = pl_d_db[self.K_r:] + block_db
        block_r_db = float(self.cfg.get("direct_block_R_loss_db", 0.0))
        if block_r_db > 0 and self.K_r > 0:
            pl_d_db[: self.K_r] = pl_d_db[: self.K_r] + block_r_db

        self.alpha_d = 10.0 ** (-pl_d_db / 20.0)           # (K,)
        self.alpha_br = 10.0 ** (-pl_br_db / 20.0)         # scalar
        self.alpha_ru = 10.0 ** (-pl_ru_db / 20.0)         # (K,)

    # ---------------------------------------------------------- channels
    def _cn(self, *shape):
        """CN(0, 1) samples from the dedicated channel RNG."""
        rng = self.channel_rng
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2.0)

    def _sample_channels(self):
        """Sample fresh unit-variance Rayleigh small-scale channels."""
        self._h_d_small = self._cn(self.K, self.M)
        self._G_small = self._cn(self.N, self.M)
        self._g_small = self._cn(self.K, self.N)
        self._apply_large_scale()

    def _apply_large_scale(self):
        self._h_d = (self.alpha_d[:, None] * self._h_d_small).astype(np.complex128)
        self._G = (self.alpha_br * self._G_small).astype(np.complex128)
        self._g = (self.alpha_ru[:, None] * self._g_small).astype(np.complex128)

    def _evolve_channels(self):
        """Gauss-Markov evolution of the SMALL-SCALE channels (dynamic_mdp).

        h_{t+1} = rho * h_t + sqrt(1 - rho^2) * eps_t,  eps_t ~ CN(0, 1).
        Innovations come from the scenario playback when available (so every
        method sees the identical channel trajectory), otherwise from the
        dedicated channel RNG.
        """
        rho = self.channel_rho
        scale = math.sqrt(max(0.0, 1.0 - rho * rho))
        if self._scenario is not None:
            t = self._step_count  # innovation index for the transition t -> t+1
            eps_hd = self._scenario["innov_h_d"][t]
            eps_G = self._scenario["innov_G"][t]
            eps_g = self._scenario["innov_g"][t]
        else:
            eps_hd = self._cn(self.K, self.M)
            eps_G = self._cn(self.N, self.M)
            eps_g = self._cn(self.K, self.N)
        self._h_d_small = rho * self._h_d_small + scale * eps_hd
        self._G_small = rho * self._G_small + scale * eps_G
        self._g_small = rho * self._g_small + scale * eps_g
        self._apply_large_scale()

    # ---------------------------------------------------------- analytical phases
    def _analytical_phases(self) -> tuple[np.ndarray, np.ndarray]:
        """Constructive-alignment prior for the weakest R/T reference user.

        Under the stored-channel convention,
            h_eff^H q = h_d^H q + g^H Phi G q.
        For an equal-gain BS direction q, element n contributes
            exp(j phi_n) * conj(g_n) * (G_n q).
        The phase below aligns every nonzero cascaded contribution with the
        direct received scalar h_d^H q. This is a single-user heuristic prior,
        not an upper bound and not a multi-user optimum.
        """
        q = np.ones(self.M, dtype=np.complex128) / math.sqrt(self.M)

        def _align(k: int) -> np.ndarray:
            direct_signal = np.vdot(self._h_d[k], q)
            cascade_signal = np.conj(self._g[k]) * (self._G @ q)
            cascade_phase = np.angle(cascade_signal)
            if np.abs(direct_signal) < self.analytical_min_direct:
                return np.mod(-cascade_phase, 2 * math.pi)
            return np.mod(np.angle(direct_signal) - cascade_phase,
                          2 * math.pi)

        if self.K_r > 0:
            k_R = int(np.argmin(np.linalg.norm(self._h_d[: self.K_r], axis=1)))
            phi_r = _align(k_R)
        else:
            phi_r = np.zeros(self.N)
        if self.K_t > 0:
            k_T = self.K_r + int(np.argmin(np.linalg.norm(self._h_d[self.K_r:], axis=1)))
            phi_t = _align(k_T)
        else:
            phi_t = np.zeros(self.N)
        return phi_r, phi_t

    # ---------------------------------------------------------- coarse AO-grid heuristic
    def _coarse_ao_grid(self, n_iter: int = 20) -> dict:
        """Coarse alternating-optimization GRID heuristic. NOT an upper bound.

        Limitations (kept intentionally; see experiments/baselines_ao.py for a
        stronger AO local-search reference):
          - phases: single-user closed-form alignment (reused every iteration);
          - beta_r: ONE shared value for all N elements, 5-point grid;
          - P_c fraction: 6-point grid, private powers split equally;
          - common split fixed uniform; objective is sum-rate only (no QoS).

        Returns the decoded dict just like the RL action decoder, plus
        diagnostics: number of RSMA evaluations and the per-iteration best
        objective (non-decreasing by construction of the greedy accept).
        """
        K = self.K
        N = self.N
        beta_r = 0.5 * np.ones(N)
        P_c = float(self.p_max * 0.5)
        P_k = np.full(K, (self.p_max - P_c) / max(K, 1), dtype=np.float64)
        common_split = np.ones(K, dtype=np.float64) / K
        phi_r = np.zeros(N); phi_t = np.zeros(N)

        beta_grid = np.array([0.2, 0.3, 0.5, 0.7, 0.8])
        pc_grid = np.array([0.1, 0.3, 0.5, 0.7, 0.85, 0.95])

        n_evals = 0
        objective_trace: list[float] = []
        for _ in range(int(n_iter)):
            phi_r, phi_t = self._analytical_phases()

            best_beta = float(beta_r[0]); best_sr = -np.inf
            for b in beta_grid:
                b_arr = float(b) * np.ones(N)
                h = self._effective_channels(b_arr, phi_r, phi_t)
                Wc_t, Wk_t, _ = self._physics_beamformers(
                    h, P_c, P_k, rzf_mix=self.bs_rzf_mix_prior)
                rs = self._rsma_rates(
                    h, P_c, P_k, common_split, Wc_t, Wk_t)
                n_evals += 1
                if rs["sum_rate"] > best_sr:
                    best_sr = rs["sum_rate"]; best_beta = float(b)
            beta_r = best_beta * np.ones(N)

            h_eff = self._effective_channels(beta_r, phi_r, phi_t)
            best_pc = P_c; best_sr = -np.inf
            for f in pc_grid:
                Pc_t = float(self.p_max * f)
                Pk_t = np.full(K, (self.p_max - Pc_t) / max(K, 1), dtype=np.float64)
                Wc_t, Wk_t, _ = self._physics_beamformers(
                    h_eff, Pc_t, Pk_t, rzf_mix=self.bs_rzf_mix_prior)
                rs = self._rsma_rates(
                    h_eff, Pc_t, Pk_t, common_split, Wc_t, Wk_t)
                n_evals += 1
                if rs["sum_rate"] > best_sr:
                    best_sr = rs["sum_rate"]; best_pc = Pc_t
            P_c = best_pc
            P_k = np.full(K, (self.p_max - P_c) / max(K, 1), dtype=np.float64)
            objective_trace.append(float(best_sr))

        powers = np.concatenate([[P_c], P_k])
        power_weights = powers / max(self.p_max, 1e-12)
        h_eff = self._effective_channels(beta_r, phi_r, phi_t)
        W_c, W_k, mix = self._physics_beamformers(
            h_eff, P_c, P_k, rzf_mix=self.bs_rzf_mix_prior)
        return {
            "P_c": P_c, "P_k": P_k, "W_c": W_c, "W_k": W_k,
            "power_weights": power_weights, "common_split": common_split,
            "beta_r": beta_r, "phi_r": phi_r, "phi_t": phi_t,
            "bs_private_rzf_mix": mix,
            "bs_action_mode": "classical_rzf_grid",
            "ao_grid_n_evals": n_evals,
            "ao_grid_objective_trace": objective_trace,
        }

    # ---------------------------------------------------------- structured BS
    @staticmethod
    def _normalise_rows(x: np.ndarray, eps: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.complex128)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)

    def _mrt_directions(self, h_eff: np.ndarray) -> np.ndarray:
        """Unit-norm matched-filter directions for h_k^H w_k."""
        return self._normalise_rows(h_eff, self.eps)

    def _rzf_directions(self, h_eff: np.ndarray) -> np.ndarray:
        """Stable unit-norm RZF directions under the stored-channel convention.

        The received channel matrix is B=conj(H), because the scalar link is
        h_k^H w. Row-normalising B removes path-loss scale from the condition
        number; the actor controls only a bounded MRT/RZF mixture, not raw
        complex coefficients.
        """
        h = np.asarray(h_eff, dtype=np.complex128).reshape(self.K, self.M)
        b = np.conj(self._normalise_rows(h, self.eps))
        gram = b @ np.conj(b).T
        reg = self.bs_rzf_regularization * np.eye(self.K, dtype=np.complex128)
        try:
            inv = np.linalg.solve(gram + reg, np.eye(self.K, dtype=np.complex128))
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(gram + reg, rcond=1e-10)
        f = np.conj(b).T @ inv             # (M,K), columns are user beams
        dirs = self._normalise_rows(f.T, self.eps)
        # Align global phases with MRT before interpolation; otherwise two
        # physically equivalent directions can cancel numerically when blended.
        mrt = self._mrt_directions(h)
        for k in range(self.K):
            inner = np.vdot(mrt[k], dirs[k])
            if abs(inner) > self.eps:
                dirs[k] *= np.exp(-1j * np.angle(inner))
        return dirs

    def _physics_beamformers(self, h_eff: np.ndarray, P_c: float,
                              P_k: np.ndarray,
                              common_beam_weights: np.ndarray | None = None,
                              rzf_mix: float | None = None
                              ) -> tuple[np.ndarray, np.ndarray, float]:
        """Build common/private beams from a shared MRT/RZF physics layer.

        Learned policies and classical baselines call this same routine so a
        reported gain cannot come from silently giving one method a stronger
        beamformer implementation. Only powers, common-beam weights and the
        bounded MRT/RZF mixing coefficient differ between methods.
        """
        h = np.asarray(h_eff, dtype=np.complex128).reshape(self.K, self.M)
        p_k = np.asarray(P_k, dtype=np.float64).reshape(self.K)
        if common_beam_weights is None:
            common_beam_weights = np.ones(self.K, dtype=np.float64) / self.K
        else:
            common_beam_weights = np.asarray(
                common_beam_weights, dtype=np.float64).reshape(self.K)
            common_beam_weights = np.maximum(common_beam_weights, 0.0)
            total = float(common_beam_weights.sum())
            common_beam_weights = (
                common_beam_weights / total if total > self.eps
                else np.ones(self.K, dtype=np.float64) / self.K)

        mrt = self._mrt_directions(h)
        rzf = self._rzf_directions(h)
        mix = self.bs_rzf_mix_prior if rzf_mix is None else float(rzf_mix)
        mix = float(np.clip(mix, 0.0, 1.0))
        private_dirs = self._normalise_rows(
            (1.0 - mix) * mrt + mix * rzf, self.eps)
        W_k = np.sqrt(np.maximum(p_k, 0.0))[:, None] * private_dirs

        common_dir = np.sum(common_beam_weights[:, None] * mrt, axis=0)
        if np.linalg.norm(common_dir) < self.eps:
            common_dir = np.sum(mrt, axis=0)
        common_dir = common_dir / max(np.linalg.norm(common_dir), self.eps)
        W_c = math.sqrt(max(float(P_c), 0.0)) * common_dir
        return W_c, W_k, mix

    def _structured_bs_decision(self, a_bs: np.ndarray,
                                h_eff: np.ndarray) -> dict:
        """Decode a compact, physics-structured BS action.

        Zero action maps to equal stream power, uniform common split/beam
        weights, and the configured RZF-prior mixture. This gives every RL
        method a strong, reproducible starting point instead of an arbitrary
        raw complex beamformer.
        """
        a = np.clip(np.asarray(a_bs, dtype=np.float64), -1.0, 1.0)
        pos = 0
        p_raw = a[pos:pos + self.K + 1]; pos += self.K + 1
        c_raw = a[pos:pos + self.K]; pos += self.K
        cb_raw = a[pos:pos + self.K]; pos += self.K
        mix_raw = float(a[pos]); pos += 1
        assert pos == a.size == 3 * self.K + 2

        p_logits = self.bs_power_logit_scale * np.arctanh(
            np.clip(p_raw, -self.bs_power_action_clip,
                    self.bs_power_action_clip))
        soft_power = _softmax(p_logits)
        floor = self.bs_min_stream_power_fraction
        power_weights = floor + (1.0 - (self.K + 1) * floor) * soft_power
        if self.force_equal_stream_power:
            power_weights = np.ones(self.K + 1, dtype=np.float64) / (self.K + 1)
        powers = self.p_max * power_weights
        P_c = float(powers[0])
        P_k = powers[1:].astype(np.float64)

        c_logits = self.common_split_logit_scale * np.arctanh(
            np.clip(c_raw, -1.0 + 1e-6, 1.0 - 1e-6))
        common_split = _softmax(c_logits)
        if self.force_uniform_common_split:
            common_split = np.ones(self.K, dtype=np.float64) / self.K

        cb_logits = self.bs_common_beam_logit_scale * np.arctanh(
            np.clip(cb_raw, -self.bs_common_beam_action_clip,
                    self.bs_common_beam_action_clip))
        common_beam_weights = _softmax(cb_logits)
        if self.force_uniform_common_beam:
            common_beam_weights = np.ones(self.K, dtype=np.float64) / self.K

        mix = float(np.clip(self.bs_rzf_mix_prior
                            + self.bs_rzf_mix_span * mix_raw, 0.0, 1.0))
        if self.force_mrt_directions:
            mix = 0.0
        W_c, W_k, mix = self._physics_beamformers(
            h_eff, P_c, P_k, common_beam_weights, rzf_mix=mix)

        return {
            "P_c": P_c, "P_k": P_k, "W_c": W_c, "W_k": W_k,
            "power_weights": power_weights,
            "common_split": common_split,
            "common_beam_weights": common_beam_weights,
            "bs_private_rzf_mix": mix,
            "bs_action_mode": "structured_rzf",
        }

    def _raw_complex_bs_decision(self, a_bs: np.ndarray,
                                 h_eff: np.ndarray) -> dict:
        """Legacy raw complex beamformer decoder for old-result reproduction."""
        a_bs = np.clip(a_bs, -1.0, 1.0)
        n_streams = self.K + 1
        n_bf = 2 * self.M * n_streams
        bf_raw = a_bs[:n_bf].reshape(n_streams, 2, self.M)
        W = (bf_raw[:, 0, :].astype(np.float64)
             + 1j * bf_raw[:, 1, :].astype(np.float64))
        norm2 = float(np.sum(np.abs(W) ** 2))
        if norm2 < self.eps:
            W = np.ones((n_streams, self.M), dtype=np.complex128)
            norm2 = float(np.sum(np.abs(W) ** 2))
        W *= math.sqrt(self.p_max / max(norm2, self.eps))
        W_c, W_k = W[0], W[1:]
        powers = np.sum(np.abs(W) ** 2, axis=1).astype(np.float64)
        power_weights = powers / max(self.p_max, self.eps)
        P_c, P_k = float(powers[0]), powers[1:]
        cs_logits = self.common_split_logit_scale * np.arctanh(
            np.clip(a_bs[n_bf:], -1.0 + 1e-6, 1.0 - 1e-6))
        common_split = _softmax(cs_logits)

        # Deprecated composite compatibility mode.
        if self.equal_power_mode:
            power_weights = np.ones(self.K + 1, dtype=np.float64) / (self.K + 1)
            common_split = np.ones(self.K, dtype=np.float64) / self.K
            P_c = float(self.p_max / (self.K + 1))
            P_k = np.full(self.K, self.p_max / (self.K + 1), dtype=np.float64)
            dirs = self._mrt_directions(h_eff)
            W_k = np.sqrt(P_k)[:, None] * dirs
            common_dir = np.sum(dirs, axis=0)
            common_dir /= max(np.linalg.norm(common_dir), self.eps)
            W_c = math.sqrt(P_c) * common_dir
        return {
            "P_c": P_c, "P_k": P_k, "W_c": W_c, "W_k": W_k,
            "power_weights": power_weights, "common_split": common_split,
            "bs_private_rzf_mix": float("nan"),
            "bs_action_mode": "raw_complex",
        }

    # ---------------------------------------------------------- action decoding
    def _decode_action(self, action_list: list[np.ndarray]):
        """Map normalized [-1,1] actions -> physical decision variables.

        RIS variables are decoded first. The structured BS backbone then uses
        the effective channel induced by the *current* RIS action, removing the
        one-step stale-channel mismatch present in the old equal-power path.
        """
        a_bs, a_ris_r, a_ris_t = action_list

        # ORDER: [beta_r (N), phi_r (N)] -- see action_schema().
        a_ris_r = np.clip(a_ris_r, -1.0, 1.0)
        beta_logits = a_ris_r[: self.N]
        phi_r_raw = a_ris_r[self.N:]
        beta_r = np.clip(0.5 * (beta_logits + 1.0), 1e-4, 1.0 - 1e-4)
        a_ris_t = np.clip(a_ris_t, -1.0, 1.0)

        if self.phase_action_mode == "residual" and self.phase_residual_scale > 0:
            prior_phi_r, prior_phi_t = self._analytical_phases()
            phi_r = np.mod(prior_phi_r
                           + self.phase_residual_scale * math.pi * phi_r_raw,
                           2 * math.pi)
            phi_t = np.mod(prior_phi_t
                           + self.phase_residual_scale * math.pi * a_ris_t,
                           2 * math.pi)
        else:
            phi_r = math.pi * (phi_r_raw + 1.0)
            phi_t = math.pi * (a_ris_t + 1.0)

        if self.ris_mode == "ao_grid":
            return self._coarse_ao_grid()
        if self.ris_mode == "fixed":
            beta_r = 0.5 * np.ones(self.N)
            phi_r = np.zeros(self.N)
            phi_t = np.zeros(self.N)
        elif self.ris_mode == "random":
            beta_r = self.misc_rng.uniform(0.1, 0.9, size=self.N)
            phi_r = self.misc_rng.uniform(0.0, 2 * math.pi, size=self.N)
            phi_t = self.misc_rng.uniform(0.0, 2 * math.pi, size=self.N)
        elif self.ris_mode == "none":
            beta_r = 0.5 * np.ones(self.N)
            phi_r = np.zeros(self.N)
            phi_t = np.zeros(self.N)
        elif self.ris_mode == "analytical":
            beta_r = 0.5 * np.ones(self.N)
            phi_r, phi_t = self._analytical_phases()

        h_candidate = self._effective_channels(beta_r, phi_r, phi_t)
        if self.bs_action_mode == "structured_rzf":
            bs = self._structured_bs_decision(a_bs, h_candidate)
        else:
            bs = self._raw_complex_bs_decision(a_bs, h_candidate)
        return {**bs, "beta_r": beta_r, "phi_r": phi_r, "phi_t": phi_t}

    # ---------------------------------------------------------- RSMA
    def _effective_channels(self, beta_r: np.ndarray, phi_r: np.ndarray, phi_t: np.ndarray) -> np.ndarray:
        """Return effective MISO channels H_eff with shape (K, M)."""
        beta_t = np.clip(1.0 - beta_r, 1e-4, 1.0 - 1e-4)
        coeff_r = np.sqrt(beta_r) * np.exp(1j * phi_r)
        coeff_t = np.sqrt(beta_t) * np.exp(1j * phi_t)

        h_eff = np.zeros((self.K, self.M), dtype=np.complex128)
        h_ris = np.zeros((self.K, self.M), dtype=np.complex128)
        for k in range(self.K):
            if self.ris_mode == "none":
                cascaded = np.zeros(self.M, dtype=np.complex128)
            else:
                coeff = coeff_r if k < self.K_r else coeff_t
                cascaded = np.sum(
                    np.conj(self._G) * (np.conj(coeff) * self._g[k])[:, None],
                    axis=0,
                )
            h_ris[k] = cascaded
            h_eff[k] = self._h_d[k] + cascaded
        self._h_ris = h_ris
        return h_eff

    @staticmethod
    def _phase_entropy(phi: np.ndarray, n_bins: int = 16) -> float:
        """Shannon entropy of phase distribution (nats), 0..log(n_bins)."""
        if phi.size == 0:
            return 0.0
        hist, _ = np.histogram(np.mod(phi, 2 * math.pi), bins=n_bins, range=(0.0, 2 * math.pi))
        p = hist.astype(np.float64)
        s = p.sum()
        if s <= 0:
            return 0.0
        p = p / s
        nz = p[p > 0]
        return float(-(nz * np.log(nz)).sum())

    def _rsma_rates(self, h_eff: np.ndarray, P_c: float, P_k: np.ndarray,
                    common_split: np.ndarray, W_c: np.ndarray | None = None,
                    W_k: np.ndarray | None = None):
        """MISO RSMA rates using |h_k^H w_j|^2 interference coupling."""
        H = np.asarray(h_eff, dtype=np.complex128).reshape(self.K, self.M)
        if W_c is None or W_k is None:
            # Deterministic matched-filter fallback for classical baselines that
            # still specify stream powers rather than explicit beamformers.
            dirs = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), self.eps)
            W_k = np.sqrt(np.maximum(P_k, 0.0))[:, None] * dirs
            common_dir = np.sum(dirs, axis=0)
            common_dir /= max(np.linalg.norm(common_dir), self.eps)
            W_c = math.sqrt(max(P_c, 0.0)) * common_dir
        W_c = np.asarray(W_c, dtype=np.complex128).reshape(self.M)
        W_k = np.asarray(W_k, dtype=np.complex128).reshape(self.K, self.M)

        common_gain = np.abs(np.einsum("km,m->k", np.conj(H), W_c)) ** 2
        private_gain = np.abs(np.einsum("km,jm->kj", np.conj(H), W_k)) ** 2
        total_private_interf = np.sum(private_gain, axis=1)
        sinr_c = common_gain / (total_private_interf + self.sigma2 + self.eps)
        rate_c = float(np.min(safe_log2(1.0 + sinr_c, self.eps)))

        desired = np.diag(private_gain)
        interference = total_private_interf - desired
        sinr_p = desired / (interference + self.sigma2 + self.eps)
        rates_p = safe_log2(1.0 + sinr_p, self.eps).astype(np.float64)
        per_user = common_split * rate_c + rates_p
        sum_rate = float(rate_c + rates_p.sum())
        h2 = np.sum(np.abs(H) ** 2, axis=1).astype(np.float64)
        return {
            "rate_c": rate_c, "rate_p": rates_p, "per_user": per_user,
            "sum_rate": sum_rate, "h2": h2, "sinr_c": sinr_c,
            "sinr_p": sinr_p, "common_gain": common_gain,
            "private_gain": private_gain,
        }

    # ---------------------------------------------------------- switching costs
    def _switching_costs(self, decoded: dict) -> dict:
        """Dimension-normalised reconfiguration costs vs the previously APPLIED
        physical action. All zero at t = 0 by definition (no previous action)."""
        if self._prev_applied is None:
            return {"phase_cost": 0.0, "power_cost": 0.0, "beta_cost": 0.0}
        prev = self._prev_applied
        dphi = np.concatenate([
            decoded["phi_r"] - prev["phi_r"],
            decoded["phi_t"] - prev["phi_t"],
        ])
        phase_cost = float(np.mean(1.0 - np.cos(dphi))) if dphi.size else 0.0
        power_cost = float(np.abs(decoded["power_weights"] - prev["power_weights"]).sum()
                           / (self.K + 1))
        beta_cost = float(np.mean(np.abs(decoded["beta_r"] - prev["beta_r"])))
        return {"phase_cost": phase_cost, "power_cost": power_cost, "beta_cost": beta_cost}

    # ---------------------------------------------------------- schemas
    def _build_schemas(self):
        """Single source of truth for observation/action layouts."""
        K, K_r, K_t, N, M = self.K, self.K_r, self.K_t, self.N, self.M

        def build(fields_spec: list[tuple[str, int, str]]) -> list[SchemaField]:
            out, pos = [], 0
            for name, dim, desc in fields_spec:
                if dim <= 0:
                    continue
                out.append(SchemaField(name, pos, pos + dim, desc))
                pos += dim
            return out

        base = [
            ("h_eff_re", K * M, "Re(H_eff) flattened over users and BS antennas"),
            ("h_eff_im", K * M, "Im(h_eff) per user, scaled by mean R-region direct gain"),
            ("prev_power_weights", K + 1, "previous softmax power weights [common, p_1..p_K]"),
            ("prev_common_split", K, "previous common-rate split c_k"),
        ]
        chan_all = [
            ("h_d_re", K * M, "Re(h_d) all users, normalised by alpha_d"),
            ("h_d_im", K * M, "Im(h_d) all users, normalised by alpha_d"),
            ("G_re", N * M, "Re(G) BS->RIS, normalised by alpha_br"),
            ("G_im", N * M, "Im(G) BS->RIS, normalised by alpha_br"),
            ("g_re", K * N, "Re(g_k) RIS->user flattened, normalised by alpha_ru"),
            ("g_im", K * N, "Im(g_k) RIS->user flattened, normalised by alpha_ru"),
        ]
        ris_state_r = [
            ("ris_beta_r", N, "currently applied beta_r, mapped to [-1, 1]"),
            ("ris_phi_r", N, "currently applied phi_r / pi - 1, in [-1, 1]"),
        ]
        ris_state_t = [
            ("ris_phi_t", N, "currently applied phi_t / pi - 1, in [-1, 1]"),
        ]

        if self.obs_include_channel:
            bs_extra = [
                ("h_d_re", K * M, "Re(h_d) all users, normalised by alpha_d"),
                ("h_d_im", K * M, "Im(h_d) all users, normalised by alpha_d"),
            ]
            r_extra = [
                ("h_d_R_re", K_r * M, "Re(h_d) R-region users"),
                ("h_d_R_im", K_r * M, "Im(h_d) R-region users"),
                ("G_re", N * M, "Re(G)"), ("G_im", N * M, "Im(G)"),
                ("g_R_re", K_r * N, "Re(g_k) R-region flattened"),
                ("g_R_im", K_r * N, "Im(g_k) R-region flattened"),
            ]
            t_extra = [
                ("h_d_T_re", K_t * M, "Re(h_d) T-region users"),
                ("h_d_T_im", K_t * M, "Im(h_d) T-region users"),
                ("G_re", N * M, "Re(G)"), ("G_im", N * M, "Im(G)"),
                ("g_T_re", K_t * N, "Re(g_k) T-region flattened"),
                ("g_T_im", K_t * N, "Im(g_k) T-region flattened"),
            ]
        else:
            bs_extra, r_extra, t_extra = [], [], []

        rs_r = ris_state_r if self.obs_include_ris_state else []
        rs_t = ris_state_t if self.obs_include_ris_state else []
        rs_all = (ris_state_r + ris_state_t) if self.obs_include_ris_state else []

        if self.obs_include_channel and self.local_obs:
            self._obs_schema = [
                build(base + bs_extra),
                build(base + r_extra + rs_r),
                build(base + t_extra + rs_t),
            ]
        elif self.obs_include_channel:
            full = base + chan_all + rs_all
            self._obs_schema = [build(full) for _ in range(3)]
        else:
            # No raw channel state: per-agent observations are the base block only.
            self._obs_schema = [build(base) for _ in range(3)]

        self._single_schema = build(base + (chan_all if self.obs_include_channel else [])
                                    + rs_all)

        if self.bs_action_mode == "structured_rzf":
            bs_schema = build([
                ("stream_power_logits", K + 1,
                 "bounded logits -> softmax common/private stream powers"),
                ("common_split_logits", K,
                 "inverse-tanh logits -> common-rate split c_k"),
                ("common_beam_weight_logits", K,
                 "user weights for the multicast common-beam direction"),
                ("private_rzf_mix_residual", 1,
                 "bounded residual around configured MRT/RZF mixture prior"),
            ])
        else:
            bs_schema = build([
                ("beamformers_complex", 2 * M * (K + 1),
                 "legacy Re/Im common/private beamformers, jointly normalized"),
                ("common_split_logits", K,
                 "inverse-tanh logits -> common-rate split c_k"),
            ])
        self._act_schema = [
            bs_schema,
            build([
                ("beta_r", N, "beta_r = (a+1)/2 clipped to [1e-4, 1-1e-4]; beta_t = 1 - beta_r"),
                ("phi_r", N, "reflection phases: absolute pi*(a+1) or residual prior + scale*pi*a"),
            ]),
            build([
                ("phi_t", N, "transmission phases: absolute pi*(a+1) or residual prior + scale*pi*a"),
            ]),
        ]

    def observation_schema(self) -> dict:
        """Field layout of per-agent and single-agent (flat) observations."""
        return {
            "agents": [[f.to_dict() for f in sch] for sch in self._obs_schema],
            "single_agent": [f.to_dict() for f in self._single_schema],
            "obs_dims": list(self.obs_dims),
            "single_agent_obs_dim": self.single_agent_obs_dim,
        }

    def action_schema(self) -> dict:
        """Field layout of per-agent actions (network outputs in [-1, 1])."""
        return {
            "agents": [[f.to_dict() for f in sch] for sch in self._act_schema],
            "act_dims": list(self.act_dims),
        }

    def export_schema(self, obs_path: str, act_path: str) -> None:
        with open(obs_path, "w", encoding="utf-8") as f:
            json.dump(self.observation_schema(), f, indent=2)
        with open(act_path, "w", encoding="utf-8") as f:
            json.dump(self.action_schema(), f, indent=2)

    # ---------------------------------------------------------- observation
    def _obs_parts_base(self) -> list[np.ndarray]:
        h = self._h_eff if self._h_eff is not None else np.zeros((self.K, self.M), dtype=np.complex128)
        # Normalisation reference: mean R-region direct gain; fall back to the
        # all-user mean when K_r = 0 (an empty-slice mean is NaN and would
        # silently zero the whole h_eff observation block via _finite()).
        ref_alpha = self.alpha_d[: self.K_r] if self.K_r > 0 else self.alpha_d
        scale = float(ref_alpha.mean()) + 1e-12
        re = (h.real / scale).astype(np.float32).reshape(-1)
        im = (h.imag / scale).astype(np.float32).reshape(-1)
        pw = self._prev_power_weights if self._prev_power_weights is not None \
            else np.ones(self.K + 1, dtype=np.float32) / (self.K + 1)
        cs = self._prev_common_split if self._prev_common_split is not None \
            else np.ones(self.K, dtype=np.float32) / self.K
        return [re, im, pw.astype(np.float32), cs.astype(np.float32)]

    def _ris_state_parts(self, which: str) -> list[np.ndarray]:
        beta = self._beta_r if self._beta_r is not None else 0.5 * np.ones(self.N)
        phi_r = self._phi_r if self._phi_r is not None else np.zeros(self.N)
        phi_t = self._phi_t if self._phi_t is not None else np.zeros(self.N)
        parts = []
        if which in ("r", "all"):
            parts.append((2.0 * beta - 1.0).astype(np.float32))
            parts.append((np.mod(phi_r, 2 * math.pi) / math.pi - 1.0).astype(np.float32))
        if which in ("t", "all"):
            parts.append((np.mod(phi_t, 2 * math.pi) / math.pi - 1.0).astype(np.float32))
        return parts

    def _finite(self, o: np.ndarray) -> np.ndarray:
        o = o.astype(np.float32)
        if not np.all(np.isfinite(o)):
            o = np.nan_to_num(o, nan=0.0, posinf=10.0, neginf=-10.0)
        return o

    def _build_observation(self) -> np.ndarray:
        """FULL single-agent observation (DDPG/TD3/PPO)."""
        parts = self._obs_parts_base()
        if self.obs_include_channel and self._h_d is not None:
            hd_re = (self._h_d.real / (self.alpha_d[:, None] + 1e-30)).astype(np.float32).reshape(-1)
            hd_im = (self._h_d.imag / (self.alpha_d[:, None] + 1e-30)).astype(np.float32).reshape(-1)
            G_re = (self._G.real / (self.alpha_br + 1e-30)).astype(np.float32).reshape(-1)
            G_im = (self._G.imag / (self.alpha_br + 1e-30)).astype(np.float32).reshape(-1)
            g_re = (self._g.real / (self.alpha_ru[:, None] + 1e-30)).astype(np.float32).reshape(-1)
            g_im = (self._g.imag / (self.alpha_ru[:, None] + 1e-30)).astype(np.float32).reshape(-1)
            parts.extend([hd_re, hd_im, G_re, G_im, g_re, g_im])
        if self.obs_include_ris_state:
            parts.extend(self._ris_state_parts("all"))
        return self._finite(np.concatenate(parts))

    def _build_per_agent_observations(self) -> list[np.ndarray]:
        """Per-agent LOCAL observations (true CTDE). Layout per _obs_schema."""
        base = np.concatenate(self._obs_parts_base())

        if not self.obs_include_channel or self._h_d is None:
            return [self._finite(base.copy()) for _ in range(self.n_agents)]

        hd_re = (self._h_d.real / (self.alpha_d[:, None] + 1e-30)).astype(np.float32)
        hd_im = (self._h_d.imag / (self.alpha_d[:, None] + 1e-30)).astype(np.float32)
        G_re = (self._G.real / (self.alpha_br + 1e-30)).astype(np.float32).reshape(-1)
        G_im = (self._G.imag / (self.alpha_br + 1e-30)).astype(np.float32).reshape(-1)
        g_re = (self._g.real / (self.alpha_ru[:, None] + 1e-30)).astype(np.float32)
        g_im = (self._g.imag / (self.alpha_ru[:, None] + 1e-30)).astype(np.float32)

        if self.local_obs:
            bs_obs = np.concatenate([base, hd_re.reshape(-1), hd_im.reshape(-1)])
            r_parts = [base, hd_re[: self.K_r].reshape(-1), hd_im[: self.K_r].reshape(-1), G_re, G_im,
                       g_re[: self.K_r].reshape(-1), g_im[: self.K_r].reshape(-1)]
            t_parts = [base, hd_re[self.K_r:].reshape(-1), hd_im[self.K_r:].reshape(-1), G_re, G_im,
                       g_re[self.K_r:].reshape(-1), g_im[self.K_r:].reshape(-1)]
            if self.obs_include_ris_state:
                r_parts.extend(self._ris_state_parts("r"))
                t_parts.extend(self._ris_state_parts("t"))
            obs_all = [bs_obs,
                       np.concatenate([p for p in r_parts if p.size > 0]) if r_parts else base,
                       np.concatenate([p for p in t_parts if p.size > 0]) if t_parts else base]
        else:
            full_parts = [base, hd_re.reshape(-1), hd_im.reshape(-1), G_re, G_im,
                          g_re.reshape(-1), g_im.reshape(-1)]
            if self.obs_include_ris_state:
                full_parts.extend(self._ris_state_parts("all"))
            full = np.concatenate(full_parts)
            obs_all = [full.copy() for _ in range(self.n_agents)]

        return [self._finite(o) for o in obs_all]

    # ---------------------------------------------------------- Gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._setup_rngs(seed)
        self._scenario = None
        scenario = (options or {}).get("scenario")

        self._step_count = 0
        self._prev_power_weights = np.ones(self.K + 1, dtype=np.float32) / (self.K + 1)
        self._prev_common_split = np.ones(self.K, dtype=np.float32) / self.K
        self._prev_reward = 0.0
        self._prev_applied = None
        self._beta_r = 0.5 * np.ones(self.N)
        self._phi_r = np.zeros(self.N)
        self._phi_t = np.zeros(self.N)

        if scenario is not None:
            # Deterministic scenario playback (ScenarioBank): identical
            # geometry and channel trajectory for every evaluated method.
            self._scenario = scenario
            self.user_positions = np.asarray(scenario["user_positions"], dtype=np.float64)
            self._compute_path_losses()
            self._h_d_small = np.asarray(scenario["h_d_small0"], dtype=np.complex128)
            self._G_small = np.asarray(scenario["G_small0"], dtype=np.complex128)
            self._g_small = np.asarray(scenario["g_small0"], dtype=np.complex128)
            self._apply_large_scale()
        else:
            if self.formulation != "static_block" and self.resample_positions:
                self.user_positions = self._sample_user_positions()
                self._compute_path_losses()
            self._sample_channels()

        self._h_eff = self._effective_channels(self._beta_r, self._phi_r, self._phi_t)
        obs = self._build_observation()
        return obs, {}

    def step(self, action):
        action_list = self._split_action(action)
        for a, d in zip(action_list, self.act_dims):
            assert a.shape[0] == d, f"Action shape mismatch: got {a.shape[0]}, expected {d}"

        if self.formulation == "static_block":
            # Legacy refresh schedule: resample small-scale fading at block
            # boundaries BEFORE applying the action (pre-refactor behaviour,
            # kept for golden-fixture regression).
            if (self._step_count % max(1, self.channel_block_steps)) == 0 and self._step_count > 0:
                self._sample_channels()

        decoded = self._decode_action(action_list)
        self._beta_r = decoded["beta_r"]
        self._phi_r = decoded["phi_r"]
        self._phi_t = decoded["phi_t"]

        # Reward is computed on the CURRENT channel h_t (the one the agent
        # observed); in dynamic_mdp the channel evolves only afterwards.
        self._h_eff = self._effective_channels(self._beta_r, self._phi_r, self._phi_t)
        rsma = self._rsma_rates(self._h_eff, decoded["P_c"], decoded["P_k"], decoded["common_split"], decoded.get("W_c"), decoded.get("W_k"))

        # ------------ QoS metrics (explicit names) ------------
        sum_rate = rsma["sum_rate"]
        per_user = rsma["per_user"]
        c_signed = self.qos_min - per_user                     # constraint fn c_k
        deficit = np.maximum(c_signed, 0.0)                    # d_k = [c_k]_+
        per_user_sat = (per_user >= self.qos_min).astype(np.float64)
        user_qos_fraction = float(per_user_sat.mean())
        all_users_qos_satisfied = bool(np.all(per_user_sat > 0.5))
        min_user_rate = float(per_user.min())
        mean_qos_deficit = float(deficit.mean())
        max_qos_deficit = float(deficit.max())
        total_power = float(decoded["P_c"] + decoded["P_k"].sum())

        # ------------ Reward: penalty/Lagrangian surrogate of P0 ------------
        sw = self._switching_costs(decoded)
        r_sr = self.r_alpha * (sum_rate / max(self.r_ref, 1e-12))
        r_dual = -float(np.dot(self.qos_lambda_vec, c_signed))
        r_aug = -0.5 * self.augmented_penalty_weight * float((deficit ** 2).sum())
        r_switch = -(self.eta_phase * sw["phase_cost"]
                     + self.eta_power * sw["power_cost"]
                     + self.eta_beta * sw["beta_cost"])
        r_bonus = (self.r_qos_bonus * user_qos_fraction
                   if self.enable_qos_shaping_bonus else 0.0)
        reward_raw = r_sr + r_dual + r_aug + r_switch + r_bonus
        reward = float(np.clip(self.r_scale * reward_raw, -self.r_clip, self.r_clip))
        if not math.isfinite(reward):
            reward = -self.r_clip
        # Reward decomposition for off-policy replay recomputation (item 1):
        # base_reward excludes the linear dual term -sum_k lambda_k*c_k, so the
        # critic can recompute reward under the CURRENT lambda at sample time:
        #   reward_current = clip(base_reward - r_scale * dot(lambda_cur, c_k)).
        base_reward = float(self.r_scale * (r_sr + r_aug + r_switch + r_bonus))
        if not math.isfinite(base_reward):
            base_reward = -self.r_clip

        # Bookkeeping for next observation / switching cost.
        self._prev_power_weights = decoded["power_weights"].astype(np.float32)
        self._prev_common_split = decoded["common_split"].astype(np.float32)
        self._prev_reward = reward
        self._prev_applied = {
            "phi_r": decoded["phi_r"].copy(),
            "phi_t": decoded["phi_t"].copy(),
            "beta_r": decoded["beta_r"].copy(),
            "power_weights": np.asarray(decoded["power_weights"], dtype=np.float64).copy(),
        }

        # ------------ Diagnostics ------------
        h2 = rsma["h2"]
        info = {
            "sum_rate": sum_rate,
            "rate_common": rsma["rate_c"],
            "per_user_rate": per_user.copy(),
            "rate_private": rsma["rate_p"].copy(),
            # QoS metrics -- explicit, unambiguous names (P0-2).
            "user_qos_fraction": user_qos_fraction,
            "all_users_qos_satisfied": all_users_qos_satisfied,
            "per_user_qos_satisfied": per_user_sat.copy(),
            "min_user_rate": min_user_rate,
            "mean_qos_deficit": mean_qos_deficit,
            "max_qos_deficit": max_qos_deficit,
            "qos_constraint_signed": c_signed.copy(),          # c_k = R_min - R_k
            "qos_lambda_vec": self.qos_lambda_vec.copy(),
            "qos_lambda_mean": float(self.qos_lambda_vec.mean()),
            # Reward reconstruction under a different lambda (item 1).
            "base_reward": base_reward,
            "reward_scale": float(self.r_scale),
            "reward_clip": float(self.r_clip),
            "total_power_W": total_power,
            # Applied-block quantities (item 6): these are the values for the
            # coherence block on which the reward was computed, captured BEFORE
            # the channel evolves. Diagnostics must read these from info, never
            # the post-transition private env state (self._h_eff, self._phi_*).
            "phi_r_applied": self._phi_r.copy(),
            "phi_t_applied": self._phi_t.copy(),
            "beta_r_applied": self._beta_r.copy(),
            "h_eff_applied_abs": np.abs(self._h_eff).copy(),
            # Reward decomposition.
            "reward_sr": float(self.r_scale * r_sr),
            "reward_dual": float(self.r_scale * r_dual),
            "reward_aug": float(self.r_scale * r_aug),
            "reward_switch": float(self.r_scale * r_switch),
            "reward_bonus": float(self.r_scale * r_bonus),
            # Switching-cost components (pre-weight).
            "phase_switch_cost": sw["phase_cost"],
            "power_switch_cost": sw["power_cost"],
            "beta_switch_cost": sw["beta_cost"],
            # Channel diagnostics.
            "h_eff_abs_mean": float(np.mean(np.linalg.norm(self._h_eff, axis=1))),
            "h_eff_abs_R": float(np.mean(np.linalg.norm(self._h_eff[: self.K_r], axis=1))) if self.K_r > 0 else 0.0,
            "h_eff_abs_T": float(np.mean(np.linalg.norm(self._h_eff[self.K_r:], axis=1))) if self.K_t > 0 else 0.0,
            "h_direct_abs_T": float(np.mean(np.linalg.norm(self._h_d[self.K_r:], axis=1))) if self.K_t > 0 else 0.0,
            "h_ris_abs_T": float(np.mean(np.linalg.norm(self._h_ris[self.K_r:], axis=1))) if (self.K_t > 0 and self._h_ris is not None) else 0.0,
            "h_direct_abs_R": float(np.mean(np.linalg.norm(self._h_d[: self.K_r], axis=1))) if self.K_r > 0 else 0.0,
            "h_ris_abs_R": float(np.mean(np.linalg.norm(self._h_ris[: self.K_r], axis=1))) if (self.K_r > 0 and self._h_ris is not None) else 0.0,
            "ris_to_direct_ratio_T": float(np.mean(np.linalg.norm(self._h_ris[self.K_r:], axis=1)) / max(np.mean(np.linalg.norm(self._h_d[self.K_r:], axis=1)), 1e-30)) if (self.K_t > 0 and self._h_ris is not None) else 0.0,
            "h2_mean": float(np.mean(h2)),
            "phase_entropy_R": self._phase_entropy(self._phi_r),
            "phase_entropy_T": self._phase_entropy(self._phi_t),
            "phase_var_R": float(np.var(self._phi_r)) if self._phi_r.size > 0 else 0.0,
            "phase_var_T": float(np.var(self._phi_t)) if self._phi_t.size > 0 else 0.0,
            "beta_r_mean": float(np.mean(self._beta_r)),
            "common_power_frac": float(decoded["power_weights"][0]),
            "bs_action_mode": decoded.get("bs_action_mode", self.bs_action_mode),
            "bs_private_rzf_mix": float(decoded.get("bs_private_rzf_mix", np.nan)),
        }
        if "ao_grid_n_evals" in decoded:
            info["ao_grid_n_evals"] = decoded["ao_grid_n_evals"]

        # ------------ Transition: channel evolves AFTER the reward ------------
        if self.formulation == "dynamic_mdp":
            self._evolve_channels()
            # next_obs h_eff uses the NEW channel with the currently applied
            # RIS configuration (held until the next action).
            self._h_eff = self._effective_channels(self._beta_r, self._phi_r, self._phi_t)

        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_steps

        obs = self._build_observation()
        return obs, reward, terminated, truncated, info

    # ---------------------------------------------------------- external drivers
    def current_step(self) -> int:
        """Current control-step index (Gauss-Markov innovation index)."""
        return self._step_count

    def advance_to_next_block(self) -> None:
        """Public API: evolve the channel to the next coherence block and
        advance the step counter, mirroring the dynamic_mdp transition order
        (evolve using the CURRENT step index, then increment). External solvers
        that drive the environment without calling step() -- e.g. the AO
        baselines playing back a ScenarioBank -- must use this instead of the
        private channel methods so the innovation index advances 0, 1, 2, ...
        consistently."""
        if self.formulation != "dynamic_mdp":
            raise RuntimeError(
                "advance_to_next_block requires env_formulation=dynamic_mdp")
        self._evolve_channels()
        self._step_count += 1
        self._h_eff = self._effective_channels(self._beta_r, self._phi_r, self._phi_t)

    def apply_decision(self, decoded: dict) -> None:
        """Apply an externally-computed decision (e.g. from an AO solver) to the
        environment's RIS state so subsequent observations/diagnostics reflect
        it. Does not compute reward or advance the channel."""
        self._beta_r = np.asarray(decoded["beta_r"], dtype=np.float64)
        self._phi_r = np.asarray(decoded["phi_r"], dtype=np.float64)
        self._phi_t = np.asarray(decoded["phi_t"], dtype=np.float64)
        self._h_eff = self._effective_channels(self._beta_r, self._phi_r, self._phi_t)

    # ---------------------------------------------------------- dual variables
    def set_qos_lambda_vec(self, vec: np.ndarray) -> None:
        """Set the per-user dual variables, projected onto [0, dual_lambda_max]."""
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        if v.size != self.K:
            raise ValueError(f"qos_lambda_vec must have K={self.K} entries, got {v.size}")
        self.qos_lambda_vec = np.clip(v, 0.0, self.dual_lambda_max)

    def set_qos_lambda(self, value: float) -> None:
        """Back-compat scalar setter: broadcasts to all users (projected)."""
        self.set_qos_lambda_vec(np.full(self.K, float(value)))

    @property
    def qos_lambda(self) -> float:
        """Back-compat scalar view: mean of the per-user dual variables."""
        return float(self.qos_lambda_vec.mean())

    # ---------------------------------------------------------- state (resume)
    def get_state(self) -> dict:
        """Full internal state for exact training resume (TrainingCheckpoint)."""
        def rng_state(r):
            return r.bit_generator.state
        state = {
            "step_count": self._step_count,
            "user_positions": self.user_positions.copy(),
            "h_d_small": None if self._h_d_small is None else self._h_d_small.copy(),
            "G_small": None if self._G_small is None else self._G_small.copy(),
            "g_small": None if self._g_small is None else self._g_small.copy(),
            "beta_r": None if self._beta_r is None else self._beta_r.copy(),
            "phi_r": None if self._phi_r is None else self._phi_r.copy(),
            "phi_t": None if self._phi_t is None else self._phi_t.copy(),
            "prev_power_weights": None if self._prev_power_weights is None else self._prev_power_weights.copy(),
            "prev_common_split": None if self._prev_common_split is None else self._prev_common_split.copy(),
            "prev_reward": self._prev_reward,
            "prev_applied": None if self._prev_applied is None else
                {k: v.copy() for k, v in self._prev_applied.items()},
            "qos_lambda_vec": self.qos_lambda_vec.copy(),
            "rng_states": {
                "geometry": rng_state(self.geometry_rng),
                "channel": rng_state(self.channel_rng),
                "misc": rng_state(self.misc_rng),
            },
        }
        return state

    def set_state(self, state: dict) -> None:
        self._step_count = int(state["step_count"])
        self.user_positions = np.asarray(state["user_positions"], dtype=np.float64)
        self._compute_path_losses()
        self._h_d_small = state["h_d_small"]
        self._G_small = state["G_small"]
        self._g_small = state["g_small"]
        if self._h_d_small is not None:
            self._apply_large_scale()
        self._beta_r = state["beta_r"]
        self._phi_r = state["phi_r"]
        self._phi_t = state["phi_t"]
        self._prev_power_weights = state["prev_power_weights"]
        self._prev_common_split = state["prev_common_split"]
        self._prev_reward = float(state["prev_reward"])
        self._prev_applied = state["prev_applied"]
        self.set_qos_lambda_vec(state["qos_lambda_vec"])
        rs = state["rng_states"]
        self.geometry_rng.bit_generator.state = rs["geometry"]
        self.channel_rng.bit_generator.state = rs["channel"]
        self.misc_rng.bit_generator.state = rs["misc"]
        if self._beta_r is not None and self._h_d is not None:
            self._h_eff = self._effective_channels(self._beta_r, self._phi_r, self._phi_t)

    def render(self):
        h = self._h_eff if self._h_eff is not None else np.zeros(self.K, dtype=complex)
        print(f"step={self._step_count}  |h_eff|^2={np.abs(h)**2}")

    # ------------------------------------------------------------------ helpers
    def _split_action(self, action) -> list[np.ndarray]:
        """Accepts a flat np.ndarray (single-agent) or a list of arrays (MADDPG)."""
        if isinstance(action, (list, tuple)):
            return [np.asarray(a, dtype=np.float32).reshape(-1) for a in action]
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        out = []
        idx = 0
        for d in self.act_dims:
            out.append(arr[idx: idx + d])
            idx += d
        assert idx == arr.size, f"Flat action length {arr.size} != expected {idx}"
        return out

    def global_state(self) -> np.ndarray:
        """Canonical centralized-critic state without duplicated local blocks."""
        state = self._build_observation()
        assert state.shape == (self.global_state_dim,)
        return state

    def per_agent_observations(self, obs: np.ndarray = None) -> list[np.ndarray]:
        """Per-agent local observations (true CTDE) — rebuilds from current env
        state; the argument is ignored (kept signature-compatible)."""
        return self._build_per_agent_observations()
