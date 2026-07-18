"""Model-complexity accounting and parameter-matched baseline sizing.

Reports, for every algorithm: actor parameters, critic parameters, total
TRAINABLE parameters (main networks only; target copies excluded because they
are not optimized and exist for every off-policy method alike), inference
parameters (actor only) and estimated forward FLOPs (2 * in * out per Linear).

`matched_td3_hidden_sizes` searches the two-layer hidden width h such that a
TD3 agent (actor + twin critic) has total trainable parameters within 5% of
the MADDPG total (3 actors + 3 centralized critics). The 5% bound is asserted
-- if it cannot be met the caller must not present the run as
parameter-matched.
"""
from __future__ import annotations
import torch.nn as nn

from networks import Actor, Critic, CentralizedCritic
from networks.critic import TwinCritic, ValueNet
from networks.actor import StochasticActor


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def linear_flops(module: nn.Module) -> int:
    """Estimated multiply-accumulate FLOPs of one forward pass (2*in*out per
    Linear layer; activations/LayerNorm ignored -- consistent across models)."""
    total = 0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            total += 2 * m.in_features * m.out_features
    return total


def _net_kwargs(net_cfg: dict) -> dict:
    return {"activation": net_cfg.get("activation", "relu"),
            "layer_norm": net_cfg.get("layer_norm", True),
            "ortho": net_cfg.get("ortho_init", True)}


def maddpg_param_counts(spec, hidden_sizes: list[int], net_cfg: dict) -> dict:
    kw = _net_kwargs(net_cfg)
    total_obs = int(sum(spec.obs_dims))
    total_act = int(sum(spec.act_dims))
    actors = [Actor(o, a, hidden_sizes, **kw) for o, a in zip(spec.obs_dims, spec.act_dims)]
    critics = [CentralizedCritic(total_obs, total_act, hidden_sizes, **kw)
               for _ in spec.obs_dims]
    actor_p = sum(count_parameters(m) for m in actors)
    critic_p = sum(count_parameters(m) for m in critics)
    return {"actor_params": actor_p, "critic_params": critic_p,
            "total_params": actor_p + critic_p,
            "inference_params": actor_p,
            "inference_flops": sum(linear_flops(m) for m in actors)}


def td3_param_counts(obs_dim: int, act_dim: int, hidden_sizes: list[int],
                     net_cfg: dict) -> dict:
    kw = _net_kwargs(net_cfg)
    actor = Actor(obs_dim, act_dim, hidden_sizes, **kw)
    critic = TwinCritic(obs_dim, act_dim, hidden_sizes, **kw)
    return {"actor_params": count_parameters(actor),
            "critic_params": count_parameters(critic),
            "total_params": count_parameters(actor) + count_parameters(critic),
            "inference_params": count_parameters(actor),
            "inference_flops": linear_flops(actor)}


def ddpg_param_counts(obs_dim: int, act_dim: int, hidden_sizes: list[int],
                      net_cfg: dict) -> dict:
    kw = _net_kwargs(net_cfg)
    actor = Actor(obs_dim, act_dim, hidden_sizes, **kw)
    critic = Critic(obs_dim, act_dim, hidden_sizes, **kw)
    return {"actor_params": count_parameters(actor),
            "critic_params": count_parameters(critic),
            "total_params": count_parameters(actor) + count_parameters(critic),
            "inference_params": count_parameters(actor),
            "inference_flops": linear_flops(actor)}


def ppo_param_counts(obs_dim: int, act_dim: int, hidden_sizes: list[int],
                     net_cfg: dict) -> dict:
    actor = StochasticActor(obs_dim, act_dim, hidden_sizes,
                            activation="tanh", layer_norm=False,
                            ortho=net_cfg.get("ortho_init", True))
    critic = ValueNet(obs_dim, hidden_sizes, activation="tanh", layer_norm=False,
                      ortho=net_cfg.get("ortho_init", True))
    return {"actor_params": count_parameters(actor),
            "critic_params": count_parameters(critic),
            "total_params": count_parameters(actor) + count_parameters(critic),
            "inference_params": count_parameters(actor),
            "inference_flops": linear_flops(actor)}


def matched_td3_hidden_sizes(obs_dim: int, act_dim: int, target_total: int,
                             net_cfg: dict, tol: float = 0.05,
                             h_min: int = 16, h_max: int = 4096) -> list[int]:
    """Two-layer hidden width so TD3 total trainable params match target_total
    within `tol` (relative). Raises if no width satisfies the bound."""
    def total(h: int) -> int:
        return td3_param_counts(obs_dim, act_dim, [h, h], net_cfg)["total_params"]

    lo, hi = h_min, h_max
    while lo < hi:
        mid = (lo + hi) // 2
        if total(mid) < target_total:
            lo = mid + 1
        else:
            hi = mid
    candidates = [max(h_min, lo - 1), lo]
    best = min(candidates, key=lambda h: abs(total(h) - target_total))
    mismatch = abs(total(best) - target_total) / max(target_total, 1)
    if mismatch > tol:
        raise ValueError(
            f"Cannot parameter-match TD3 within {tol:.0%}: best width {best} "
            f"gives {total(best)} params vs target {target_total} "
            f"(mismatch {mismatch:.1%}). Do not report this run as "
            "parameter-matched.")
    return [best, best]
