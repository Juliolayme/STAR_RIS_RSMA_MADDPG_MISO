"""PPO dual updates are rollout-consistent (item 2): the dual (lambda) vector
is updated ONLY after a PPO policy/GAE update, so every optimization batch uses
a single lambda vector (lambda is fixed while a rollout is being collected)."""
from __future__ import annotations
import os

from conftest import full_cfg


def test_dual_update_only_after_ppo_learn(tmp_path, monkeypatch):
    import experiments.train as T

    cfg = full_cfg()
    cfg["ppo"]["rollout_length"] = 20        # rollout spans ~2 episodes
    cfg["training"]["total_episodes"] = 4
    cfg["training"]["eval_every"] = 1000     # skip validation
    cfg["training"]["log_dir"] = os.path.join(str(tmp_path), "logs")
    cfg["training"]["ckpt_dir"] = os.path.join(str(tmp_path), "ckpt")

    calls = []
    orig_learn = T.PPOAgent.learn
    orig_dual = T.DualUpdater.update

    def learn_wrap(self, last_v=None):
        calls.append("learn")
        return orig_learn(self, last_v)

    def dual_wrap(self, env, mean_c, ep=None, total_episodes=None):
        calls.append("dual")
        return orig_dual(self, env, mean_c, ep=ep, total_episodes=total_episodes)

    monkeypatch.setattr(T.PPOAgent, "learn", learn_wrap)
    monkeypatch.setattr(T.DualUpdater, "update", dual_wrap)

    T.train_ppo(cfg, run_name="ppo_dual",
                log_dir=cfg["training"]["log_dir"],
                ckpt_dir=cfg["training"]["ckpt_dir"])

    assert "learn" in calls, "PPO never updated -- test inconclusive"
    # Every dual update must immediately follow a PPO learn (lambda fixed during
    # the whole rollout that the learn consumed).
    for idx, c in enumerate(calls):
        if c == "dual":
            assert idx > 0 and calls[idx - 1] == "learn", \
                "lambda updated without a preceding PPO update (mid-rollout)"
