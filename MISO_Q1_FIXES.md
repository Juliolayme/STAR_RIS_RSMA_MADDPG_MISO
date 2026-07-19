# MISO/Q1 correctness hardening

Base snapshot: `agent/miso-q1-hardening` at commit
`67d6adec0ecb5e735f35e01a36c41ebd1dec3c9e`.

## Correctness fixes

1. **Analytical STAR-RIS phase prior**
   - Derived directly from
     `h_d^H q + g^H Phi G q`.
   - The nonzero-direct-link branch now aligns each cascaded received term with
     `h_d^H q`; the zero-direct fallback still aligns cascaded terms to a
     zero-phase reference.
   - A deterministic nonzero-direct oracle prevents conjugation/sign regressions.

2. **PPO time-limit handling**
   - Stores `next_value`, `terminated`, and `episode_end` per transition.
   - TD residual bootstraps through time-limit truncation but not through a true
     terminal.
   - GAE recursion stops at both terminal and truncation/reset boundaries, so a
     rollout spanning multiple episodes cannot leak advantages across resets.

3. **Canonical MADDPG centralized state**
   - Local actor observations remain `[73, 577, 401]` for the default MISO
     configuration.
   - Critics consume one canonical global state of dimension `681`, rather than
     concatenating local observations with repeated shared blocks.
   - Joint critic input is `681 + 140 = 821` dimensions.
   - Replay, normalizer, inference checkpoint, and resumable checkpoint schemas
     include canonical global states.

4. **Fair TD3-Matched accounting**
   - MADDPG critic parameter counting uses the canonical state dimension.
   - Complexity-only model construction disables orthogonal initialization,
     which does not affect parameter/FLOP counts and makes width search fast.
   - Default configuration: MADDPG `1,338,767` trainable parameters;
     TD3-Matched hidden width `[372, 372]`, `1,338,970` parameters
     (`0.0152%` mismatch).

5. **Research-integrity/documentation guards**
   - Primary paper configuration remains `phase_action_mode: absolute`.
   - Residual phase is documented only as an optional preregistered ablation.
   - Historical SISO/static-pipeline outputs remain explicitly non-final.
   - Thesis MISO equations, dimensions, conclusion, and diagram prompts were
     corrected to remove stale SISO/residual claims.
   - Golden physics snapshot records source-file provenance and is explicitly a
     regression snapshot, not an independent correctness oracle.

## Validation completed

- `python -m pytest -q`: **129 passed**.
- Analytical phase randomized audit: corrected prior beat the old expression in
  **200/200** channel realizations.
- Two-episode CLI smoke shards completed for:
  `maddpg`, `ddpg`, `td3`, `td3_matched`, and `ppo`.
- `git diff --check`: passed.

## Compatibility and experiment status

The MADDPG critic input and replay schema changed. **Do not resume or evaluate
old MADDPG checkpoints as if they belonged to this source.** Retrain every
algorithm/seed from the frozen source SHA, including recalculated TD3-Matched.

Passing these gates means the source is suitable for experiment freeze. It does
not create final Q1 evidence. Chapter 4 tables, figures, latency, scalability,
and statistical claims still require a fresh complete MISO rerun on the locked
seed split and ScenarioBank.
