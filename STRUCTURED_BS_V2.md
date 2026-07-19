# Structured-BS V2 — physics-informed MISO controller

## Why this revision exists

The opened v1 locked-test experiment showed that direct end-to-end output of
all complex BS beamformer coefficients was physically dominated by a simple
MRT/equal-power controller. Structured-BS V2 removes that failure mode instead
of attempting post-hoc hyperparameter tuning.

This revision is a **new method**. It does not reuse or reinterpret the old 40
shards, checkpoints, replay files, aggregate tables, or v1 locked-test bank.

## BS action schema

For K users, the BS action is reduced from `2*M*(K+1)+K` to `3*K+2`:

1. `K+1` bounded stream-power logits;
2. `K` common-rate split logits;
3. `K` common-beam user-weight logits;
4. one bounded residual around an MRT/RZF private-beam mixture.

For the default M=4, K=4, N=32 configuration:

- BS action: **44 -> 14**;
- total action: **140 -> 110**;
- local observations: `[73, 577, 401]`;
- canonical critic state: `681`.

The deterministic decoder first applies the current STAR-RIS action, computes
the resulting effective MISO channel, constructs unit-norm MRT and regularized
zero-forcing directions, then applies the actor's low-dimensional powers,
common split/beam weights and bounded mixture residual. Zero actor output maps
to a strong reproducible RZF/equal-power prior rather than an arbitrary complex
beamformer.

The primary STAR-RIS action is also physics-informed:

```text
phi = corrected analytical phase prior + 0.15*pi*actor residual
```

## Fair classical baselines

Learned policies and AO-Grid now call the same `_physics_beamformers` routine.
AO-Grid therefore cannot be weakened by silently falling back to MRT while the
learned method uses RZF.

The former composite `EqualPower+Learned` label is retired. Clean ablations are:

- `EqualPowerOnly`;
- `MRTDirectionsOnly`;
- `UniformCommonSplitOnly`;
- `UniformCommonBeamOnly`;
- `AnalyticalRIS`, `FixedRIS`, `RandomRIS`, `NoRIS`;
- `ClassicalMRTEqualPowerFixedRIS` as an explicitly composite classical baseline.

## Validation-only physics sanity

These are deterministic zero-action diagnostics on the **development
validation bank** (50 scenarios), not final paper results:

| Controller | Sum-rate mean | User QoS fraction | All-users QoS | Mean min-user rate |
|---|---:|---:|---:|---:|
| Structured RZF + analytical residual prior | 14.0628 | 1.0000 | 1.0000 | 1.6857 |
| Structured MRT + analytical residual prior | 5.0398 | 1.0000 | 1.0000 | 0.9532 |
| Classical MRT + equal power + fixed RIS | 4.7831 | 1.0000 | 1.0000 | 0.7236 |
| Legacy raw-complex zero decoder | 1.5636 | 0.8200 | 0.3000 | 0.2146 |

The sanity check shows that the catastrophic beam-direction collapse is gone.
It does **not** establish that MADDPG outperforms TD3 or TD3-Matched. All
algorithms receive the same structured action map, so superiority must still be
demonstrated by a fresh paired experiment.

## Verification

- `python -m pytest -q`: **134 passed**; 22 expected SciPy SLSQP clipping warnings.
- CLI smoke completed for MADDPG, DDPG, TD3, TD3-Matched and PPO.
- Smoke locked-bank outputs were finite and all five algorithms completed the
  full train/evaluate/ablation/report path.
- Default full-model parameter accounting:
  - MADDPG: `1,308,017`;
  - TD3-Matched `[373, 373]`: `1,310,088`;
  - mismatch: `0.1583%`.

## Frozen provenance

- Method: `structured_bs_v2`
- Source-tree SHA-256: `a363f99a6300cc698dcb581f17fb3479b408e3b454043913e261e5e5de0442e2`
- Effective config SHA-256: `7e9ef64bcb2f19d324c70a9d69492fb1c3e7b034cf8b28023f7e421f127936b3`
- Seed-split SHA-256: `dc327042efcad7007efa0006f0b10461eb69055992a3cdcd6416612d15db13e8`
- Fresh locked v2 test seeds: `[81011, 81023, 81041, 81071, 81101]`
- Opened historical v1 test seeds, forbidden for new confirmatory claims:
  `[70001, 70002, 70003, 70004, 70005]`

## Compatibility

Old checkpoints fail by design because action dimensions and network shapes
changed. Retrain all five algorithms and all eight training seeds. Do not mix a
V2 shard with any `source_sha` or `config_sha` from the raw-complex experiment.

## Publication gate

Before spending quota on scalability, the full 8-seed N=32 run should show that
trained policies at least preserve the strong validation prior and are not
dominated by the clean classical baseline. A claim that MADDPG benefits from
multi-agent decomposition still requires a paired, Holm-corrected comparison
against **TD3-Matched**, not only TD3/DDPG/PPO.
