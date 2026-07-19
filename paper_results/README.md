# Historical outputs intentionally omitted

The numerical CSV/figure/checkpoint outputs from the opened raw-complex/v1
experiment are intentionally not distributed with Structured-BS V2. They are a
negative diagnostic only and must not be reported as confirmatory MISO/Q1
evidence or mixed with V2 shards.

Structured-BS V2 changes the BS action dimension, the physical beamforming
decoder, the primary RIS phase action, the ablation definitions, and the locked
test split. Every algorithm and training seed must therefore be retrained from
the V2 source/config hashes documented in `STRUCTURED_BS_V2.md`.
