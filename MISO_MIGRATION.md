# MISO migration

This branch replaces the former SISO physical layer with a multi-antenna BS model.

## Physical model

- BS antennas: `M >= 2` (default `M=4`).
- Effective channel: `H_eff` has shape `(K, M)`.
- BS action: complex common beamformer and `K` complex private beamformers, plus the common-rate split.
- The environment projects the joint beamformer matrix onto the total BS power constraint.
- Common/private SINRs use the full MISO coupling `|h_k^H w_j|^2`.
- STAR-RIS reflection/transmission coefficients alter every antenna component of the cascaded channel.

## Experimental integrity

The goal is to make multi-agent decomposition meaningful, not to weaken TD3. TD3 remains a centralized baseline with the same physical action and observation space. Any claim that MADDPG outperforms TD3 must be based on the frozen multi-seed evaluation and paired statistical tests. A short smoke run only validates execution.

## Recommended run order

```bash
python main.py --config config/smoke.yaml --quick --episodes 5 --algos maddpg td3
python main.py --config config/pilot.yaml --episodes 300 --algos maddpg td3 td3_matched
python main.py --config config/config.yaml --algos maddpg td3 td3_matched
```
