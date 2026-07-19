# Paper results snapshot - historical only

This directory is retained for audit/reference, but the CSV files must not be
reported as final MISO/Q1 evidence.

The current source uses the MISO STAR-RIS RSMA model (`M=4` by default), while
parts of this snapshot were produced by an older source tree and/or SISO
configuration. Treat every table and figure derived from this directory as
historical until the full MISO experiment matrix is rerun from a frozen commit
and verified by `source_sha`, `config_sha`, and locked validation/test banks.

Required before using this directory for a paper/thesis result:

- rerun all primary algorithms on the frozen MISO source;
- verify all 40 primary algorithm-seed shards are present;
- regenerate aggregation, significance, latency, complexity and scalability
  tables from those shards;
- report MADDPG vs TD3/TD3-Matched only according to paired tests with Holm
  correction.
