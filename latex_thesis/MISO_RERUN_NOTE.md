# MISO rerun note

The thesis text and generated tables are being migrated from the historical
SISO snapshot to the MISO implementation.

Do not submit Chapter 4 tables/figures as final MISO evidence until the full
MISO experiment matrix is rerun and the following artifacts are regenerated:

- `latex_thesis/tables/algorithm_comparison.tex`
- `latex_thesis/tables/significance.tex`
- `latex_thesis/tables/scalability.tex`
- `latex_thesis/tables/latency_cpu.tex`
- all training/scalability figures derived from `paper_results/`

Current authoritative code schema for the default MISO config (`M=4`, `K=4`,
`K_R=3`, `N=32`) is:

- action dimension: 140
- local observation dimensions: 73, 577, 401
- centralized critic input: 1191
