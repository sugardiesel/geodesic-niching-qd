# Retraining-pool comparison

Comparison of Baseline B and Contribution trajectory features before encoder retraining,
using the five matched horseshoe seeds from the final k=20 experiment set. The archive-driven
pool contains evaluations 2,001-10,000. Bootstrap and full-pool statistics are recorded
separately.

`seed_pool_similarity_summary.csv` contains per-seed results;
`feature_distribution_comparison.csv` gives feature-level comparisons. Similar raw-feature
distributions do not establish that the independently trained encoders are equivalent.

## Reproduction

From the repository root:

```text
uv run python scripts/analyze_phase5_encoder_pool_similarity.py
```
