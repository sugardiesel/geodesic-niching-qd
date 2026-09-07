# Retraining-pool comparison

These tables compare the raw trajectory features collected by Baseline B and the
Contribution before retraining, using the five matched horseshoe seeds in
`results/phase5/`. The archive-driven pool contains evaluations 2,001 through
10,000; bootstrap and full-retraining-pool comparisons are logged separately.

The files were regenerated on 2026-09-07 from the corrected k=20 production runs.
They replace the earlier supplementary outputs, which used the pre-correction
Contribution runs. No archive, search, or encoder was changed. This analysis is
not a shared-encoder experiment: similar raw-feature distributions cannot
establish that independently trained representations are identical.

From the repository root:

```powershell
uv run python scripts/analyze_phase5_encoder_pool_similarity.py
```

For a separate output directory, add
`--output-dir reproduced/encoder_pool_similarity`.
