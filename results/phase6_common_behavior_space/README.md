# Common behavior-space analysis

These files use the corrected k=20 Contribution archives and the original Baseline B archives.
The metric space remains the originally specified 8 x 8 x 8 grid: path efficiency, loop score
clipped to [0, 12], and food count clipped to [0, 8], normalized to [0, 1].

## Elite identification

Final-stage insertion records identify Baseline B elites added after retraining. Earlier
fitness/counter ties require codes from the final encoder, obtained by replaying 18 historical
policies with the original checkpoint and environment timing. Contribution identities use
the saved full-run final-space codes. Unresolved matches raise an error.

This corrects 22 of 3,694 previously selected trajectory records (15 Baseline B, 7 Contribution).
The original archive policies, fitness values, and production Table 1 are not changed. All
six common-space Wilcoxon p-values remain unchanged, although some metric means change.
The full identity changes are in `../submission_verification/common_space_identity_changes.json`.

## Results

Values are mean +/- sample standard deviation. Coverage is a percentage of 512 common cells.

| Map | Condition | Coverage (%) | Shifted QD | Pairwise distance |
| --- | --- | ---: | ---: | ---: |
| Horseshoe | Baseline B | 13.6719 +/- 2.0899 | 1461.9734 +/- 224.4089 | 0.384111 +/- 0.0420 |
| Horseshoe | Contribution | 12.7734 +/- 0.5794 | 1395.4402 +/- 89.1736 | 0.415546 +/- 0.0382 |
| Open | Baseline B | 14.8438 +/- 1.9236 | 1573.4610 +/- 236.0289 | 0.432476 +/- 0.0426 |
| Open | Contribution | 13.1510 +/- 1.3005 | 1443.2913 +/- 93.0593 | 0.417429 +/- 0.0142 |

Use `common_behavior_metrics_by_seed.csv` for individual seed values and
`common_behavior_condition_summary.csv` for unrounded means and standard deviations.
Paired Wilcoxon statistics, exact p-values, and rank-biserial effects are in
`common_behavior_wilcoxon_paired_tests.csv`.

## Reproduction

Run from the repository root:

```powershell
uv run python scripts/recover_baseline_elite_latents.py
uv run python scripts/analyze_common_behavior_space.py
```

The first command verifies the historical ties; the second uses the resulting final-space
codes and existing raw feature logs. Neither launches evolution or autoencoder training.
