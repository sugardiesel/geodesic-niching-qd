# Final results

This directory collects the final tables and figures derived from the completed searches in
`results/phase5/` and `results/phase6_open_robustness/`. Generating it does not launch
MAP-Elites.

## Experiment set

- Horseshoe map: seeds 1001-1005 for all three conditions.
- Open map: seeds 1001-1003 for all three conditions.
- 20,000 evaluations per run.
- Rolling-only geodesic support with endpoint-only centroids and `k=20`.
- Baseline A and Baseline B are the matched reference runs used in the final aggregation.

## Tables

- `data/corrected_k20_per_seed_metrics.csv`: all per-seed quality, diversity, runtime, and
  throughput metrics.
- `data/corrected_k20_condition_mean_std.csv`: condition means and sample standard deviations.
- `data/corrected_k20_wilcoxon_paired_tests.csv`: all paired tests, seed arrays, differences, and
  rank-biserial effects.
- `data/corrected_k20_baseline_b_vs_contribution_wilcoxon.csv`: Baseline B versus Contribution.
- `data/rolling_only_k_sensitivity_report.csv`: final permanent-reachability and
  distance-ratio values for `k in {3,5,10,20,30}`.
- `data/phase1_horseshoe_u_bottom_detour_provenance.csv`: shortest-path provenance for the
  reported horseshoe landmark pair.

Files prefixed by `horseshoe_` and `open_` are unmodified copies of the source aggregate outputs.
`artifact_manifest.csv` maps packaged artifacts to the original result files or generating code.

## Main results

Baseline B versus Contribution showed no detectable difference under paired Wilcoxon tests for
the archive metrics or the shared raw behavior-space metrics. Sample sizes are small (five and
three paired seeds), so these are limited-power results rather than evidence of exact equality.

Post-retraining assignment disagreement was 14.096% on the horseshoe map and 13.780% on the open
map. The connectivity audit found mean permanent starvation of 222.8 horseshoe and 204.3 open-map
centroids out of 625 at the locked `k=20`.

## Rebuild

```powershell
uv run python scripts/prepare_report_inputs.py
uv run python scripts/audit_rolling_only_k_sensitivity.py
uv run python analysis/report_k_sensitivity_figure.py --layout vertical
uv run python analysis/report_k_sensitivity_figure.py --layout horizontal
```

The first command rebuilds summary tables, provenance, archive heatmaps, QD/coverage curves, and
the horseshoe detour figure. The second command validates the `k=20` reachability rows
against the exact production replay before writing the k-sensitivity table.
