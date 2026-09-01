# Results Directory Guide

This directory contains the corrected production outputs and the supporting validation artifacts
used during development. The `phase*` prefixes are internal project milestones, not sections of
the report and not separate versions of the final method.

## Start Here

Open [`final_summary/`](final_summary/) first. It is the compact examiner-facing index containing
the report-ready tables, figures, statistical tests, and provenance for the authoritative result
set.

The authoritative searches themselves are:

- [`phase5/`](phase5/): horseshoe-map production sweep, with five matched seeds per condition.
- [`phase6_open_robustness/`](phase6_open_robustness/): open-map robustness sweep, with three
  matched seeds per condition.

Both use 20,000 evaluations per run. Their Contribution runs use the corrected endpoint-only,
rolling-support graph at `k=20`. These are the only search outputs used for the final cross-map
comparisons. Each production directory includes generated per-run configs, raw evaluation logs,
archive cells, metric histories, summaries, and paired statistical tests.

## What The Phase Names Mean

| Label | Development milestone | Role in the repository |
|---|---|---|
| Phase 1 | Environment and maze-distance validation | Establishes the task, throughput, and detour geometry. |
| Phase 2 | Hand-crafted MAP-Elites validation | Early Baseline A sanity run; not the final multi-seed result. |
| Phase 3/3B | Learned-descriptor development | Autoencoder, retraining, normalization, and latent-space checks. |
| Phase 4 | Geodesic graph development | Dynamic graph and neighborhood-size diagnostics. |
| Phase 5 | Main production experiment | Authoritative five-seed horseshoe comparison. |
| Phase 6 | Robustness and post-hoc analysis | Authoritative open-map comparison and mechanism analyses. |
| Audit | Correctness verification | Post-implementation graph, connectivity, and support-set checks. |

## Directory Index

### Environment And Baseline Validation

- [`phase1_environment/`](phase1_environment/): random and hand-coded rollouts, speed validation,
  and the original Euclidean-versus-maze-geodesic diagnostic.
- [`phase1_distance_diagnostic_corrected/`](phase1_distance_diagnostic_corrected/): corner-safe
  shortest-path recomputation and report-grade region-split plots.
- [`phase2_handcrafted_map_elites/`](phase2_handcrafted_map_elites/): single-seed Baseline A sanity
  archive and learning curves.

### Learned-Descriptor Development

- [`phase3_aurora_euclidean_seed_1001/`](phase3_aurora_euclidean_seed_1001/): initial
  learned-descriptor Baseline B run.
- [`phase3b_aurora_frozen_normalized_seed_1001/`](phase3b_aurora_frozen_normalized_seed_1001/):
  frozen-encoder ablation.
- [`phase3b_aurora_one_retrain_normalized_seed_1001/`](phase3b_aurora_one_retrain_normalized_seed_1001/):
  locked one-retrain encoder design used by the production pipeline.
- [`phase3b_aurora_original_raw_loss_diagnostic_seed_1001/`](phase3b_aurora_original_raw_loss_diagnostic_seed_1001/):
  apples-to-apples reconstruction-loss diagnostic for the earlier loss definition.
- [`phase3_latent_diagnostic/`](phase3_latent_diagnostic/): representative latent-distance versus
  true maze-distance analysis.
- [`phase3_knn_k_sensitivity/`](phase3_knn_k_sensitivity/): static latent-space `k` sensitivity.
- [`phase4_dynamic_graph_k_sensitivity/`](phase4_dynamic_graph_k_sensitivity/): early dynamic-graph
  sensitivity checkpoint; the final corrected audit is listed below.

### Final Production And Post-Hoc Analyses

- [`phase5/`](phase5/): authoritative horseshoe production runs and aggregate statistics.
- [`phase6_open_robustness/`](phase6_open_robustness/): authoritative open-map production runs and
  aggregate statistics.
- [`phase6_common_behavior_space/`](phase6_common_behavior_space/): representation-independent
  comparison using fixed raw trajectory features.
- [`phase6_high_blowup_arrival_analysis/`](phase6_high_blowup_arrival_analysis/): timing and arrival
  analysis of large Euclidean-to-geodesic assignment corrections.
- [`phase6_depth_fitness_ceiling_analysis/`](phase6_depth_fitness_ceiling_analysis/): post-hoc test
  of geodesic depth against final niche fitness.
- [`phase6_encoder_pool_similarity/`](phase6_encoder_pool_similarity/): comparison of the
  trajectory pools used to retrain the independently learned encoders.

### Correctness And Connectivity Audits

- [`audit_fix/`](audit_fix/): intermediate outputs used while correcting and rechecking graph and
  reporting behavior. These are supporting audit artifacts, not an additional production set.
- [`audit_connectivity_verification/`](audit_connectivity_verification/): exact replay of the
  corrected endpoint-only graph, dynamic `k` sensitivity, and centroid-starvation measurements.
- [`elite_support_connectivity_diagnostic/`](elite_support_connectivity_diagnostic/): opt-in
  elite-retention support-set test. It did not pass its promotion gate and is not the production
  default.

Known-buggy pre-audit search outputs are intentionally excluded from this examiner-facing
repository and must not be combined with the corrected production runs.
