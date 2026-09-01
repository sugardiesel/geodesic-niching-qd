# Result Directory Guide

The repository includes completed corrected outputs so the reported analysis can be inspected
without repeating the expensive searches.

## Authoritative Searches

- `phase5/`: horseshoe-map production sweep, five matched seeds per condition.
- `phase6_open_robustness/`: open-map robustness sweep, three matched seeds per condition.
- `final_summary/`: compact cross-map tables, final figures, and provenance.

Each production directory contains the generated per-run configs, raw evaluation logs, archive
cells, metric histories, summaries, and aggregate paired statistics. The Contribution runs in
these directories use the corrected endpoint-only rolling graph at `k=20`.

## Core Diagnostics

- `phase1_environment/` and `phase1_distance_diagnostic_corrected/`: environment validation and
  the corner-safe Euclidean-versus-maze-geodesic analysis.
- `phase3b_aurora_one_retrain_normalized_seed_1001/`: locked learned-descriptor design.
- `phase3_latent_diagnostic/` and `phase3_knn_k_sensitivity/`: representative latent analyses.
- `audit_connectivity_verification/`: exact corrected graph replay and centroid starvation.
- `elite_support_connectivity_diagnostic/`: opt-in elite-retention test that failed its gate.
- `phase6_common_behavior_space/`: representation-independent archive comparison.
- Other `phase6_*` directories: mechanism-level post-hoc analyses described in their summaries.

Known-buggy pre-audit outputs are intentionally absent from the examiner-facing tree. They remain
available in earlier Git history for audit traceability but must not be aggregated with the
corrected production runs.
