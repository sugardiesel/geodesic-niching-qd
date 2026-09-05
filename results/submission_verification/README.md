# Submission verification

## Penalty counter

`GeodesicGridArchive.refresh_graph()` calls `endpoint_shortest_path_distances()` and passes
its result to `_apply_disconnection_penalty()`. The matrix has one row per centroid and one
column per support node. Support includes both visited points and centroids. With 1,000
distinct visited points and 625 centroids, the matrix is 625 x 1,625, not a table of candidates.

`np.count_nonzero(~np.isfinite(distances))` counts non-finite entries in that matrix.
The cumulative counter adds this count on each nonempty graph refresh. It includes
centroid-to-centroid entries and can count the same disconnected pair repeatedly.
Candidate queries do not increment this counter. The graph is recreated at retraining,
so the final production counter covers the 41 post-retraining refreshes, not both encoders.

| Map | Seed | Cumulative penalized cache entries | Penalized winning assignments |
| --- | ---: | ---: | ---: |
| Horseshoe | 1001 | 24498301 | 0 / 30000 |
| Horseshoe | 1002 | 22259923 | 0 / 30000 |
| Horseshoe | 1003 | 23545725 | 0 / 30000 |
| Horseshoe | 1004 | 25676696 | 0 / 30000 |
| Horseshoe | 1005 | 24441829 | 0 / 30000 |
| Open | 1001 | 17136333 | 0 / 30000 |
| Open | 1002 | 22594276 | 0 / 30000 |
| Open | 1003 | 24450782 | 0 / 30000 |

Cache counts come from each `phase4_summary.json`; winning-route counts come from
`../audit_connectivity_verification/penalty_audit/penalty_audit_by_seed.csv`.
The 30,000 decisions include 20,000 original assignments and the 10,000-candidate archive
rebuild, not 30,000 environment evaluations. Zero penalized winners does not imply that
penalties had no effect on novelty-weighted parent selection.

The older zero-cache-penalty finding concerned the superseded pre-audit graph. It should not
be relabeled as a winning-assignment statistic. The corrected graph's zero winning-route
count is a separate, later measurement.

The forced-disconnection unit test checks a 2 x 4 matrix: four penalized entries include
two centroid-to-visited and two centroid-to-centroid entries. A finite winning query leaves
the counter unchanged; another graph refresh increments it from four to eight.

```powershell
uv run pytest -q tests/test_geodesic_archive.py
```

## Common-space recovery

The final-encoder identity check corrects 22 selected records: 15 Baseline B and seven
Contribution elites. All 3,694 final elites are accounted for, without changing any saved
production archive. `common_space_identity_changes.json` lists the old and corrected
evaluation indices relative to commit `ab7a6967fedb467aadd544ec4132b9ddc5f74b74`.
The six paired Wilcoxon p-values are unchanged. Reproduction commands and numerical results
are in `../phase6_common_behavior_space/README.md`.

## Historical environment timing

Baseline B identity recovery replays 18 historical policies with the original checkpoint.
Saved fitness, episode counters, and trajectory diagnostics agree within 1e-8; the helper
aborts otherwise. The original timing sets `respawn_timer = respawn_steps`; the current
environment sets `respawn_timer = respawn_steps + 1` to account for the immediate decrement.

The comparison CSVs here verify both timings on selected saved policies. Two Baseline B
policies (horseshoe seed 1002, evaluations 2441 and 3380) and four Contribution policies
(horseshoe seed 1001, evaluations 2055, 2070, 2085 and 2100) reproduce exactly with historical
timing. Corrected timing changes three of the four Contribution trajectories; for evaluation
2055 it changes survival by 92 steps and food count by two. This is not a harmless rounding
difference. These checks do not establish a timing difference between conditions; both
checked conditions used historical timing. They also do not certify every production policy.

Run from the repository root:

```powershell
uv run python scripts/verify_historical_respawn_timing.py `
  --seed-dir results/phase5/baseline_b_learned_bd_euclidean/seed_1002 `
  --evaluations 2441 3380 `
  --output reproduced/baseline_respawn_replay.csv
uv run python scripts/verify_historical_respawn_timing.py `
  --seed-dir results/phase5/contribution_geodesic_niching/seed_1001 `
  --evaluations 2055 2070 2085 2100 `
  --output reproduced/contribution_respawn_replay.csv
```

Genome recovery uses the saved fitness/assignment history and the original RNG stream. It
verifies archive insertion decisions and does not evaluate or select new search candidates.
Only the requested historical policies are evaluated. The main search entry points still
use the current corrected environment; they have not been silently switched to legacy timing.
