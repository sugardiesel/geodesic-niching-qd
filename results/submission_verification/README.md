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

The replay denominator is different: 18 candidate policies resolve eight tied Baseline B
elites. Four of those eight matches changed and four were already correct. Another 11
Baseline B corrections follow from the recorded archive history without replay. This gives
15 corrected elite identities, not 15 corrected policies out of 18 replays. All seven corrected
Contribution identities match their saved final-space descriptor with distance exactly 0.0,
as well as fitness and episode counters. Those seven policies were not re-simulated.

## Historical environment timing

Baseline B identity recovery replays 18 historical policies with the original checkpoint.
Saved fitness, episode counters, and trajectory diagnostics agree within 1e-8; the helper
aborts otherwise. The original timing sets `respawn_timer = respawn_steps`; the current
default sets `respawn_timer = respawn_steps + 1` to account for the immediate decrement.
Both rules are now available through `--food-respawn-timing historical|corrected`; the
default remains corrected. The original Phase 1 compatibility class remains an independent
reference for the historical rule.

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
  --food-respawn-timing historical --require-match `
  --output reproduced/baseline_respawn_replay.csv
uv run python scripts/verify_historical_respawn_timing.py `
  --seed-dir results/phase5/contribution_geodesic_niching/seed_1001 `
  --evaluations 2055 2070 2085 2100 `
  --food-respawn-timing historical --require-match `
  --output reproduced/contribution_respawn_replay.csv
```

Genome recovery uses the saved fitness/assignment history and the original RNG stream. It
verifies archive insertion decisions and does not evaluate or select new search candidates.
Only the requested historical policies are evaluated. The explicit historical-mode check
produces zero error in all nine logged diagnostics for the four Contribution policies above
(`contribution_historical_mode_verification.csv`). All 18 Baseline B identity replays also
retain zero diagnostic error. `--require-match` exits with an error if any discrepancy exceeds
1e-8. To compare the two rules instead, use `--food-respawn-timing both` without that assertion.

All three experiment entry points and the sweep accept the same timing option. The sweep
stores the selection in generated YAML, and each runner records it in its output summary.
Resume rejects a mismatched or unrecorded timing version; aggregation-only mode does not
accept a timing override. The README production commands select historical timing explicitly,
but no new search was run to test this addition. Tests inspect the runner configuration while
replacing the search function with a stop marker, so they cannot launch evolution.

Verification on 2026-09-05: all 31 unit tests pass, including the replay-count reconciliation,
historical timer/reference comparison, CLI/config propagation, and resume safeguards. Ruff
reports no errors. Run these checks without launching a search:

```powershell
uv run python -m unittest discover -s tests -v
uv run ruff check .
```
