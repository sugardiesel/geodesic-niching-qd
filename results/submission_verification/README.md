# Replay data and counter definitions

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

## Elite identification

The common-space analysis associates each final elite with its saved trajectory record.
It uses final-encoder descriptors, archive insertion history, and replay of ambiguous
Baseline B policies. Contribution matches use the saved final-space codes rather than
policy replay.

`common_space_identity_changes.json` records 22 corrected matches among 3,694 elites
(15 Baseline B, seven Contribution). Method details, metrics, and reproduction commands
are in the [common-space analysis](../phase6_common_behavior_space/).

## Food-respawn timing

The saved experiments use `--food-respawn-timing historical`, which sets
`respawn_timer = respawn_steps`. The current default, `corrected`, sets
`respawn_timer = respawn_steps + 1` to account for the immediate decrement and therefore
respawns food one step later. All experiment entry points accept this option and record it
in their summaries. Sweep resume rejects a mismatched or unrecorded timing setting.

`baseline_respawn_replay.csv` and `contribution_respawn_replay.csv` compare the two rules on
selected saved policies. `contribution_historical_mode_verification.csv` contains the
historical-only comparison. These are selected-policy replays, not full search reruns.

### Reproduction

From the repository root:

```text
uv run python scripts/verify_historical_respawn_timing.py --seed-dir results/phase5/baseline_b_learned_bd_euclidean/seed_1002 --evaluations 2441 3380 --food-respawn-timing historical --require-match --output reproduced/baseline_respawn_replay.csv
uv run python scripts/verify_historical_respawn_timing.py --seed-dir results/phase5/contribution_geodesic_niching/seed_1001 --evaluations 2055 2070 2085 2100 --food-respawn-timing historical --require-match --output reproduced/contribution_respawn_replay.csv
```

Genome recovery uses the saved fitness and assignment history with the original random
number stream. Only the specified policies are evaluated; no evolution is performed.
`--require-match` raises an error if fitness, episode counters, or trajectory diagnostics
differ from the saved values by more than 1e-8. To compare the two timer rules, use
`--food-respawn-timing both` without `--require-match`.
