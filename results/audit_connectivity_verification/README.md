# Corrected graph connectivity verification

This audit uses the corrected endpoint-safe graph: the rolling trajectory points are routing
nodes, the 25x25 archive centroids are endpoints, centroid-to-centroid edges are removed, and
centroids cannot be shortest-path transit nodes. No MAP-Elites search was rerun for this audit.

## Static representative samples

The static check uses 400 representative saved trajectory latents. `ratio` is mean finite
geodesic distance divided by mean Euclidean distance over the same finite pairs.

| Map | k | Ratio | Components | Finite-pair fraction |
|---|---:|---:|---:|---:|
| Horseshoe | 3 | 2.132 | 1 | 1.000 |
| Horseshoe | 5 | 1.190 | 1 | 1.000 |
| Horseshoe | 10 | 1.066 | 1 | 1.000 |
| Horseshoe | 20 | 1.020 | 1 | 1.000 |
| Horseshoe | 30 | 1.009 | 1 | 1.000 |
| Open | 3 | 1.810 | 4 | 0.406 |
| Open | 5 | 1.162 | 1 | 1.000 |
| Open | 10 | 1.044 | 1 | 1.000 |
| Open | 20 | 1.014 | 1 | 1.000 |
| Open | 30 | 1.006 | 1 | 1.000 |

## Dynamic production-style graphs

Each row aggregates the actual corrected Contribution streams at that snapshot: five horseshoe
seeds or three open-map seeds. Graph support is the last 1,000 visited latents plus the 625 archive
centroid endpoints. `components` is the routing-point component mean with the seed range in
parentheses. `connected` counts fully connected routing graphs. `finite` and `ratio` are means over
all support pairs, with the ratio evaluated only on finite pairs.

| Map | Eval | k | Ratio | Components mean (range) | Finite | Connected seeds |
|---|---:|---:|---:|---:|---:|---:|
| Horseshoe | 2,000 | 3 | 1.711 | 44.0 (29-54) | 0.256 | 0/5 |
| Horseshoe | 2,000 | 5 | 1.319 | 20.4 (13-28) | 0.405 | 0/5 |
| Horseshoe | 2,000 | 10 | 1.142 | 7.6 (4-10) | 0.526 | 0/5 |
| Horseshoe | 2,000 | 20 | 1.069 | 3.2 (2-4) | 0.638 | 0/5 |
| Horseshoe | 2,000 | 30 | 1.052 | 2.0 (1-3) | 0.705 | 1/5 |
| Horseshoe | 10,000 | 3 | 1.762 | 34.8 (24-43) | 0.184 | 0/5 |
| Horseshoe | 10,000 | 5 | 1.306 | 10.4 (8-12) | 0.436 | 0/5 |
| Horseshoe | 10,000 | 10 | 1.108 | 5.2 (3-7) | 0.544 | 0/5 |
| Horseshoe | 10,000 | 20 | 1.054 | 2.2 (2-3) | 0.631 | 0/5 |
| Horseshoe | 10,000 | 30 | 1.037 | 1.8 (1-2) | 0.679 | 1/5 |
| Horseshoe | 20,000 | 3 | 1.862 | 39.2 (28-44) | 0.204 | 0/5 |
| Horseshoe | 20,000 | 5 | 1.323 | 11.2 (5-17) | 0.488 | 0/5 |
| Horseshoe | 20,000 | 10 | 1.098 | 3.4 (1-7) | 0.585 | 1/5 |
| Horseshoe | 20,000 | 20 | 1.046 | 1.6 (1-3) | 0.685 | 3/5 |
| Horseshoe | 20,000 | 30 | 1.041 | 1.0 (1-1) | 0.751 | 5/5 |
| Open | 2,000 | 3 | 1.613 | 40.0 (33-49) | 0.243 | 0/3 |
| Open | 2,000 | 5 | 1.391 | 17.0 (14-21) | 0.433 | 0/3 |
| Open | 2,000 | 10 | 1.108 | 6.7 (4-9) | 0.523 | 0/3 |
| Open | 2,000 | 20 | 1.048 | 3.0 (2-4) | 0.607 | 0/3 |
| Open | 2,000 | 30 | 1.032 | 3.0 (2-4) | 0.641 | 0/3 |
| Open | 10,000 | 3 | 1.768 | 45.3 (41-51) | 0.228 | 0/3 |
| Open | 10,000 | 5 | 1.300 | 14.0 (9-19) | 0.471 | 0/3 |
| Open | 10,000 | 10 | 1.125 | 3.7 (2-6) | 0.598 | 0/3 |
| Open | 10,000 | 20 | 1.047 | 2.0 (1-3) | 0.681 | 1/3 |
| Open | 10,000 | 30 | 1.039 | 1.0 (1-1) | 0.754 | 3/3 |
| Open | 20,000 | 3 | 1.819 | 46.3 (39-56) | 0.143 | 0/3 |
| Open | 20,000 | 5 | 1.333 | 10.3 (6-16) | 0.523 | 0/3 |
| Open | 20,000 | 10 | 1.096 | 1.7 (1-2) | 0.630 | 1/3 |
| Open | 20,000 | 20 | 1.040 | 1.3 (1-2) | 0.699 | 2/3 |
| Open | 20,000 | 30 | 1.031 | 1.0 (1-1) | 0.772 | 3/3 |

No tested k is both curvature-revealing and robustly connected throughout the dynamic runs.
`k=20` is less fragmented than k=3/5/10 but is not connectivity-safe, while k=30 is still
fragmented at early horseshoe/open snapshots and is already close to Euclidean distance.

## Exact production replay and centroid starvation

The replay reconstructs both encoder stages, both archive rebuilds, and every live 250-evaluation
graph refresh. It reproduces all 144,000 saved online cell assignments exactly. The 30,000 audited
decisions per seed include the 10,000 pre-retrain decisions, the 10,000-candidate final-space
rebuild, and the 10,000 post-retrain online decisions.

| Map | Seed | Actual penalty-selected assignments | Mean reachable centroids post-retrain | Never reachable | Unreachable >50% of refreshes | Final occupied | Never-reachable occupied |
|---|---:|---:|---:|---:|---:|---:|---:|
| Horseshoe | 1001 | 0/30,000 | 53.86% | 220 | 292 | 192 | 0 |
| Horseshoe | 1002 | 0/30,000 | 57.13% | 191 | 266 | 208 | 0 |
| Horseshoe | 1003 | 0/30,000 | 56.01% | 203 | 278 | 212 | 0 |
| Horseshoe | 1004 | 0/30,000 | 48.83% | 265 | 322 | 177 | 0 |
| Horseshoe | 1005 | 0/30,000 | 51.69% | 235 | 300 | 192 | 0 |
| Open | 1001 | 0/30,000 | 68.91% | 124 | 192 | 264 | 0 |
| Open | 1002 | 0/30,000 | 55.22% | 242 | 278 | 196 | 0 |
| Open | 1003 | 0/30,000 | 51.22% | 247 | 303 | 198 | 0 |

The matrix-level repair was active at every post-retrain refresh because many centroid-to-support
entries were infinite. Nevertheless, an infinite entry was never on the winning assignment path:
the finite penalty itself was selected 0 times. This does not make disconnection harmless. A
permanently unreachable centroid always loses to some reachable centroid with a real finite path,
so it can be starved without the penalty becoming the winning distance. Across seeds, a mean of
222.8 horseshoe centroids and 204.3 open-map centroids were never reachable at any post-retrain
refresh; all remained empty.

Therefore the corrected Contribution coverage drop cannot be interpreted as verified evidence of
an intrinsic coverage-for-quality trade-off. It is confounded by endpoint/grid support failure.
The requested Baseline-B-versus-Contribution trade-off heatmap was intentionally not produced
because its prerequisite connectivity gate failed.

## Reproduction

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\phase3_knn_k_sensitivity.py --sample-npz results\phase5\baseline_b_learned_bd_euclidean\seed_1001\representative_trajectory_sample.npz --k-values 3 5 10 20 30 --output-dir results\audit_connectivity_verification\static_horseshoe_seed_1001
.\.venv\Scripts\python.exe scripts\phase3_knn_k_sensitivity.py --sample-npz results\phase6_open_robustness\baseline_b_learned_bd_euclidean\seed_1001\representative_trajectory_sample.npz --k-values 3 5 10 20 30 --output-dir results\audit_connectivity_verification\static_open_seed_1001

$seeds = 1001, 1002, 1003, 1004, 1005
foreach ($seed in $seeds) {
    .\.venv\Scripts\python.exe scripts\phase4_dynamic_graph_k_sensitivity.py --phase4-dir "results\phase5\contribution_geodesic_niching\seed_$seed" --config "results\phase5\configs\contribution_geodesic_niching_seed_$seed.yaml" --snapshots 2000 10000 20000 --k-values 3 5 10 20 30 --rolling-buffer-size 1000 --output-dir "results\audit_connectivity_verification\dynamic_horseshoe_seed_$seed"
}

$seeds = 1001, 1002, 1003
foreach ($seed in $seeds) {
    .\.venv\Scripts\python.exe scripts\phase4_dynamic_graph_k_sensitivity.py --phase4-dir "results\phase6_open_robustness\contribution_geodesic_niching\seed_$seed" --config "results\phase6_open_robustness\configs\contribution_geodesic_niching_seed_$seed.yaml" --snapshots 2000 10000 20000 --k-values 3 5 10 20 30 --rolling-buffer-size 1000 --output-dir "results\audit_connectivity_verification\dynamic_open_seed_$seed"
}

.\.venv\Scripts\python.exe scripts\audit_geodesic_connectivity_penalty.py
```

The dynamic per-seed CSVs are under `dynamic_<map>_seed_<seed>/`. The combined exact tables and
JSON summary are under `penalty_audit/`.
