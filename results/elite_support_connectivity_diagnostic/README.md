# Elite-retention support diagnostic

This post-hoc test uses the eight saved final Contribution streams and does not rerun policies or
MAP-Elites search. The support set was defined before inspecting results:

- all current archive elite latent codes, retained until that niche's elite changes;
- the most recent 500 raw candidate latents;
- stable exact deduplication, with elite points taking precedence over recent duplicates;
- all 625 fixed grid centroids as endpoints only;
- no centroid-centroid edges and no centroid transit.

Graphs are refreshed every 250 evaluations. Historical candidate and archive-assignment streams
are held fixed to isolate the support-set change. The eval-10,000 archive rebuild is reconstructed
in final-encoder coordinates, and reconstructed elite counts are checked against saved run
summaries. The acceptance criteria are one routing component at every post-retrain refresh, zero
never-reachable centroids, and at least 95% mean centroid reachability on both maps without making
the graph effectively Euclidean.

## Static reference

The 400-trajectory static latent diagnostic does not depend on archive support composition, but it
was recomputed for every available seed. Values are means across five horseshoe or three open-map
seeds.

| Map | k | Ratio | Components | Finite-pair fraction |
|---|---:|---:|---:|---:|
| Horseshoe | 3 | 1.780 | 2.40 | 0.949 |
| Horseshoe | 5 | 1.193 | 1.00 | 1.000 |
| Horseshoe | 10 | 1.062 | 1.00 | 1.000 |
| Horseshoe | 20 | 1.019 | 1.00 | 1.000 |
| Horseshoe | 30 | 1.009 | 1.00 | 1.000 |
| Open | 3 | 1.760 | 2.67 | 0.734 |
| Open | 5 | 1.171 | 1.00 | 1.000 |
| Open | 10 | 1.048 | 1.00 | 1.000 |
| Open | 20 | 1.013 | 1.00 | 1.000 |
| Open | 30 | 1.006 | 1.00 | 1.000 |

## Dynamic snapshots

Each cell is `geodesic/Euclidean ratio / routing components / finite-pair fraction / reachable-
centroid fraction`, averaged across seeds. Ratios use only finite pairs.

| Map/eval | k=3 | k=5 | k=10 | k=20 | k=30 |
|---|---|---|---|---|---|
| Horse 2k | 1.538/52.6/.119/.328 | 1.597/24.0/.252/.378 | 1.176/7.0/.421/.436 | 1.079/3.0/.546/.573 | 1.071/1.8/.647/.667 |
| Horse 10k | 1.694/43.8/.098/.332 | 1.333/12.6/.338/.363 | 1.129/4.6/.451/.409 | 1.052/2.6/.543/.536 | 1.037/2.4/.600/.619 |
| Horse 20k | 1.644/53.4/.071/.409 | 1.401/10.0/.411/.444 | 1.113/2.8/.546/.487 | 1.049/1.4/.654/.610 | 1.038/1.0/.733/.699 |
| Open 2k | 1.477/44.3/.126/.298 | 1.425/17.7/.284/.327 | 1.116/7.3/.394/.383 | 1.062/3.0/.497/.523 | 1.051/2.7/.559/.609 |
| Open 10k | 1.702/49.3/.086/.362 | 1.352/12.3/.361/.414 | 1.167/3.7/.509/.476 | 1.055/1.3/.628/.591 | 1.040/1.3/.698/.681 |
| Open 20k | 1.836/61.7/.072/.420 | 1.444/12.7/.387/.470 | 1.125/1.3/.605/.536 | 1.048/1.0/.693/.643 | 1.031/1.0/.757/.723 |

The apparently connected final routing graphs at larger k do not imply endpoint coverage. Even
when the real-point routing graph is one component, many centroids have no retained edge to any
real point and are therefore unreachable.

## Full post-retrain refresh history

These statistics use all 41 graph refreshes from eval 10,000 through 20,000, not only the three
snapshots. Counts are means across seeds.

| Map | k | Mean reachable | Never reachable | Unreachable >50% | Connected refreshes | Mean components |
|---|---:|---:|---:|---:|---:|---:|
| Horseshoe | 3 | 38.24% | 336.8 | 386.4 | 0.00% | 49.28 |
| Horseshoe | 5 | 41.65% | 318.2 | 364.8 | 0.00% | 11.60 |
| Horseshoe | 10 | 46.03% | 290.8 | 337.8 | 2.44% | 3.53 |
| Horseshoe | 20 | 58.21% | 221.0 | 259.8 | 42.93% | 1.79 |
| Horseshoe | 30 | 66.89% | 169.4 | 208.2 | 72.20% | 1.34 |
| Open | 3 | 40.57% | 338.0 | 370.3 | 0.00% | 56.82 |
| Open | 5 | 45.33% | 309.0 | 340.3 | 0.00% | 13.30 |
| Open | 10 | 51.46% | 271.7 | 302.0 | 54.47% | 2.12 |
| Open | 20 | 62.87% | 202.7 | 230.7 | 95.12% | 1.07 |
| Open | 30 | 71.52% | 153.0 | 176.7 | 97.56% | 1.02 |

At k=5, no seed has a single connected post-retrain refresh. At k=10, the horseshoe graph is
connected in only about 2.4% of refreshes. k=20/30 improve routing connectivity, but hundreds of
centroid endpoints remain permanently unreachable, while the final distance ratio has fallen to
about 1.03-1.05.

## Direct k=20 comparison with rolling-only support

| Map | Seed | Mean reachable old -> elite | Never old -> elite | >50% old -> elite |
|---|---:|---:|---:|---:|
| Horseshoe | 1001 | 53.86% -> 58.03% | 220 -> 221 | 292 -> 262 |
| Horseshoe | 1002 | 57.13% -> 61.44% | 191 -> 190 | 266 -> 240 |
| Horseshoe | 1003 | 56.01% -> 60.76% | 203 -> 203 | 278 -> 247 |
| Horseshoe | 1004 | 48.83% -> 52.40% | 265 -> 259 | 322 -> 295 |
| Horseshoe | 1005 | 51.69% -> 58.43% | 235 -> 232 | 300 -> 255 |
| Open | 1001 | 68.91% -> 73.59% | 124 -> 124 | 192 -> 160 |
| Open | 1002 | 55.22% -> 58.10% | 242 -> 241 | 278 -> 263 |
| Open | 1003 | 51.22% -> 56.92% | 247 -> 243 | 303 -> 269 |

At k=20, mean reachable fraction improves by 4.71 percentage points on horseshoe and 4.42 on the
open map. Unreachable-more-than-half counts improve by 31.8 and 27.0 cells. However, mean
never-reachable counts improve by only 1.8 and 1.7 cells. The targeted aging-out hypothesis is
therefore not the main cause of permanent starvation.

## Conclusion

Elite support gives a modest partial improvement but does not make k=5 or k=10 safe and does not
substantially reduce permanently unreachable centroids. Larger k still leaves 153-221
never-reachable centroids on average and makes graph distances nearly Euclidean.

This supports a structural limitation of the tested design: a sparse evolving real-point graph
cannot reliably maintain attachments to a fixed dense rectangular centroid grid under the same
k-NN rule, even when successful elite locations are retained.

## Reproduction

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\audit_elite_support_connectivity.py
.\.venv\Scripts\python.exe -m pytest
```

The diagnostic takes roughly 3.5 minutes on the audited machine and performs no environment
rollouts or evolutionary search. Per-seed static, snapshot, and all-refresh tables are saved next
to this README.
