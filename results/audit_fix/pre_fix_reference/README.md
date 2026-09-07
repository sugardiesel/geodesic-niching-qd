# Historical comparison inputs

These twelve summary files are the inputs to
`scripts/compare_pre_post_audit_results.py`. They come from the superseded
pre-correction runs, not the final experiment set. No old checkpoints or full
trajectory logs are included. `manifest.csv` records the source snapshots and
SHA256 hashes of the unchanged copies.

The final results remain in `results/phase5/`, `results/phase6_open_robustness/`
and `results/final_summary/`. Production aggregation does not read this folder.

From the repository root:

```powershell
uv run python scripts/compare_pre_post_audit_results.py
```

The output goes to `results/audit_fix/pre_post_comparison/`. The historical
comparison is supplementary and must not replace the final tables.
