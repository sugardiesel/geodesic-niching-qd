# Historical comparison inputs

This directory contains summaries from the superseded pre-correction runs.
`scripts/compare_pre_post_audit_results.py` uses them to compare the earlier
and final results.

The final results remain in `results/phase5/`, `results/phase6_open_robustness/`
and `results/final_summary/`. Production aggregation does not read this folder.

From the repository root:

```text
uv run python scripts/compare_pre_post_audit_results.py
```

The output goes to `results/audit_fix/pre_post_comparison/`.
