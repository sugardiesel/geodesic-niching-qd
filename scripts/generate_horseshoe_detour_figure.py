"""Generate the corrected horseshoe detour figure and provenance CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.phase1_detour_figure import generate_horseshoe_detour_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1_horseshoe.yaml")
    parser.add_argument(
        "--legacy-summary",
        default="results/phase1_environment/phase1_summary.json",
    )
    parser.add_argument(
        "--corrected-summary",
        default="results/phase1_distance_diagnostic_corrected/phase1_summary.json",
    )
    parser.add_argument("--pair", default="u_bottom_detour")
    parser.add_argument(
        "--output-stem",
        default="results/final_summary/figures/phase1_horseshoe_u_bottom_detour",
    )
    parser.add_argument(
        "--provenance",
        default="results/final_summary/data/phase1_horseshoe_u_bottom_detour_provenance.csv",
    )
    args = parser.parse_args()

    result = generate_horseshoe_detour_artifacts(
        config_path=args.config,
        legacy_summary_path=args.legacy_summary,
        corrected_summary_path=args.corrected_summary,
        output_stem=args.output_stem,
        provenance_path=args.provenance,
        pair_name=args.pair,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
