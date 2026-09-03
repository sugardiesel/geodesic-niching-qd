# ruff: noqa: E402,I001
"""Build compact figures from the final rolling-only connectivity diagnostics."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_CSV = (
    ROOT
    / "results"
    / "final_summary"
    / "data"
    / "rolling_only_k_sensitivity_report.csv"
)
OUTPUT_STEM = (
    ROOT
    / "results"
    / "final_summary"
    / "figures"
    / "k_sensitivity_centroid_starvation_diagnostic"
)
HORIZONTAL_OUTPUT_STEM = (
    ROOT
    / "results"
    / "final_summary"
    / "figures"
    / "k_sensitivity_centroid_starvation_horizontal"
)

MAPS = ("horseshoe", "open")
COLORS = {"horseshoe": "#c2410c", "open": "#2563eb"}
LABELS = {"horseshoe": "horseshoe", "open": "open"}
MARKERS = {"horseshoe": "o", "open": "s"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rows_for_map(rows: list[dict[str, str]], map_name: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["map"] == map_name]
    return sorted(selected, key=lambda row: int(row["k"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout",
        choices=("vertical", "horizontal"),
        default="vertical",
        help="Choose the vertical or compact horizontal scientific figure.",
    )
    return parser.parse_args()


def register_outputs(output_stem: Path) -> None:
    manifest_path = ROOT / "results" / "final_summary" / "artifact_manifest.csv"
    rows = read_rows(manifest_path) if manifest_path.exists() else []
    new_files = {
        output_stem.with_suffix(suffix)
        .relative_to(ROOT / "results" / "final_summary")
        .as_posix()
        for suffix in (".pdf", ".png")
    }
    rows = [row for row in rows if row["file"] not in new_files]
    for file_name in sorted(new_files):
        rows.append(
            {
                "category": "figure",
                "file": file_name,
                "source": (
                    "analysis/report_k_sensitivity_figure.py + "
                    "results/final_summary/data/rolling_only_k_sensitivity_report.csv"
                ),
                "purpose": "Rolling-only k-sensitivity and permanent-centroid-starvation figure.",
            }
        )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "file", "source", "purpose"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    horizontal = args.layout == "horizontal"
    rows = read_rows(DATA_CSV)

    plt.rcParams.update(
        {
            "font.size": 8.0 if horizontal else 9.5,
            "axes.labelsize": 8.5 if horizontal else 10.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 7.0 if horizontal else 9,
            "ytick.labelsize": 7.5 if horizontal else 9,
            "legend.fontsize": 7.5 if horizontal else 10,
            "lines.linewidth": 1.35 if horizontal else 1.5,
            "lines.markersize": 4.2 if horizontal else 4.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    if horizontal:
        fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.10), sharex=True)
    else:
        # Fixed canvas dimensions preserve legibility at single-column width.
        fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.10), sharex=True)

    for map_name in MAPS:
        map_rows = rows_for_map(rows, map_name)
        k_values = [int(row["k"]) for row in map_rows]
        axes[0].plot(
            k_values,
            [1.0 - float(row["never_reachable_centroids_mean"]) / 625.0 for row in map_rows],
            marker=MARKERS[map_name],
            color=COLORS[map_name],
            label=LABELS[map_name],
        )
        axes[1].plot(
            k_values,
            [float(row["geodesic_to_euclidean_ratio_at_20000_mean"]) for row in map_rows],
            marker=MARKERS[map_name],
            color=COLORS[map_name],
            label=LABELS[map_name],
        )

    axes[0].set_ylabel(
        "Ever-reachable\nfraction",
        labelpad=3 if horizontal else 7,
        linespacing=0.95,
    )
    axes[0].set_ylim(0.40, 0.78)
    axes[1].set_ylabel(
        "Geodesic / Euclidean\nratio",
        labelpad=3 if horizontal else 7,
        linespacing=0.95,
    )
    axes[1].axhline(1.0, color="#6b7280", linewidth=0.8, linestyle="--", zorder=0)
    axes[1].set_ylim(0.98, 1.93)
    if horizontal:
        for axis in axes:
            axis.set_xlabel("$k$", labelpad=0)
    else:
        axes[1].set_xlabel("$k$", labelpad=2)

    for label, axis in zip(("(a)", "(b)"), axes, strict=True):
        axis.text(
            0.98 if horizontal else 0.0,
            0.96 if horizontal else 1.025,
            label,
            transform=axis.transAxes,
            ha="right" if horizontal else "left",
            va="top" if horizontal else "bottom",
            fontsize=8.5 if horizontal else 10,
            fontweight="bold",
        )

    for axis in axes:
        axis.set_xticks([3, 5, 10, 20, 30])
        axis.set_xlim(2.0, 31.0)
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.tick_params(axis="both", which="major", pad=1 if horizontal else 2)

    handles, labels = axes[0].get_legend_handles_labels()
    if horizontal:
        axes[0].legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            ncol=1,
            frameon=False,
            handlelength=1.2,
            labelspacing=0.15,
            borderaxespad=0.0,
        )
        fig.subplots_adjust(left=0.15, right=0.99, bottom=0.25, top=0.97, wspace=0.62)
        output_stem = HORIZONTAL_OUTPUT_STEM
    else:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=2,
            frameon=False,
            handlelength=1.8,
            columnspacing=1.5,
        )
        fig.subplots_adjust(left=0.24, right=0.98, bottom=0.14, top=0.84, hspace=0.78)
        output_stem = OUTPUT_STEM

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    save_options = {"bbox_inches": "tight", "pad_inches": 0.01}
    fig.savefig(output_stem.with_suffix(".pdf"), **save_options)
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
        **save_options,
    )
    plt.close(fig)
    register_outputs(output_stem)


if __name__ == "__main__":
    main()
