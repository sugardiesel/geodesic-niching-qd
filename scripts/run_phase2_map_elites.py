"""Run Phase 2: hand-crafted MAP-Elites with final (x, y) behavior descriptor."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from algorithms.map_elites import run_map_elites
from analysis.phase2_plots import save_archive_heatmap, save_metric_curves
from envs.forage_maze import ForageMaze2D, load_env_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_handcrafted_map_elites.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{args.config} must contain a mapping.")

    env_config_path = config["experiment"]["env_config"]
    env = ForageMaze2D(load_env_config(env_config_path))
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    archive, records, metrics = run_map_elites(env, config)
    elapsed = time.perf_counter() - start

    archive_csv = output_dir / "archive_cells.csv"
    metrics_csv = output_dir / "metrics.csv"
    evaluations_csv = output_dir / "evaluations.csv"
    write_csv(archive_csv, archive.cell_rows())
    write_csv(metrics_csv, metrics)
    write_csv(evaluations_csv, [record_to_row(record) for record in records])

    heatmap_files = save_archive_heatmap(
        archive,
        env,
        output_dir / "archive_heatmap_final_xy",
        "Phase 2 MAP-Elites archive: final (x, y) descriptor",
    )
    curve_files = save_metric_curves(
        metrics,
        output_dir / "qd_score_coverage_curves",
        title="Phase 2 hand-crafted MAP-Elites progress",
    )

    final_metrics = archive.metrics()
    summary = {
        "config": args.config,
        "env_config": env_config_path,
        "seed": int(config["experiment"]["seed"]),
        "descriptor": config["algorithm"]["descriptor"],
        "grid_bins": config["algorithm"]["grid_bins"],
        "hidden_dim": int(config["algorithm"]["hidden_dim"]),
        "parameter_count": int(archive.genome_size),
        "total_evaluations": int(config["algorithm"]["total_evaluations"]),
        "initial_random": int(config["algorithm"]["initial_random"]),
        "elapsed_seconds": elapsed,
        "evaluations_per_second": int(config["algorithm"]["total_evaluations"]) / elapsed,
        "final_metrics": final_metrics,
        "occupancy_entropy": final_metrics["occupancy_entropy"],
        "output_files": [
            str(archive_csv),
            str(metrics_csv),
            str(evaluations_csv),
            *heatmap_files,
            *curve_files,
            str(output_dir / "phase2_summary.json"),
        ],
    }
    (output_dir / "phase2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def record_to_row(record: object) -> dict[str, object]:
    return {
        "evaluation": record.evaluation,
        "source": record.source,
        "fitness": record.fitness,
        "descriptor_x": float(record.descriptor[0]),
        "descriptor_y": float(record.descriptor[1]),
        "cell_x": int(record.cell[0]),
        "cell_y": int(record.cell[1]),
        "inserted": bool(record.inserted),
        "replaced": bool(record.replaced),
        "steps": int(record.steps),
        "food_collected": int(record.food_collected),
        "wall_collisions": int(record.wall_collisions),
        "hazard_contacts": int(record.hazard_contacts),
    }


def print_summary(summary: dict[str, object]) -> None:
    metrics = summary["final_metrics"]
    print("PHASE 2 HAND-CRAFTED MAP-ELITES SUMMARY")
    print(f"config: {summary['config']}")
    print(f"env_config: {summary['env_config']}")
    print(f"seed: {summary['seed']}")
    print(f"descriptor: {summary['descriptor']}")
    print(f"grid_bins: {summary['grid_bins']}")
    print(f"parameter_count: {summary['parameter_count']}")
    print(
        "runtime: {total_evaluations} evals in {elapsed_seconds:.3f}s = "
        "{evaluations_per_second:.2f} eval/s".format(**summary)
    )
    print(
        "final archive: filled={filled_cells}, coverage={coverage:.3%}, "
        "QD-score={qd_score:.3f}, max fitness={max_fitness:.3f}, "
        "mean fitness={mean_fitness:.3f}".format(**metrics)
    )
    print(f"occupancy entropy (coverage-derived): {summary['occupancy_entropy']:.3f}")
    print("output files:")
    for output_file in summary["output_files"]:
        print(f"  {output_file}")


if __name__ == "__main__":
    main()
