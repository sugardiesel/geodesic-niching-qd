"""Build the horseshoe detour figure from validated config and diagnostic outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from analysis.phase1_plots import save_detour_map_plot
from envs.forage_maze import ForageMaze2D
from envs.maze_distance import GridShortestPath


def generate_horseshoe_detour_artifacts(
    config_path: str | Path,
    legacy_summary_path: str | Path,
    corrected_summary_path: str | Path,
    output_stem: str | Path,
    provenance_path: str | Path,
    pair_name: str = "u_bottom_detour",
) -> dict[str, Any]:
    """Generate the figure and its legacy-versus-corrected provenance row."""
    config_path = Path(config_path)
    legacy_summary_path = Path(legacy_summary_path)
    corrected_summary_path = Path(corrected_summary_path)
    output_stem = Path(output_stem)
    provenance_path = Path(provenance_path)

    env = ForageMaze2D.from_config_path(config_path)
    pair = _named_entry(env.map_cfg.get("diagnostic_pairs", []), pair_name)
    point_a = np.asarray(pair["a"], dtype=np.float64)
    point_b = np.asarray(pair["b"], dtype=np.float64)

    legacy_summary = _load_json(legacy_summary_path)
    corrected_summary = _load_json(corrected_summary_path)
    legacy_landmark = _named_entry(legacy_summary["landmark_diagnostic_pairs"], pair_name)
    corrected_landmark = _named_entry(corrected_summary["landmark_diagnostic_pairs"], pair_name)

    shortest = GridShortestPath(
        env,
        grid_size=int(env.config["shortest_path"]["grid_size"]),
        connectivity=int(env.config["shortest_path"]["connectivity"]),
    )
    path = shortest.shortest_path(point_a, point_b)
    if len(path) < 2:
        raise RuntimeError(f"No finite Dijkstra path found for {pair_name}.")

    euclidean = float(np.linalg.norm(point_a - point_b))
    current_geodesic = shortest.distance(point_a, point_b)
    path_length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    corrected_geodesic = float(corrected_landmark["geodesic"])
    if not np.isclose(path_length, current_geodesic, rtol=0.0, atol=1e-10):
        raise RuntimeError("Rendered Dijkstra polyline does not match the current solver distance.")
    if not np.isclose(current_geodesic, corrected_geodesic, rtol=0.0, atol=1e-10):
        raise RuntimeError(
            "Corrected Phase 1.5 summary does not match the current Dijkstra solver."
        )
    if not np.isclose(euclidean, float(corrected_landmark["euclidean"]), atol=1e-12):
        raise RuntimeError("Configured landmark endpoints do not match the corrected summary.")

    legacy_distance = float(legacy_landmark["geodesic"])
    legacy_ratio = float(legacy_landmark["geodesic_to_euclidean_ratio"])
    corrected_ratio = corrected_geodesic / euclidean
    legacy_diagnostic = legacy_summary["phase1_5_distance_diagnostic"]
    corrected_diagnostic = corrected_summary["phase1_5_distance_diagnostic"]
    pair_validation = corrected_summary["pair_set_validation"]
    report_values = corrected_summary["report_authoritative_values"]

    row = {
        "pair_name": pair_name,
        "config_path": config_path.as_posix(),
        "solver": "envs/maze_distance.py::GridShortestPath",
        "wall_renderer": "analysis/phase1_plots.py::draw_environment",
        "food_hazard_markers_are_fixed_config_features": "true",
        "food_hazard_positions_source": "configs/phase1_horseshoe.yaml::map.foods/map.hazards",
        "point_a_x": f"{point_a[0]:.12f}",
        "point_a_y": f"{point_a[1]:.12f}",
        "point_b_x": f"{point_b[0]:.12f}",
        "point_b_y": f"{point_b[1]:.12f}",
        "grid_size": str(shortest.grid_size),
        "connectivity": str(shortest.connectivity),
        "agent_radius": f"{env.agent_radius:.12f}",
        "euclidean_distance": f"{euclidean:.12f}",
        "legacy_geodesic_distance": f"{legacy_distance:.12f}",
        "corrected_geodesic_distance": f"{corrected_geodesic:.12f}",
        "geodesic_change_percent": f"{100.0 * (corrected_geodesic / legacy_distance - 1.0):.9f}",
        "legacy_geodesic_to_euclidean_ratio": f"{legacy_ratio:.12f}",
        "corrected_geodesic_to_euclidean_ratio": f"{corrected_ratio:.12f}",
        "path_grid_point_count": str(len(path)),
        "rendered_polyline_length": f"{path_length:.12f}",
        "legacy_overall_pearson_r": f"{float(legacy_diagnostic['pearson_correlation']):.12f}",
        "corrected_overall_pearson_r": (
            f"{float(corrected_diagnostic['pearson_correlation']):.12f}"
        ),
        "legacy_horseshoe_region_pearson_r": (
            f"{float(legacy_diagnostic['horseshoe_region_pearson_correlation']):.12f}"
        ),
        "corrected_horseshoe_region_pearson_r": (
            f"{float(corrected_diagnostic['horseshoe_region_pearson_correlation']):.12f}"
        ),
        "legacy_open_field_pearson_r": (
            f"{float(legacy_diagnostic['open_field_pearson_correlation']):.12f}"
        ),
        "corrected_open_field_pearson_r": (
            f"{float(corrected_diagnostic['open_field_pearson_correlation']):.12f}"
        ),
        "legacy_horseshoe_region_pair_count": str(
            int(legacy_diagnostic["horseshoe_region_pair_count"])
        ),
        "corrected_horseshoe_region_pair_count": str(
            int(corrected_diagnostic["horseshoe_region_pair_count"])
        ),
        "legacy_open_field_pair_count": str(int(legacy_diagnostic["open_field_pair_count"])),
        "corrected_open_field_pair_count": str(
            int(corrected_diagnostic["open_field_pair_count"])
        ),
        "legacy_median_pair_ratio": (
            f"{float(legacy_diagnostic['median_geodesic_to_euclidean_ratio']):.12f}"
        ),
        "corrected_median_pair_ratio": (
            f"{float(corrected_diagnostic['median_geodesic_to_euclidean_ratio']):.12f}"
        ),
        "legacy_p95_pair_ratio": (
            f"{float(legacy_diagnostic['p95_geodesic_to_euclidean_ratio']):.12f}"
        ),
        "corrected_p95_pair_ratio": (
            f"{float(corrected_diagnostic['p95_geodesic_to_euclidean_ratio']):.12f}"
        ),
        "legacy_max_pair_ratio": (
            f"{float(legacy_diagnostic['max_geodesic_to_euclidean_ratio']):.12f}"
        ),
        "corrected_max_pair_ratio": (
            f"{float(corrected_diagnostic['max_geodesic_to_euclidean_ratio']):.12f}"
        ),
        "legacy_pair_count_ratio_gt_3": str(
            int(legacy_diagnostic["detour_pair_count_ratio_gt_3"])
        ),
        "corrected_pair_count_ratio_gt_3": str(
            int(corrected_diagnostic["detour_pair_count_ratio_gt_3"])
        ),
        "legacy_pair_count_ratio_gt_5": str(
            int(legacy_diagnostic["detour_pair_count_ratio_gt_5"])
        ),
        "corrected_pair_count_ratio_gt_5": str(
            int(corrected_diagnostic["detour_pair_count_ratio_gt_5"])
        ),
        "pair_set_euclidean_values_match_legacy_csv_at_8_decimals": str(
            bool(pair_validation["euclidean_values_match_legacy_csv_at_8_decimals"])
        ).lower(),
        "pair_set_max_absolute_euclidean_difference": (
            f"{float(pair_validation['max_absolute_euclidean_difference_from_saved_csv']):.12g}"
        ),
        "changed_geodesic_pair_count_at_8_decimals": str(
            int(pair_validation["changed_geodesic_pair_count_at_8_decimals"])
        ),
        "mean_absolute_geodesic_change": (
            f"{float(pair_validation['mean_absolute_geodesic_change']):.12f}"
        ),
        "max_absolute_geodesic_change": (
            f"{float(pair_validation['max_absolute_geodesic_change']):.12f}"
        ),
        "region_labels_match_legacy_csv": str(
            bool(pair_validation["region_labels_match_legacy_csv"])
        ).lower(),
        "report_authoritative_overall_pearson_r_3dp": (
            f"{float(report_values['overall_pearson_r_rounded_3dp']):.3f}"
        ),
        "report_authoritative_horseshoe_pearson_r_3dp": (
            f"{float(report_values['horseshoe_region_pearson_r_rounded_3dp']):.3f}"
        ),
        "report_authoritative_open_field_pearson_r_3dp": (
            f"{float(report_values['open_field_pearson_r_rounded_3dp']):.3f}"
        ),
        "report_values_note": str(report_values["note"]),
        "legacy_summary_source": legacy_summary_path.as_posix(),
        "corrected_summary_source": corrected_summary_path.as_posix(),
        "supersession_note": (
            "Supersedes the original Phase 1.5 u_bottom_detour geodesic value: the "
            "corrected solver forbids diagonal corner cutting around blocked grid cells; "
            "the exact historical pair set, map, endpoints, grid size, and connectivity "
            "are unchanged."
        ),
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    with provenance_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    with provenance_path.open("r", newline="", encoding="utf-8") as handle:
        label_row = next(csv.DictReader(handle))
    figure_files = save_detour_map_plot(
        env=env,
        point_a=point_a,
        point_b=point_b,
        shortest_path=path,
        output_stem=output_stem,
        euclidean_distance=float(label_row["euclidean_distance"]),
        geodesic_distance=float(label_row["corrected_geodesic_distance"]),
    )

    return {
        "pair_name": pair_name,
        "point_a": point_a.tolist(),
        "point_b": point_b.tolist(),
        "euclidean_distance": euclidean,
        "legacy_geodesic_distance": legacy_distance,
        "corrected_geodesic_distance": corrected_geodesic,
        "corrected_ratio": corrected_ratio,
        "path_grid_point_count": len(path),
        "rendered_polyline_length": path_length,
        "figure_files": figure_files,
        "provenance_file": str(provenance_path),
    }


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def _named_entry(entries: list[dict[str, Any]], name: str) -> dict[str, Any]:
    try:
        return next(entry for entry in entries if str(entry.get("name")) == name)
    except StopIteration as exc:
        raise KeyError(f"No entry named {name!r}.") from exc
