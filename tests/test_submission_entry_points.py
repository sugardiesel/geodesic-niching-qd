import ast
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts import prepare_report_inputs as report_inputs
from scripts import run_phase5_sweep as sweep

ROOT = Path(__file__).resolve().parents[1]


def argument_default(script, flag):
    tree = ast.parse((ROOT / "scripts" / script).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == flag
        ):
            return ast.literal_eval(next(k.value for k in node.keywords if k.arg == "default"))
    raise AssertionError(f"Missing default: {script} {flag}")


class SubmissionInputTests(unittest.TestCase):
    def test_single_run_diagnostics_keep_outputs_outside_saved_results(self):
        for script in ("phase3_latent_diagnostic.py", "phase3_knn_k_sensitivity.py",
                       "phase4_dynamic_graph_k_sensitivity.py",
                       "validate_phase4_geodesic_approximation.py"):
            path = Path(argument_default(script, "--output-dir"))
            with self.subTest(script=script):
                self.assertEqual(path.parts[0], "reproduced")

    def test_latent_diagnostic_defaults_have_checkpoint_sample_and_summary(self):
        path = ROOT / argument_default("phase3_latent_diagnostic.py", "--phase3-config")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        run = ROOT / config["experiment"]["output_dir"].replace("\\", "/")
        for name in ("autoencoder_final.pt", "representative_trajectory_sample.npz",
                     "phase3_summary.json"):
            with self.subTest(name=name):
                self.assertTrue((run / name).is_file())

    def test_static_graph_default_sample_is_included(self):
        path = ROOT / argument_default("phase3_knn_k_sensitivity.py", "--sample-npz")
        self.assertTrue(path.is_file())

    def test_dynamic_and_approximation_defaults_have_saved_inputs(self):
        for script in ("phase4_dynamic_graph_k_sensitivity.py",
                       "validate_phase4_geodesic_approximation.py"):
            run = ROOT / argument_default(script, "--phase4-dir")
            config = ROOT / argument_default(script, "--config")
            self.assertTrue(config.is_file())
            for name in ("archive_cells.csv", "evaluations.csv", "visited_latents_final_space.csv",
                         "geodesic_graph_support_points.csv"):
                with self.subTest(script=script, name=name):
                    self.assertTrue((run / name).is_file())

    def test_historical_summary_inputs_match_their_manifest(self):
        directory = ROOT / "results/audit_fix/pre_fix_reference"
        with (directory / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 12)
        for row in rows:
            with self.subTest(file=row["file"]):
                digest = hashlib.sha256((directory / row["file"]).read_bytes()).hexdigest()
                self.assertEqual(digest, row["sha256"])


class SweepOutputTests(unittest.TestCase):
    def protocol(self):
        return {
            "production_budget": {
                "seeds": [1001],
                "total_evaluations_per_seed": 20000,
                "conditions": ["baseline_b_learned_bd_euclidean"],
            },
            "food_respawn_timing": "historical",
        }

    def test_building_specs_does_not_create_output_files(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "new_run"
            specs = sweep.build_run_specs(self.protocol(), output)
            self.assertEqual(len(specs), 1)
            self.assertFalse(output.exists())

    def test_aggregate_only_preserves_saved_configs(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            spec = sweep.build_run_specs(self.protocol(), output)[0]
            path = spec["config_path"]
            path.parent.mkdir(parents=True)
            saved = b"experiment: {food_respawn_timing: historical}\n# saved run\n"
            path.write_bytes(saved)
            protocol = output / "protocol.yaml"
            protocol.write_text(yaml.safe_dump(self.protocol()), encoding="utf-8")
            argv = ["sweep", "--protocol", str(protocol), "--output-dir", str(output),
                    "--aggregate-only"]
            with (
                patch.object(sys, "argv", argv),
                patch.object(sweep, "aggregate_phase5") as aggregate,
                patch.object(sweep, "run_condition_seed") as search,
            ):
                sweep.main()
                aggregate.assert_called_once()
                search.assert_not_called()
            self.assertEqual(path.read_bytes(), saved)

    def test_resume_preserves_config_on_skip_and_rejection(self):
        for timing in (None, "corrected", "historical"):
            with self.subTest(timing=timing), tempfile.TemporaryDirectory() as temp:
                spec = sweep.build_run_specs(self.protocol(), Path(temp))[0]
                path = spec["config_path"]
                path.parent.mkdir(parents=True)
                saved = b"original: saved-configuration\n"
                path.write_bytes(saved)
                spec["summary_path"].parent.mkdir(parents=True)
                spec["summary_path"].write_text(json.dumps({"food_respawn_timing": timing}))
                with patch.object(sweep.subprocess, "Popen") as process:
                    if timing == "historical":
                        sweep.run_condition_seed(spec, resume=True)
                    else:
                        with self.assertRaisesRegex(ValueError, "Cannot resume"):
                            sweep.run_condition_seed(spec, resume=True)
                    process.assert_not_called()
                self.assertEqual(path.read_bytes(), saved)

    def test_new_run_writes_requested_config_before_starting(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = sweep.build_run_specs(self.protocol(), Path(temp))[0]
            with patch.object(sweep.subprocess, "Popen", side_effect=RuntimeError("before search")):
                with self.assertRaisesRegex(RuntimeError, "before search"):
                    sweep.run_condition_seed(spec, resume=False)
            actual = yaml.safe_load(spec["config_path"].read_text(encoding="utf-8"))
            self.assertEqual(actual, spec["config"])


class ArtifactManifestTests(unittest.TestCase):
    def test_preparation_retains_existing_diagnostic_entries_without_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            for name in ("summary.csv", "diagnostic.csv"):
                (output / name).write_text("test data")
            def row(name, source):
                return {"category": "data", "file": name, "source": source, "purpose": "test"}
            report_inputs.write_csv(output / "artifact_manifest.csv", [
                row("summary.csv", "old"), row("diagnostic.csv", "diagnostic script"),
                row("diagnostic.csv", "duplicate"), row("missing.csv", "removed"),
            ])
            with patch.object(report_inputs, "OUTPUT_DIR", output):
                rows = [row("summary.csv", "current")]
                report_inputs.write_manifest(rows)
                report_inputs.write_manifest([row("summary.csv", "current")])
            self.assertEqual(report_inputs.read_csv(output / "artifact_manifest.csv"), [
                row("summary.csv", "current"), row("diagnostic.csv", "diagnostic script"),
            ])
