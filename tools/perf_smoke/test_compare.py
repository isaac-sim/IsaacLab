# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Host-only integration tests for performance smoke comparisons."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import cli, compare
from .contract import Contract
from .metrics import METRICS, PerfSmokeError
from .store import BaselineRow


def _measurement(fps: float, startup: float = 10.0) -> dict[str, float]:
    return {"total_fps": fps, "startup_time_s": startup, "gpu_mem_peak_gb": 2.0, "ram_peak_gb": 4.0}


def _runtime_bundle(fps: float, *, task: str = "task") -> dict:
    return {
        "run": {
            "task": task,
            "num_envs": 8,
            "config": {"physics_backend": "physx", "rendering_backend": "none", "presets": []},
        },
        "versions": {"torch": "2.0", "warp": "1.0", "isaacsim": "6.0"},
        "hardware": {"gpu_devices": [{"name": "NVIDIA L40S"}], "cpu_name": "CPU"},
        "runtime": {
            "total_fps": {"mean": fps},
            "startup_time_s": {"app_launch": 1, "env_creation": 2, "first_step": 3},
        },
        "resources": {"gpu_mem_gb": {"peak": 2}, "ram_gb": {"peak": 4}},
    }


def _write_pair_leg(
    root: Path,
    leg: str,
    values: list[float],
    *,
    task: str = "task",
) -> None:
    leg_dir = root / leg
    leg_dir.mkdir(parents=True)
    (leg_dir / "status").write_text("ok\n", encoding="utf-8")
    for index, value in enumerate(values, 1):
        sample_dir = leg_dir / f"sample-{index}"
        sample_dir.mkdir()
        (sample_dir / f"benchmark_runtime_{index}.json").write_text(
            json.dumps(_runtime_bundle(value, task=task)), encoding="utf-8"
        )


class TestAsvComparison(unittest.TestCase):
    """Check the pinned ASV comparison and the policies retained around it."""

    def setUp(self):
        self.contract = Contract(
            workload={"task": "task", "physics_backend": "physx", "render_backend": None},
            runtime={"gpu_model": "l40s"},
        )
        self.policy = {"defaults": {"warn_regression_pct": 5, "fail_regression_pct": 10}}

    def evaluate(self, baseline, candidate):
        measurements = [_measurement(value) for value in candidate]
        return compare.compare(
            self.contract,
            measurements,
            [_measurement(value) for value in baseline],
            self.policy,
        )

    def test_relative_regressions_and_improvement(self):
        # Three baseline samples exercise the CI fallback; twenty permit ASV's
        # Mann–Whitney test. FPS equality also checks the threshold factor.
        for count in (3, 20):
            for fps, expected in (
                (100, compare.PASS),
                (120, compare.PASS),
                (95, compare.PASS),
                (93, compare.WARN),
                (90, compare.WARN),
                (80, compare.FAIL),
            ):
                with self.subTest(count=count, fps=fps):
                    result = self.evaluate([100] * count, [fps] * 3)
                    self.assertEqual(result.verdict, expected)

        for baseline, candidate, expected in (
            ([60, 100, 140], [50, 80, 110], compare.PASS),
            # An even-sample median must not change when reversing FPS comparisons.
            ([99] * 10 + [101] * 10, [89.995] * 3, compare.FAIL),
            ([100] * 3, [0] * 3, compare.FAIL),
            ([0] * 3, [100] * 3, compare.SKIP),
        ):
            with self.subTest(baseline=baseline, candidate=candidate):
                self.assertEqual(self.evaluate(baseline, candidate).verdict, expected)

    def test_insufficient_independent_samples_skip(self):
        for baseline, candidate in (([], [80] * 3), ([100] * 2, [80] * 3), ([100] * 3, [80])):
            with self.subTest(baseline=baseline, candidate=candidate):
                self.assertEqual(self.evaluate(baseline, candidate).verdict, compare.SKIP)

    def test_hard_floor_gates_without_history_even_for_advisory_task(self):
        self.policy.update(
            {
                "per_task_regression_pct": {"task": {"advisory_only": True}},
                "hard_floor_fps": {"l40s": {"task": {"physx": 50}}},
            }
        )
        result = self.evaluate([], [0, 100, 100])
        self.assertEqual(result.verdict, compare.FAIL)
        self.assertIn("hard floor", result.metrics[0].note)
        self.assertIn("hard floor", result.message)

    def test_hard_floor_respects_metric_direction(self):
        thresholds = compare.Thresholds(warn_pct=5, fail_pct=10, hard_floor=15, gating=False)
        breached = compare._evaluate(METRICS[1], [10, 20, 10], [], thresholds, min_samples=3)
        within_limit = compare._evaluate(METRICS[1], [10, 10, 10], [], thresholds, min_samples=3)

        self.assertEqual(breached.verdict, compare.FAIL)
        self.assertTrue(breached.gating)
        self.assertEqual(breached.note, "above hard floor 15")
        self.assertEqual(within_limit.verdict, compare.SKIP)
        self.assertFalse(within_limit.gating)

    def test_advisory_task_does_not_gate(self):
        self.policy["per_task_regression_pct"] = {"task": {"advisory_only": True}}
        result = self.evaluate([100] * 3, [80] * 3)
        self.assertEqual(result.verdict, compare.SKIP)
        self.assertEqual(result.metrics[0].verdict, compare.FAIL)

    def test_higher_is_worse_metrics_are_advisory(self):
        measurements = [_measurement(100, startup=20)] * 3
        result = compare.compare(
            self.contract,
            measurements,
            [_measurement(100)] * 3,
            self.policy,
        )
        self.assertEqual(result.verdict, compare.PASS)
        self.assertEqual(result.metrics[1].verdict, compare.FAIL)

    def test_cli_filters_contracts_and_preserves_failure_in_aggregate(self):
        matching = BaselineRow(self.contract.as_dict(), self.contract.hash, _measurement(100), "commit", "date", "run")
        other = Contract(workload={"task": "other"}, runtime={})
        mismatched = BaselineRow(other.as_dict(), self.contract.hash, _measurement(1), "commit", "date", "run")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle.json"
            bundle.write_text("{}")
            output = root / "combination" / "comparison.json"
            with (
                contextlib.redirect_stdout(io.StringIO()),
                patch.object(cli.contract_mod, "build", return_value=self.contract),
                patch.object(cli.metrics_mod, "extract", return_value=_measurement(80)),
                patch.object(cli.store_mod, "is_configured", return_value=True),
                patch.object(cli.store_mod, "read", return_value=[matching] * 3 + [mismatched] * 10),
            ):
                status = cli.main(
                    [
                        "compare",
                        "--benchmark_result",
                        str(bundle),
                        str(bundle),
                        str(bundle),
                        "--output_json",
                        str(output),
                    ]
                )
            self.assertEqual(status, 1)
            self.assertEqual(json.loads(output.read_text())["verdict"], compare.FAIL)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["aggregate", "--comparison_dir", directory]), 1)

    def test_cli_infrastructure_errors_and_missing_credentials_do_not_gate(self):
        for configured, expected in ((False, compare.SKIP), (True, compare.ERROR)):
            with self.subTest(configured=configured), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bundle = root / "bundle.json"
                bundle.write_text("{}")
                output = root / "comparison.json"
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                    patch.object(cli.contract_mod, "build", return_value=self.contract),
                    patch.object(cli.metrics_mod, "extract", return_value=_measurement(80)),
                    patch.object(cli.store_mod, "is_configured", return_value=configured),
                    patch.object(cli.store_mod, "read", side_effect=PerfSmokeError("unavailable")),
                ):
                    status = cli.main(
                        [
                            "compare",
                            "--benchmark_result",
                            str(bundle),
                            "--output_json",
                            str(output),
                        ]
                    )
                self.assertEqual(status, 0)
                self.assertEqual(json.loads(output.read_text())["verdict"], expected)

    def test_pair_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            candidate = root / "candidate"
            cases = {
                "changed": ([90, 100, 110], [99, 110, 121], "task"),
                "incomplete": ([90, 100], [90, 100, 110], "task"),
                "incompatible": ([100, 100, 100], [100, 100, 100], "other"),
            }
            for leg, (baseline_values, candidate_values, candidate_task) in cases.items():
                _write_pair_leg(baseline, leg, baseline_values)
                _write_pair_leg(candidate, leg, candidate_values, task=candidate_task)

            changed = cli._pair_row(baseline, candidate, "changed")
            self.assertEqual(
                (changed["status"], changed["baseline_fps"], changed["candidate_fps"], changed["change_pct"]),
                ("improved", 100, 110, 10),
            )
            self.assertEqual(
                cli._pair_row(baseline, candidate, "incomplete")["reason"],
                "baseline has 2 samples; expected 3",
            )
            self.assertEqual(
                cli._pair_row(baseline, candidate, "incompatible")["reason"],
                "baseline and candidate workloads differ",
            )


if __name__ == "__main__":
    unittest.main()
