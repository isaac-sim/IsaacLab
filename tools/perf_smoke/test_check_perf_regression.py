# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ``check_perf_regression.py``.

Stdlib ``unittest`` (not pytest) because ``tools/conftest.py`` blocks pytest
collection under ``tools/``. Run directly:

.. code-block:: bash

    ./isaaclab.sh -p tools/perf_smoke/test_check_perf_regression.py
    # or
    python3 tools/perf_smoke/test_check_perf_regression.py
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_perf_regression as cpr  # noqa: E402

TASK = "Isaac-Cartpole-Direct-v0"
GPU = "NVIDIA L40"
BASELINE_FPS = 626772
THRESHOLD_PCT = 10.0


def _result_doc(fps: float | int | str | None, gpu: str | None = GPU) -> dict:
    """Build a minimal OmniPerf-shaped result document for tests."""
    doc: dict = {
        "runtime": {cpr.METRIC_NAME: fps},
    }
    if gpu is not None:
        doc["hardware_info"] = {
            "gpu_current_device": 0,
            "gpu_devices": {"0": {"name": gpu, "total_memory_gb": 44.5}},
        }
    return doc


def _baseline_doc(fps: int = BASELINE_FPS, threshold: float = THRESHOLD_PCT, gpu_key: str = GPU) -> dict:
    """Build a minimal baseline document for tests."""
    return {
        TASK: {
            "per_gpu": {
                gpu_key: {
                    "baseline_fps": fps,
                    "max_regression_pct": threshold,
                },
            },
        },
    }


class CheckPerfRegressionTests(unittest.TestCase):
    """End-to-end CLI behavior for the comparator."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.results_dir = self.tmp / "perf-output"
        self.results_dir.mkdir()
        self.baseline_path = self.tmp / "baseline.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_result(self, doc: dict | str, name: str | None = None) -> Path:
        """Write a result file. ``doc`` may be a dict or a raw string for malformed-JSON tests."""
        path = self.results_dir / (name or f"benchmark_non_rl_{TASK}_2026-05-22.json")
        if isinstance(doc, str):
            path.write_text(doc, encoding="utf-8")
        else:
            path.write_text(json.dumps(doc), encoding="utf-8")
        return path

    def _write_baseline(self, doc: dict) -> None:
        self.baseline_path.write_text(json.dumps(doc), encoding="utf-8")

    def _run(self, *extra: str) -> tuple[int, str]:
        """Invoke ``main`` with the standard arg set plus ``extra`` overrides."""
        argv = [
            "--task",
            TASK,
            "--results-dir",
            str(self.results_dir),
            "--baseline",
            str(self.baseline_path),
            *extra,
        ]
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cpr.main(argv)
        return code, buf.getvalue().strip()

    # ----- pass / regression / improvement -----

    def test_pass_within_threshold(self) -> None:
        # 5% drop on a 10% threshold -> PASS
        self._write_result(_result_doc(int(BASELINE_FPS * 0.95)))
        self._write_baseline(_baseline_doc())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_PASS)
        self.assertIn("RESULT=PASS", out)
        self.assertIn("delta_pct=-5.00", out)

    def test_pass_improvement(self) -> None:
        self._write_result(_result_doc(int(BASELINE_FPS * 1.08)))
        self._write_baseline(_baseline_doc())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_PASS)
        self.assertIn("delta_pct=+8.00", out)

    def test_pass_at_exact_threshold(self) -> None:
        # Exactly -10% on a 10% threshold -> PASS (boundary is inclusive of pass).
        # Use the float-valued FPS rather than int-truncating; truncation lands
        # us a fraction of a percent past -10% and would flip the result.
        self._write_result(_result_doc(BASELINE_FPS * 0.90))
        self._write_baseline(_baseline_doc())
        code, _ = self._run()
        self.assertEqual(code, cpr.EXIT_PASS)

    def test_regression_below_threshold(self) -> None:
        self._write_result(_result_doc(int(BASELINE_FPS * 0.85)))
        self._write_baseline(_baseline_doc())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_REGRESSION)
        self.assertIn("RESULT=REGRESSION", out)
        self.assertIn("delta_pct=-15.00", out)

    # ----- structural failures -> HARD_FAILURE -----

    def test_no_results_file(self) -> None:
        self._write_baseline(_baseline_doc())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("reason=no_results_found", out)

    def test_multiple_results_strict(self) -> None:
        self._write_result(_result_doc(BASELINE_FPS), name=f"benchmark_non_rl_{TASK}_a.json")
        self._write_result(_result_doc(BASELINE_FPS), name=f"benchmark_non_rl_{TASK}_b.json")
        self._write_baseline(_baseline_doc())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("multiple_results", out)

    def test_multiple_results_allow(self) -> None:
        self._write_result(_result_doc(BASELINE_FPS), name=f"benchmark_non_rl_{TASK}_a.json")
        self._write_result(_result_doc(BASELINE_FPS), name=f"benchmark_non_rl_{TASK}_b.json")
        self._write_baseline(_baseline_doc())
        code, _ = self._run("--allow-multiple")
        self.assertEqual(code, cpr.EXIT_PASS)

    def test_malformed_json(self) -> None:
        self._write_result("{not valid json")
        self._write_baseline(_baseline_doc())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("malformed_json", out)

    def test_missing_metric_phase(self) -> None:
        self._write_result({"hardware_info": {"gpu_devices": {"0": {"name": GPU}}}})
        self._write_baseline(_baseline_doc())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("missing_phase", out)

    def test_missing_metric_name(self) -> None:
        doc = _result_doc(BASELINE_FPS)
        doc["runtime"] = {"some other metric": 1.0}
        self._write_result(doc)
        self._write_baseline(_baseline_doc())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("missing_metric", out)

    def test_nan_metric(self) -> None:
        doc = _result_doc(BASELINE_FPS)
        doc["runtime"][cpr.METRIC_NAME] = float("nan")
        # JSON doesn't natively support NaN; write via the allow_nan default and confirm
        # the loader rejects it cleanly. We sidestep the dump by writing manually.
        path = self.results_dir / f"benchmark_non_rl_{TASK}.json"
        path.write_text(
            '{"runtime": {"' + cpr.METRIC_NAME + '": NaN}, '
            '"hardware_info": {"gpu_current_device": 0, "gpu_devices": {"0": {"name": "' + GPU + '"}}}}',
            encoding="utf-8",
        )
        self._write_baseline(_baseline_doc())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        # Python's json loader accepts NaN by default and yields float('nan'); the
        # comparator must catch it explicitly.
        self.assertTrue("nan_metric" in out or "malformed_json" in out)

    def test_zero_metric(self) -> None:
        self._write_result(_result_doc(0))
        self._write_baseline(_baseline_doc())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("non_positive_metric", out)

    def test_string_metric(self) -> None:
        self._write_result(_result_doc("not a number"))
        self._write_baseline(_baseline_doc())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("invalid_metric_type", out)

    def test_missing_baseline_task(self) -> None:
        self._write_result(_result_doc(BASELINE_FPS))
        self._write_baseline({"some-other-task": _baseline_doc()[TASK]})
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("missing_baseline_task", out)

    def test_missing_baseline_field(self) -> None:
        self._write_result(_result_doc(BASELINE_FPS))
        broken = _baseline_doc()
        del broken[TASK]["per_gpu"][GPU]["max_regression_pct"]
        self._write_baseline(broken)
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("missing_baseline_field", out)

    def test_baseline_gpu_mismatch(self) -> None:
        # Result reports an unknown GPU.
        self._write_result(_result_doc(BASELINE_FPS, gpu="NVIDIA Some Other GPU"))
        self._write_baseline(_baseline_doc())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("baseline_gpu_mismatch", out)

    def test_no_gpu_in_result_without_override(self) -> None:
        self._write_result(_result_doc(BASELINE_FPS, gpu=None))
        self._write_baseline(_baseline_doc())
        code, out = self._run()
        self.assertEqual(code, cpr.EXIT_HARD_FAILURE)
        self.assertIn("unknown_gpu", out)

    def test_gpu_override(self) -> None:
        self._write_result(_result_doc(BASELINE_FPS, gpu=None))
        self._write_baseline(_baseline_doc())
        code, _ = self._run("--gpu-override", GPU)
        self.assertEqual(code, cpr.EXIT_PASS)

    def test_gpu_substring_match(self) -> None:
        # Baseline keyed by a coarse tag matches a device whose name contains it.
        self._write_result(_result_doc(BASELINE_FPS, gpu="NVIDIA L40"))
        self._write_baseline(_baseline_doc(gpu_key="L40"))
        code, _ = self._run()
        self.assertEqual(code, cpr.EXIT_PASS)


class HelperFunctionTests(unittest.TestCase):
    """Direct unit tests for helper functions where edge cases are easier to express."""

    def test_extract_fps_rejects_bool(self) -> None:
        with self.assertRaises(cpr.CompareError):
            cpr._extract_fps({"runtime": {cpr.METRIC_NAME: True}})

    def test_extract_fps_accepts_int(self) -> None:
        self.assertEqual(cpr._extract_fps({"runtime": {cpr.METRIC_NAME: 1234}}), 1234.0)

    def test_extract_gpu_name_missing_phase(self) -> None:
        self.assertIsNone(cpr._extract_gpu_name({}))

    def test_extract_gpu_name_picks_current_device(self) -> None:
        doc = {
            "hardware_info": {
                "gpu_current_device": 1,
                "gpu_devices": {
                    "0": {"name": "GPU A"},
                    "1": {"name": "GPU B"},
                },
            },
        }
        self.assertEqual(cpr._extract_gpu_name(doc), "GPU B")

    def test_match_gpu_substring_either_direction(self) -> None:
        per_gpu = {"L40": {"baseline_fps": 1.0, "max_regression_pct": 1.0}}
        key, _ = cpr._match_gpu(per_gpu, "NVIDIA L40")
        self.assertEqual(key, "L40")

    def test_threshold_boundary_is_inclusive_pass(self) -> None:
        # Mirrors test_pass_at_exact_threshold but exercised via a math invariant.
        baseline = 1000.0
        threshold = 10.0
        measured = baseline * (1 - threshold / 100)
        delta = (measured - baseline) / baseline * 100
        self.assertTrue(math.isclose(delta, -threshold))
        # Comparator uses strict ``<`` so exactly -threshold passes.
        self.assertFalse(delta < -threshold)


if __name__ == "__main__":
    unittest.main(verbosity=2)
