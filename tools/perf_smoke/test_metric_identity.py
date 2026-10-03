# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Source-equivalence checks for FPS aggregation, without simulation imports."""

import ast
import copy
import tempfile
import unittest
from pathlib import Path

from .metric_identity import PACKAGE, _normalized, metric_definition

RUNTIME = '''
def run(argv):
    """Runtime documentation."""
    from isaaclab.benchmark import builders, stepping
    num_envs = env.num_envs
    step_times = stepping.run_runtime_loop(env, args.num_steps)
    fps = [num_envs / t for t in step_times]
    startup = 1.0
    return builders.build_runtime(
        startup_time_s=startup, iteration_times_s=step_times, total_fps=fps,
        steps_per_iteration=num_envs, aggregate_throughput=True,
    )
'''
BUILDERS = '''
from .metrics import mean_std_peak

def _effective(value):
    return value

def build_runtime(
    *, total_fps, iteration_times_s, steps_per_iteration, aggregate_throughput=False, startup_time_s=None,
):
    """Builder documentation."""
    value = mean_std_peak(total_fps)
    if aggregate_throughput:
        value = _effective(steps_per_iteration / sum(iteration_times_s))
    if startup_time_s is None:
        raise ValueError("Startup timing is missing")
    return Runtime(total_fps=value, startup_time_s=startup_time_s)

def unrelated_training():
    return 10
'''
METRICS = '''
def mean_std_peak(values):
    """Metric documentation."""
    return sum(values) / len(values)
'''
STEPPING = """
import time

def run_runtime_loop(env, num_steps):
    times = []
    for _ in range(num_steps):
        start = time.perf_counter()
        env.step()
        times.append(time.perf_counter() - start)
    return times
"""


class MetricIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / PACKAGE / "entrypoints").mkdir(parents=True)
        self.runtime = self.root / PACKAGE / "entrypoints/runtime.py"
        self.builders = self.root / PACKAGE / "builders.py"
        self.metrics = self.root / PACKAGE / "metrics.py"
        self.stepping = self.root / PACKAGE / "stepping.py"
        self.runtime.write_text(RUNTIME)
        self.builders.write_text(BUILDERS)
        self.metrics.write_text(METRICS)
        self.stepping.write_text(STEPPING)

    def identity(self):
        return metric_definition(self.root)["total_fps"]

    def test_identical_source_and_docstring_only_changes_share_identity(self):
        original = metric_definition(self.root)
        self.assertTrue(original["total_fps"].startswith("source-fps-v2:"))
        self.assertTrue(original["aggregate_throughput"])
        self.runtime.write_text("\n\n# Comment\n" + RUNTIME.replace("Runtime documentation.", "Canary B."))
        self.builders.write_text(BUILDERS.replace("Builder documentation.", "Updated documentation."))
        self.metrics.write_text(METRICS.replace("Metric documentation.", "Updated documentation."))
        self.assertEqual(self.identity(), original["total_fps"])

    def test_startup_and_independent_diagnostic_edits_preserve_fps_comparison(self):
        original = self.identity()
        for path, source, old, new in (
            (self.runtime, RUNTIME, "startup = 1.0", "startup = 2.0"),
            (self.runtime, RUNTIME, "startup_time_s=startup", "startup_time_s=startup * 1000"),
            (self.builders, BUILDERS, "Startup timing is missing", "Provide startup timing"),
            (self.builders, BUILDERS, "startup_time_s=startup_time_s", "startup_time_s=startup_time_s * 1000"),
        ):
            with self.subTest(change=new):
                path.write_text(source.replace(old, new))
                self.assertEqual(self.identity(), original)
                path.write_text(source)

    def test_changed_timing_inputs_and_measurement_boundary_change_fps_identity(self):
        original = self.identity()
        for path, source, old, new in (
            (
                self.runtime,
                RUNTIME,
                "step_times = stepping.run_runtime_loop(env, args.num_steps)",
                "step_times = [t * 2 for t in stepping.run_runtime_loop(env, args.num_steps)]",
            ),
            (self.runtime, RUNTIME, "fps = [num_envs", "step_times *= 2\n    fps = [num_envs"),
            (self.stepping, STEPPING, "time.perf_counter() - start", "(time.perf_counter() - start) / 1000"),
            (self.stepping, STEPPING, "        env.step()", "        env.step()\n        env.synchronize()"),
        ):
            with self.subTest(change=new):
                path.write_text(source.replace(old, new))
                changed = self.identity()
                self.assertIsNotNone(changed)
                self.assertNotEqual(changed, original)
                path.write_text(source)

        conditional = RUNTIME.replace("    fps =", "    if args.scaled:\n        step_times *= 2\n    fps =")
        self.runtime.write_text(conditional)
        scaled = self.identity()
        self.runtime.write_text(conditional.replace("if args.scaled:", "if not args.scaled:"))
        self.assertIsNotNone(self.identity())
        self.assertNotEqual(self.identity(), scaled)

    def test_literal_and_default_aggregation_settings_are_distinguished(self):
        aggregate = self.identity()
        self.runtime.write_text(RUNTIME.replace("aggregate_throughput=True", "aggregate_throughput=False"))
        arithmetic = self.identity()
        self.assertNotEqual(aggregate, arithmetic)
        self.runtime.write_text(RUNTIME.replace(", aggregate_throughput=True", ""))
        self.assertEqual(self.identity(), arithmetic)
        self.builders.write_text(BUILDERS.replace("aggregate_throughput=False", "aggregate_throughput=True"))
        self.assertEqual(self.identity(), aggregate)

    def test_runtime_argument_helpers_include_their_constants(self):
        source = RUNTIME.replace("iteration_times_s=step_times", "iteration_times_s=scale(step_times)")
        source += "\nSCALE = 1\ndef scale(values):\n    return [value * SCALE for value in values]\n"
        self.runtime.write_text(source)
        original = self.identity()
        self.runtime.write_text(source.replace("SCALE = 1", "SCALE = 2"))
        self.assertIsNotNone(original)
        self.assertIsNotNone(self.identity())
        self.assertNotEqual(self.identity(), original)

    def test_builder_and_referenced_helper_changes_change_identity(self):
        original = self.identity()
        for path, source, old, new in (
            (self.builders, BUILDERS, "steps_per_iteration /", "steps_per_iteration *"),
            (self.builders, BUILDERS, "return value\n", "return value * 2\n"),
            (self.metrics, METRICS, "sum(values)", "max(values)"),
        ):
            with self.subTest(path=path, change=new):
                path.write_text(source.replace(old, new))
                self.assertNotEqual(self.identity(), original)
                path.write_text(source)
        self.builders.write_text(BUILDERS.replace("return 10", "return 20"))
        self.assertEqual(self.identity(), original)

    def test_unknown_producer_shapes_and_syntax_keep_readable_reason(self):
        for source in (
            RUNTIME.replace("aggregate_throughput=True", "aggregate_throughput=setting"),
            RUNTIME.replace("aggregate_throughput=True", "**settings"),
            RUNTIME.replace("builders.build_runtime", "other.build_runtime"),
            RUNTIME.replace("from isaaclab.benchmark", "from another.benchmark"),
            "def run(:\n",
        ):
            with self.subTest(source=source):
                self.runtime.write_text(source)
                result = metric_definition(self.root)
                self.assertIsNone(result["total_fps"])
                self.assertIn("unknown:", result["reason"])

    def test_missing_referenced_metric_implementation_is_unknown(self):
        self.metrics.unlink()
        result = metric_definition(self.root)
        self.assertIsNone(result["total_fps"])
        self.assertIn("metrics.py", result["reason"])

    def test_referenced_module_constant_changes_are_not_equivalent(self):
        self.builders.write_text("SCALE = 1\n" + BUILDERS.replace("return value\n", "return value * SCALE\n"))
        original = self.identity()
        self.builders.write_text(self.builders.read_text().replace("SCALE = 1", "SCALE = 2"))
        self.assertNotEqual(self.identity(), original)

    def test_empty_type_parameters_match_older_python_ast_without_erasing_generics(self):
        legacy = ast.parse("def helper(value):\n    return value\n").body[0]
        if hasattr(legacy, "type_params"):
            del legacy.type_params
        modern = copy.deepcopy(legacy)
        modern._fields = tuple(dict.fromkeys((*modern._fields, "type_params")))
        modern.type_params = []
        self.assertEqual(_normalized(legacy), _normalized(modern))
        modern.type_params = [ast.Name(id="T", ctx=ast.Load())]
        self.assertNotEqual(_normalized(legacy), _normalized(modern))

    def test_empty_argument_lists_remain_in_the_fingerprint_serialization(self):
        node = ast.parse("producer()", mode="eval").body
        self.assertEqual(_normalized(node), "Call(func=Name(id='producer', ctx=Load()), args=[], keywords=[])")


if __name__ == "__main__":
    unittest.main()
