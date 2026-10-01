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
    from isaaclab.benchmark import builders
    return builders.build_runtime(total_fps=fps, steps_per_iteration=num_envs, aggregate_throughput=True)
'''
BUILDERS = '''
from .metrics import mean_std_peak

def _effective(value):
    return value

def build_runtime(*, total_fps, steps_per_iteration, aggregate_throughput=False):
    """Builder documentation."""
    value = mean_std_peak(total_fps)
    if aggregate_throughput:
        value = _effective(steps_per_iteration / sum(total_fps))
    return value

def unrelated_training():
    return 10
'''
METRICS = '''
def mean_std_peak(values):
    """Metric documentation."""
    return sum(values) / len(values)
'''


class MetricIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / PACKAGE / "entrypoints").mkdir(parents=True)
        self.runtime = self.root / PACKAGE / "entrypoints/runtime.py"
        self.builders = self.root / PACKAGE / "builders.py"
        self.metrics = self.root / PACKAGE / "metrics.py"
        self.runtime.write_text(RUNTIME)
        self.builders.write_text(BUILDERS)
        self.metrics.write_text(METRICS)

    def identity(self):
        return metric_definition(self.root)["total_fps"]

    def test_identical_source_and_docstring_only_changes_share_identity(self):
        original = metric_definition(self.root)
        self.assertTrue(original["total_fps"].startswith("source-fps-v1:"))
        self.assertTrue(original["aggregate_throughput"])
        self.runtime.write_text("\n\n# Comment\n" + RUNTIME.replace("Runtime documentation.", "Canary B."))
        self.builders.write_text(BUILDERS.replace("Builder documentation.", "Updated documentation."))
        self.metrics.write_text(METRICS.replace("Metric documentation.", "Updated documentation."))
        self.assertEqual(self.identity(), original["total_fps"])

    def test_literal_and_default_aggregation_settings_are_distinguished(self):
        aggregate = self.identity()
        self.runtime.write_text(RUNTIME.replace("aggregate_throughput=True", "aggregate_throughput=False"))
        arithmetic = self.identity()
        self.assertNotEqual(aggregate, arithmetic)
        self.runtime.write_text(RUNTIME.replace(", aggregate_throughput=True", ""))
        self.assertEqual(self.identity(), arithmetic)
        self.builders.write_text(BUILDERS.replace("aggregate_throughput=False", "aggregate_throughput=True"))
        self.assertNotEqual(self.identity(), arithmetic)

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
