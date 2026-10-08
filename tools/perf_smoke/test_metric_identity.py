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

from .build_compare import compare_evidence
from .metric_identity import PACKAGE, _normalized, metric_definition
from .report import render_build_comparison
from .test_build_compare import bundle, evidence

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
        # Keep the same fingerprint so previously recorded results remain comparable.
        self.assertEqual(
            original["total_fps"], "source-fps-v2:6f3a5c2eed5e5203cd671e6f368b237adac846d6a2e419fa3075418ed8f3b85e"
        )
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
            (
                self.runtime,
                RUNTIME,
                RUNTIME,
                RUNTIME.replace("    startup =", "    metadata = describe(**settings)\n    startup =")
                + "\ndef describe(**values):\n    return dict(**values)\n",
            ),
            (
                self.runtime,
                RUNTIME,
                RUNTIME,
                RUNTIME.replace("    startup =", '    metadata = describe(name="status")\n    startup =')
                + "\ndef describe(**values):\n    from unrelated_diagnostics import builders\n"
                + "    return builders.build_runtime(**values)\n",
            ),
            (
                self.runtime,
                RUNTIME,
                RUNTIME,
                RUNTIME.replace("    startup =", '    metadata = describe(name="status")\n    startup =')
                + "\ndef describe(**values):\n    import unrelated_diagnostics as builders\n"
                + "    return builders.build_runtime(**values)\n",
            ),
            (
                self.runtime,
                RUNTIME,
                "    startup =",
                "    def describe(**values):\n        from unrelated_diagnostics import builders\n"
                '        return builders.build_runtime(**values)\n    metadata = describe(name="status")\n'
                "    startup =",
            ),
            (
                self.runtime,
                RUNTIME,
                "    startup =",
                "    def describe(**values):\n        import unrelated_diagnostics as builders\n"
                '        return builders.build_runtime(**values)\n    metadata = describe(name="status")\n'
                "    startup =",
            ),
            (
                self.runtime,
                RUNTIME,
                "    startup =",
                "    from isaaclab.benchmark.builders import build_runtime, build_run_identity\n"
                "    metadata = build_run_identity(**identity_fields)\n    startup =",
            ),
            (self.builders, BUILDERS, "Startup timing is missing", "Provide startup timing"),
            (self.builders, BUILDERS, "startup_time_s=startup_time_s", "startup_time_s=startup_time_s * 1000"),
        ):
            with self.subTest(change=new):
                try:
                    path.write_text(source.replace(old, new))
                    self.assertEqual(self.identity(), original)
                finally:
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
            (self.builders, BUILDERS, "    value =", "    total_fps.append(1.0)\n    value ="),
        ):
            with self.subTest(path=path, change=new):
                path.write_text(source.replace(old, new))
                self.assertIsNotNone(self.identity())
                self.assertNotEqual(self.identity(), original)
                path.write_text(source)
        self.builders.write_text(BUILDERS.replace("return 10", "return 20"))
        self.assertEqual(self.identity(), original)

    def test_equivalent_producer_refactors_keep_percentage_comparisons(self):
        helper = (
            RUNTIME.replace("return builders.build_runtime(", "return finish(")
            + """
def finish(**values):
    from isaaclab.benchmark import builders
    return builders.build_runtime(**values)
"""
        )
        keywords = (
            RUNTIME.replace("return builders.build_runtime(", "kwargs = dict(")
            + "    return builders.build_runtime(**kwargs)\n"
        )
        positional = (
            RUNTIME.replace("return builders.build_runtime(", "return finish(")
            + """
def finish(startup_time_s, iteration_times_s, total_fps, steps_per_iteration, aggregate_throughput=True):
    from isaaclab.benchmark import builders
    return builders.build_runtime(
        startup_time_s=startup_time_s, iteration_times_s=iteration_times_s, total_fps=total_fps,
        steps_per_iteration=steps_per_iteration, aggregate_throughput=aggregate_throughput,
    )
"""
        )
        positional = positional.replace("finish(\n        startup_time_s=startup,", "finish(\n        startup,")
        early = BUILDERS.replace(
            "        value = _effective(steps_per_iteration / sum(iteration_times_s))",
            "        value = _effective(steps_per_iteration / sum(iteration_times_s))\n"
            "        return Runtime(total_fps=value, startup_time_s=startup_time_s)",
        )
        guard = BUILDERS.replace(
            "    if aggregate_throughput:\n        value = _effective(steps_per_iteration / sum(iteration_times_s))",
            "    if not aggregate_throughput:\n        return Runtime(total_fps=value, startup_time_s=startup_time_s)\n"
            "    value = _effective(steps_per_iteration / sum(iteration_times_s))",
        )
        for label, runtime, builders, aggregate in (
            ("early aggregate return", RUNTIME, early, True),
            ("early fallback return", RUNTIME, early, False),
            ("guard before aggregate calculation", RUNTIME, guard, True),
            ("guard takes fallback return", RUNTIME, guard, False),
            (
                "early result expression",
                RUNTIME,
                BUILDERS.replace(
                    "value = _effective(steps_per_iteration / sum(iteration_times_s))",
                    "return Runtime(total_fps=_effective(steps_per_iteration / sum(iteration_times_s)))",
                ),
                True,
            ),
            ("forwarded keywords", helper, BUILDERS, True),
            ("helper parameters", positional, BUILDERS, True),
            ("explicit keyword dictionary", keywords, BUILDERS, True),
        ):
            with self.subTest(refactor=label):
                direct = RUNTIME.replace("aggregate_throughput=True", f"aggregate_throughput={aggregate}")
                runtime = runtime.replace("aggregate_throughput=True", f"aggregate_throughput={aggregate}")
                self.runtime.write_text(direct)
                self.builders.write_text(BUILDERS)
                original = self.identity()
                self.assertIsNotNone(original)
                self.runtime.write_text(runtime)
                self.builders.write_text(builders)
                self.assertEqual(self.identity(), original)
                baseline = evidence({"leg": [bundle(100)] * 3}, formula=original)
                candidate = evidence({"leg": [bundle(110)] * 3}, formula=self.identity())
                report = compare_evidence(baseline, candidate)
                self.assertEqual(report["rows"][0]["status"], "compared")
                self.assertEqual(report["rows"][0]["change_pct"], 10)
                self.assertIn("+10.00%", render_build_comparison(report))
                changed = builders.replace("steps_per_iteration /", "steps_per_iteration *")
                if not aggregate:
                    changed = builders.replace("mean_std_peak(total_fps)", "mean_std_peak([v * 2 for v in total_fps])")
                self.builders.write_text(changed)
                self.assertIsNotNone(self.identity())
                self.assertNotEqual(self.identity(), original)
                candidate.context["metric_definition"]["total_fps"] = self.identity()
                row = compare_evidence(baseline, candidate)["rows"][0]
                self.assertEqual(row["status"], "incompatible")
                self.assertIsNone(row["change_pct"])
                self.builders.write_text(builders)
                self.stepping.write_text(
                    STEPPING.replace("time.perf_counter() - start", "(time.perf_counter() - start) * 2")
                )
                self.assertIsNotNone(self.identity())
                self.assertNotEqual(self.identity(), original)
                self.stepping.write_text(STEPPING)

        nested = BUILDERS.replace(
            "        value = _effective(steps_per_iteration / sum(iteration_times_s))",
            "        if iteration_times_s:\n            value = _effective(value / sum(iteration_times_s))",
        )
        guard = nested.replace(
            "    if aggregate_throughput:\n        if iteration_times_s:\n            value =",
            "    if not aggregate_throughput:\n        return Runtime(total_fps=value)\n"
            "    if iteration_times_s:\n        value =",
        )
        for aggregate in (True, False):
            with self.subTest(nested_guard=aggregate):
                self.runtime.write_text(
                    RUNTIME.replace("aggregate_throughput=True", f"aggregate_throughput={aggregate}")
                )
                self.builders.write_text(nested)
                original = self.identity()
                self.assertIsNotNone(original)
                self.builders.write_text(guard)
                self.assertEqual(self.identity(), original)

    def test_unknown_producer_shapes_and_syntax_keep_readable_reason(self):
        mapping = RUNTIME.replace("return builders.build_runtime(", "kwargs = dict(")
        helper = (
            RUNTIME.replace("return builders.build_runtime(", "return finish(")
            + """
def finish(**values):
    from isaaclab.benchmark import builders
    return builders.build_runtime(**values)
"""
        )
        preview = RUNTIME.replace("return builders.build_runtime(", "preview = builders.build_runtime(")
        producer = RUNTIME[RUNTIME.index("    return builders.build_runtime(") :].replace(
            "steps_per_iteration=num_envs", "steps_per_iteration=num_envs * 2"
        )
        nested = (
            preview
            + "    def finalize():\n"
            + "\n".join("    " + line for line in producer.splitlines())
            + "\n    return finalize()\n"
        )
        nested_helper = (
            preview
            + producer.replace("builders.build_runtime(", "finish(")
            + "\ndef finish(**values):\n    from isaaclab.benchmark import builders\n"
            + "    def finalize():\n        return builders.build_runtime(**values)\n    return finalize()\n"
        )
        for source in (
            mapping
            + '    alias = kwargs\n    alias["steps_per_iteration"] *= 2\n'
            + "    return builders.build_runtime(**kwargs)\n",
            mapping + '    kwargs["steps_per_iteration"] *= 2\n    return builders.build_runtime(**kwargs)\n',
            mapping + "    num_envs *= 2\n    return builders.build_runtime(**kwargs)\n",
            mapping.replace("steps_per_iteration=num_envs", "steps_per_iteration=next_value()")
            + "    ignored = next_value()\n    return builders.build_runtime(**kwargs)\n",
            helper.replace("def finish(", "@scale_result\ndef finish("),
            helper + "\nfinish = scale_result(finish)\n",
            helper.replace("def run(argv):", "def run(argv, finish):"),
            nested,
            "from unrelated_diagnostics import builders\n" + nested,
            preview
            + producer.replace("builders.build_runtime(", "finish(builders,")
            + "\ndef finish(builders, **values):\n    return builders.build_runtime(**values)\n",
            preview
            + "    from isaaclab.benchmark import builders as b\n"
            + producer.replace("builders.build_runtime(", "b.build_runtime("),
            preview
            + "    from isaaclab.benchmark.builders import build_runtime as produce\n"
            + producer.replace("builders.build_runtime(", "produce("),
            preview
            + producer.replace("builders.build_runtime(", "finish(")
            + "\ndef finish(**values):\n    from isaaclab.benchmark import builders\n"
            + '    values["steps_per_iteration"] *= 2\n    return builders.build_runtime(**values)\n',
            nested_helper,
            "from unrelated_diagnostics import builders\n" + nested_helper,
            preview
            + producer.replace("builders.build_runtime(", "finish(")
            + "\ndef _keep(value):\n    return value\n\ndef finish(**values):\n"
            + "    from isaaclab.benchmark import builders\n    return _keep(builders.build_runtime(**values))\n",
            RUNTIME.replace("return builders.build_runtime(", "return finish(other_builders,")
            + "\ndef finish(builders, **values):\n    return builders.build_runtime(**values)\n",
            RUNTIME.replace("from isaaclab.benchmark", "from unrelated_diagnostics")
            + "\ndef unused():\n    from isaaclab.benchmark import builders\n",
            "from unrelated_diagnostics import builders\n"
            + helper.replace(
                "    from isaaclab.benchmark import builders\n    return builders.build_runtime(**values)",
                "    return builders.build_runtime(**values)",
            ),
            helper.replace(
                "    from isaaclab.benchmark import builders\n    return builders.build_runtime(**values)",
                "    return builders.build_runtime(**values)",
            ),
            "dict = custom_mapping\n" + mapping + "    return builders.build_runtime(**kwargs)\n",
            helper.replace(
                "finish(\n        startup_time_s=startup", "finish(\n        startup_time_s=prepare_startup()"
            ),
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

        prefix = """SCALE = 2
def mutate():
    global SCALE
    SCALE = 100
    return 0

def run(argv):
    from isaaclab.benchmark import builders
    times = [2.0]
    fps = [1.0]
"""
        producer = (
            "builders.build_runtime(startup_time_s=mutate(), iteration_times_s=times, total_fps=fps, "
            "steps_per_iteration=SCALE, aggregate_throughput=True)"
        )
        self.runtime.write_text(prefix + "    return " + producer + "\n")
        direct = self.identity()
        self.assertIsNotNone(direct)
        self.runtime.write_text(
            prefix + "    return finish(SCALE, times, fps)\n\ndef finish(scale, times, fps):\n"
            "    from isaaclab.benchmark import builders\n    return "
            + producer.replace("steps_per_iteration=SCALE", "steps_per_iteration=scale")
            + "\n"
        )
        result = metric_definition(self.root)
        self.assertIsNone(result["total_fps"])
        self.assertIn("captured", result["reason"])
        row = compare_evidence(
            evidence({"leg": [bundle(50)] * 3}, formula=direct),
            evidence({"leg": [bundle(1)] * 3}, formula=result["total_fps"]),
        )["rows"][0]
        self.assertEqual(row["status"], "unknown")
        self.assertIsNone(row["change_pct"])

        self.runtime.write_text(RUNTIME)
        self.builders.write_text(
            BUILDERS.replace(
                "    if startup_time_s is None:",
                "    if aggregate_throughput:\n        return Runtime(total_fps=value)\n"
                "    alias = value\n    alias.mean *= 2\n    if startup_time_s is None:",
            )
        )
        result = metric_definition(self.root)
        self.assertIsNone(result["total_fps"])
        self.assertIn("early FPS result", result["reason"])

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
