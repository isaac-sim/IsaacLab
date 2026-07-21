# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-safe checks for the PhysX articulation selector benchmark grid."""

from __future__ import annotations

import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.test.mock_interfaces import MockArticulation
from isaaclab.utils.warp import ProxyArray

_ITEM_BENCHMARK_NAMES = {
    "write_joint_state_to_sim",
    "write_joint_position_to_sim",
    "write_joint_velocity_to_sim",
    "write_joint_stiffness_to_sim",
    "write_joint_damping_to_sim",
    "write_joint_position_limit_to_sim",
    "write_joint_velocity_limit_to_sim",
    "write_joint_effort_limit_to_sim",
    "write_joint_armature_to_sim",
    "write_joint_friction_coefficient_to_sim",
    "set_joint_position_target",
    "set_joint_velocity_target",
    "set_joint_effort_target",
    "set_masses",
    "set_coms",
    "set_inertias",
    "set_external_force_and_torque",
}


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
    """Return simple names assigned by an AST assignment."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return {target.id for target in targets if isinstance(target, ast.Name)}


def _stub_generator(name: str):
    """Create a deterministic input generator for an AST-defined benchmark."""

    def generator(config):
        inputs = {
            "payload": torch.zeros((config.num_instances, 1)),
            "env_ids": torch.arange(config.num_instances, dtype=torch.int32),
        }
        if "joint" in name and "root" not in name:
            inputs["joint_ids"] = torch.arange(config.num_joints, dtype=torch.int32)
        elif any(token in name for token in ("masses", "coms", "inertias", "external_force")):
            inputs["body_ids"] = torch.arange(config.num_bodies, dtype=torch.int32)
        return inputs

    return generator


def _load_benchmark_namespace() -> dict:
    """Execute production registration/helpers and actual benchmark definitions without Kit."""
    benchmark_path = Path(__file__).parents[2] / "benchmark" / "assets" / "benchmark_articulation.py"
    tree = ast.parse(benchmark_path.read_text(), filename=str(benchmark_path))
    names = {
        "ITEM_SELECTOR_MODES",
        "ITEM_SELECTOR_KEYS",
        "ITEM_SELECTOR_FACTORIES",
        "BENCHMARKS",
        "_ItemSelectorInputFactory",
        "_register_item_selector_modes",
        "_register_benchmark_selector_modes",
        "_make_tensor_dtype_generator",
        "_measure_callable",
        "_measure_finder_paths",
        "_expected_item_selector_modes",
        "_summarize_writer_results",
        "_SelectorBenchmarkRunner",
        "_print_selector_summary",
    }
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, (ast.Assign, ast.AnnAssign)) and any(name in names for name in _assigned_names(node)))
        or (isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names)
    ]
    benchmark_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign) and any(name == "BENCHMARKS" for name in _assigned_names(node))
    )
    generator_names = {
        node.id for node in ast.walk(benchmark_node) if isinstance(node, ast.Name) and node.id.startswith("gen_")
    }
    namespace = {
        "MethodBenchmarkDefinition": lambda **kwargs: SimpleNamespace(**kwargs),
        "MethodBenchmarkRunner": object,
        "contextlib": contextlib,
        "np": np,
        "time": __import__("time"),
        "torch": torch,
        "wp": wp,
    }
    namespace.update({name: _stub_generator(name) for name in generator_names})
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), benchmark_path, "exec"), namespace)
    return namespace


def _make_robot() -> MockArticulation:
    """Create a CPU mock with full joint/body finder behavior."""
    return MockArticulation(
        num_instances=4,
        num_joints=3,
        num_bodies=2,
        joint_names=[f"joint_{index}" for index in range(3)],
        body_names=[f"body_{index}" for index in range(2)],
        device="cpu",
    )


def _make_runner(namespace: dict, config, factories: dict):
    """Construct the pure local runner without initializing benchmark backends."""
    runner = object.__new__(namespace["_SelectorBenchmarkRunner"])
    runner._config = config
    runner._selector_factories = factories
    runner._item_benchmark_names = set(factories)
    runner.selector_results = {}
    runner._sync_device = lambda: None
    return runner


def test_actual_benchmarks_register_only_meaningful_item_modes() -> None:
    """Bind the full grid to actual item definitions while excluding root and mask definitions."""
    namespace = _load_benchmark_namespace()
    benchmarks = {benchmark.name: benchmark for benchmark in namespace["BENCHMARKS"]}
    factories = namespace["ITEM_SELECTOR_FACTORIES"]

    assert set(namespace["ITEM_SELECTOR_KEYS"]) == _ITEM_BENCHMARK_NAMES
    assert set(factories) == _ITEM_BENCHMARK_NAMES
    for name in _ITEM_BENCHMARK_NAMES:
        assert tuple(benchmarks[name].input_generators) == namespace["ITEM_SELECTOR_MODES"]
    assert tuple(benchmarks["write_root_state_to_sim"].input_generators) == (
        "torch_list",
        "torch_tensor_int32",
        "torch_tensor_int64",
    )
    mask_benchmarks = [benchmark for benchmark in benchmarks.values() if benchmark.name.endswith("_mask")]
    assert all(tuple(benchmark.input_generators) == ("warp_mask",) for benchmark in mask_benchmarks)
    assert not set(factories).intersection(benchmark.name for benchmark in mask_benchmarks)


@pytest.mark.parametrize(
    ("benchmark_name", "item_key", "finder_name"),
    [("write_joint_state_to_sim", "joint_ids", "find_joints"), ("set_masses", "body_ids", "find_bodies")],
)
def test_actual_registered_modes_share_payload_and_environment_identity(
    benchmark_name: str, item_key: str, finder_name: str
) -> None:
    """Prepare exact selector representations through production registration once."""
    namespace = _load_benchmark_namespace()
    benchmark = next(benchmark for benchmark in namespace["BENCHMARKS"] if benchmark.name == benchmark_name)
    factory = namespace["ITEM_SELECTOR_FACTORIES"][benchmark_name]
    robot = _make_robot()
    config = SimpleNamespace(articulation=robot, device="cpu", num_instances=4, num_joints=3, num_bodies=2)

    inputs_by_mode = {mode: generator(config) for mode, generator in benchmark.input_generators.items()}
    first_inputs = inputs_by_mode["torch_list"]

    assert factory.setup_count == 1
    assert all(inputs["env_ids"] is first_inputs["env_ids"] for inputs in inputs_by_mode.values())
    assert all(inputs["payload"] is first_inputs["payload"] for inputs in inputs_by_mode.values())
    assert first_inputs["env_ids"].dtype == torch.int32
    assert isinstance(inputs_by_mode["torch_list"][item_key], list)
    assert inputs_by_mode["torch_tensor_int32"][item_key].dtype == torch.int32
    assert inputs_by_mode["torch_tensor_int64"][item_key].dtype == torch.int64
    assert isinstance(inputs_by_mode["warp_int32"][item_key], wp.array)
    assert inputs_by_mode["warp_int32"][item_key].dtype == wp.int32
    assert isinstance(inputs_by_mode["warp_int64"][item_key], wp.array)
    assert inputs_by_mode["warp_int64"][item_key].dtype == wp.int64
    proxy = inputs_by_mode["proxy_int32"][item_key]
    assert isinstance(proxy, ProxyArray)
    assert proxy is getattr(robot, finder_name)(".*", as_proxy=True)[0]
    assert proxy._torch_cache is None


def test_cold_cache_setup_is_outside_each_timed_finder_interval() -> None:
    """Exclude old-selector release/cache clear from cold timing and prime cached timing once."""
    namespace = _load_benchmark_namespace()
    events = []

    class Selector:
        _torch_cache = None

    class Articulation:
        def __init__(self):
            self.selector = None

        def _clear_selector_cache(self):
            events.append("clear")
            self.selector = None

        def find_joints(self, _expression, *, as_proxy):
            events.append("find")
            if self.selector is None:
                self.selector = Selector()
            return self.selector, []

    class Clock:
        value = 0.0

        @classmethod
        def perf_counter(cls):
            events.append("clock")
            cls.value += 1.0
            return cls.value

    namespace["time"] = Clock
    namespace["wp"] = SimpleNamespace(synchronize=lambda: events.append("sync"))

    results = namespace["_measure_finder_paths"](Articulation(), "find_joints", 2, 0)
    clock_indices = [index for index, event in enumerate(events) if event == "clock"]
    intervals = [events[start + 1 : stop] for start, stop in zip(clock_indices[::2], clock_indices[1::2])]

    assert intervals[:2] == [["find", "sync"], ["find", "sync"]]
    assert intervals[2:] == [["find"], ["find"]]
    assert all("clear" not in interval for interval in intervals)
    assert events[:2] == ["clear", "sync"]
    assert all(stats["n"] == stats["attempts"] == 2 for stats in results.values())
    assert all(stats["failures"] == 0 for stats in results.values())


def test_runner_keeps_setup_and_integrity_checks_outside_timed_intervals() -> None:
    """Run the production runner with a controlled clock and observe timing boundaries."""
    namespace = _load_benchmark_namespace()
    events = []

    class Clock:
        value = 0.0

        @classmethod
        def perf_counter(cls):
            events.append("clock")
            cls.value += 1.0
            return cls.value

    class Factory:
        def assert_proxy_unmaterialized(self):
            events.append("check")

    namespace["time"] = Clock
    config = SimpleNamespace(device="cpu", warmup_steps=1, num_iterations=2)
    runner = _make_runner(namespace, config, {"write_joint_state_to_sim": Factory()})

    def generator(_config):
        events.append("generator")
        return {"payload": object()}

    def writer(**_inputs):
        events.append("writer")

    result = runner._benchmark_method(writer, "write_joint_state_to_sim_proxy_int32", generator, [])
    clock_indices = [index for index, event in enumerate(events) if event == "clock"]
    intervals = [events[start + 1 : stop] for start, stop in zip(clock_indices[::2], clock_indices[1::2])]

    assert events.index("generator") < clock_indices[0]
    assert intervals == [["writer"], ["writer"]]
    assert all("check" not in interval for interval in intervals)
    assert events.count("check") == 3
    assert result["n"] == result["attempts"] == 2
    assert result["failures"] == 0


def test_runner_detects_proxy_torch_materialization_and_accepts_clean_writer() -> None:
    """Reject a writer that materializes the prepared proxy and retain a clean full-count result."""
    namespace = _load_benchmark_namespace()
    benchmark = next(benchmark for benchmark in namespace["BENCHMARKS"] if benchmark.name == "write_joint_state_to_sim")
    factory = namespace["ITEM_SELECTOR_FACTORIES"][benchmark.name]
    config = SimpleNamespace(
        articulation=_make_robot(),
        device="cpu",
        num_instances=4,
        num_joints=3,
        num_bodies=2,
        warmup_steps=0,
        num_iterations=2,
    )
    runner = _make_runner(namespace, config, {benchmark.name: factory})
    generator = benchmark.input_generators["proxy_int32"]

    def materializing_writer(**inputs):
        _ = inputs["joint_ids"].torch

    with pytest.raises(AssertionError, match="materialized"):
        runner._benchmark_method(materializing_writer, f"{benchmark.name}_proxy_int32", generator, [])

    clean_namespace = _load_benchmark_namespace()
    clean_benchmark = next(
        benchmark for benchmark in clean_namespace["BENCHMARKS"] if benchmark.name == "write_joint_state_to_sim"
    )
    clean_factory = clean_namespace["ITEM_SELECTOR_FACTORIES"][clean_benchmark.name]
    clean_config = SimpleNamespace(
        articulation=_make_robot(),
        device="cpu",
        num_instances=4,
        num_joints=3,
        num_bodies=2,
        warmup_steps=0,
        num_iterations=2,
    )
    clean_runner = _make_runner(clean_namespace, clean_config, {clean_benchmark.name: clean_factory})
    result = clean_runner._benchmark_method(
        lambda **_inputs: None,
        f"{clean_benchmark.name}_proxy_int32",
        clean_benchmark.input_generators["proxy_int32"],
        [],
    )

    assert result["n"] == result["attempts"] == 2
    assert result["failures"] == 0
    assert clean_factory.proxy_selector._torch_cache is None


def test_runner_fails_immediately_on_timed_writer_exception() -> None:
    """Do not publish partial sample counts or ratios after a timed writer failure."""
    namespace = _load_benchmark_namespace()
    config = SimpleNamespace(device="cpu", warmup_steps=0, num_iterations=2)
    runner = _make_runner(namespace, config, {})
    calls = 0

    def writer(**_inputs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("timed failure")

    with pytest.raises(RuntimeError, match="timed failure"):
        runner._benchmark_method(writer, "write_root_state_to_sim_torch_list", lambda _config: {}, [])
    assert runner.selector_results == {}


def test_summary_filters_root_modes_and_reports_counts_and_missing_baselines(capsys) -> None:
    """Summarize only registered item methods and make single-mode baselines conspicuous."""
    namespace = _load_benchmark_namespace()
    results = {
        "write_root_state_to_sim_torch_list": {
            "median": 9.0,
            "iqr": 1.0,
            "n": 3,
            "attempts": 3,
            "failures": 0,
        },
        "write_joint_state_to_sim_proxy_int32": {
            "median": 2.0,
            "iqr": 0.25,
            "n": 3,
            "attempts": 3,
            "failures": 0,
        },
        "set_masses_torch_list": {"median": 8.0, "iqr": 1.0, "n": 3, "attempts": 3, "failures": 0},
        "set_masses_torch_tensor_int32": {"median": 4.0, "iqr": 0.5, "n": 3, "attempts": 3, "failures": 0},
        "set_masses_proxy_int32": {"median": 2.0, "iqr": 0.25, "n": 3, "attempts": 3, "failures": 0},
        "set_coms_torch_list": {"median": 8.0, "iqr": 1.0, "n": 2, "attempts": 3, "failures": 0},
        "set_coms_torch_tensor_int32": {"median": 4.0, "iqr": 0.5, "n": 3, "attempts": 3, "failures": 0},
        "set_coms_proxy_int32": {"median": 2.0, "iqr": 0.25, "n": 3, "attempts": 3, "failures": 0},
    }

    summary = namespace["_summarize_writer_results"](results, {"write_joint_state_to_sim", "set_masses", "set_coms"})
    namespace["_print_selector_summary"]({}, summary)
    output = capsys.readouterr().out

    assert "write_root_state_to_sim" not in summary
    proxy = summary["write_joint_state_to_sim"]["proxy_int32"]
    assert proxy["n"] == proxy["attempts"] == 3
    assert proxy["failures"] == 0
    assert proxy["ratio_vs_torch_tensor_int32"] is None
    assert proxy["ratio_vs_torch_list"] is None
    default_proxy = summary["set_masses"]["proxy_int32"]
    assert default_proxy["ratio_vs_torch_tensor_int32"] == 0.5
    assert default_proxy["ratio_vs_torch_list"] == 0.25
    incomplete_baseline_proxy = summary["set_coms"]["proxy_int32"]
    assert incomplete_baseline_proxy["ratio_vs_torch_tensor_int32"] == 0.5
    assert incomplete_baseline_proxy["ratio_vs_torch_list"] is None
    assert "n=3/3" in output
    assert "n/a" in output


def test_runner_reraises_immediate_registered_selector_failure() -> None:
    namespace = _load_benchmark_namespace()
    benchmark = next(benchmark for benchmark in namespace["BENCHMARKS"] if benchmark.name == "write_joint_state_to_sim")
    factory = namespace["ITEM_SELECTOR_FACTORIES"][benchmark.name]
    config = SimpleNamespace(
        articulation=_make_robot(),
        device="cpu",
        num_instances=4,
        num_joints=3,
        num_bodies=2,
        warmup_steps=0,
        num_iterations=2,
    )
    runner = _make_runner(namespace, config, {benchmark.name: factory})

    def fail_immediately(**_inputs):
        raise RuntimeError("immediate probe failure")

    with pytest.raises(RuntimeError, match="immediate probe failure"):
        runner._benchmark_method(
            fail_immediately,
            f"{benchmark.name}_proxy_int32",
            benchmark.input_generators["proxy_int32"],
            [],
        )

    assert runner.selector_results == {}


def test_runner_reraises_missing_registered_selector_method() -> None:
    namespace = _load_benchmark_namespace()
    benchmark = next(benchmark for benchmark in namespace["BENCHMARKS"] if benchmark.name == "write_joint_state_to_sim")
    factory = namespace["ITEM_SELECTOR_FACTORIES"][benchmark.name]
    config = SimpleNamespace(device="cpu", warmup_steps=0, num_iterations=1)
    runner = _make_runner(namespace, config, {benchmark.name: factory})

    with pytest.raises(AttributeError, match=rf"registered selector benchmark.*{benchmark.name}"):
        runner._benchmark_method(
            None,
            f"{benchmark.name}_proxy_int32",
            benchmark.input_generators["proxy_int32"],
            [],
        )

    assert runner.selector_results == {}


def test_summary_rejects_missing_expected_registered_mode() -> None:
    namespace = _load_benchmark_namespace()
    benchmark_name = "set_masses"
    results = {
        f"{benchmark_name}_{mode}": {
            "median": 1.0,
            "iqr": 0.1,
            "n": 1,
            "attempts": 1,
            "failures": 0,
        }
        for mode in namespace["ITEM_SELECTOR_MODES"][:-1]
    }

    with pytest.raises(RuntimeError, match=rf"{benchmark_name}.*proxy_int32"):
        namespace["_summarize_writer_results"](
            results,
            {benchmark_name},
            expected_modes=namespace["ITEM_SELECTOR_MODES"],
        )


@pytest.mark.parametrize(
    "expected_modes",
    [
        ("proxy_int32",),
        ("torch_list", "torch_tensor_int32", "proxy_int32"),
    ],
)
def test_summary_accepts_intentional_partial_expected_modes(expected_modes: tuple[str, ...]) -> None:
    namespace = _load_benchmark_namespace()
    benchmark_name = "set_masses"
    results = {
        f"{benchmark_name}_{mode}": {
            "median": 1.0,
            "iqr": 0.1,
            "n": 1,
            "attempts": 1,
            "failures": 0,
        }
        for mode in expected_modes
    }

    summary = namespace["_summarize_writer_results"](
        results,
        {benchmark_name},
        expected_modes=expected_modes,
    )

    assert tuple(summary[benchmark_name]) == expected_modes


@pytest.mark.parametrize(
    ("configured_mode", "expected_modes"),
    [
        (
            "all",
            ("torch_list", "torch_tensor_int32", "torch_tensor_int64", "warp_int32", "warp_int64", "proxy_int32"),
        ),
        ("proxy_int32", ("proxy_int32",)),
        (("torch_list", "proxy_int32", "warp_mask"), ("torch_list", "proxy_int32")),
    ],
)
def test_cli_mode_selection_declares_expected_item_modes(configured_mode, expected_modes: tuple[str, ...]) -> None:
    namespace = _load_benchmark_namespace()

    assert namespace["_expected_item_selector_modes"](configured_mode) == expected_modes


def test_successful_writer_summary_contains_all_six_registered_modes() -> None:
    namespace = _load_benchmark_namespace()
    benchmark = next(benchmark for benchmark in namespace["BENCHMARKS"] if benchmark.name == "write_joint_state_to_sim")
    factory = namespace["ITEM_SELECTOR_FACTORIES"][benchmark.name]
    config = SimpleNamespace(
        articulation=_make_robot(),
        device="cpu",
        num_instances=4,
        num_joints=3,
        num_bodies=2,
        warmup_steps=0,
        num_iterations=1,
    )
    runner = _make_runner(namespace, config, {benchmark.name: factory})

    for mode, generator in benchmark.input_generators.items():
        runner._benchmark_method(lambda **_inputs: None, f"{benchmark.name}_{mode}", generator, [])

    summary = namespace["_summarize_writer_results"](
        runner.selector_results,
        {benchmark.name},
        expected_modes=namespace["ITEM_SELECTOR_MODES"],
    )

    assert tuple(summary[benchmark.name]) == namespace["ITEM_SELECTOR_MODES"]
