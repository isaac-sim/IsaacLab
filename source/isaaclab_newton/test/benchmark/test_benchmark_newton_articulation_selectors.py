# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-safe checks for the Newton articulation selector benchmark grid."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.test.mock_interfaces import MockArticulation
from isaaclab.utils.warp import ProxyArray


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
    """Return simple names assigned by an AST assignment."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return {target.id for target in targets if isinstance(target, ast.Name)}


def _load_selector_helpers() -> dict:
    """Load pure selector benchmark helpers without starting the simulator."""
    benchmark_path = Path(__file__).parents[2] / "benchmark" / "assets" / "benchmark_articulation.py"
    tree = ast.parse(benchmark_path.read_text(), filename=str(benchmark_path))
    names = {
        "ITEM_SELECTOR_MODES",
        "_ItemSelectorInputFactory",
        "_register_item_selector_modes",
        "_measure_callable",
        "_measure_finder_paths",
        "_summarize_writer_results",
    }
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, (ast.Assign, ast.AnnAssign)) and any(name in names for name in _assigned_names(node)))
        or (isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names)
    ]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    namespace = {"np": __import__("numpy"), "time": __import__("time"), "torch": torch, "wp": wp}
    exec(compile(ast.fix_missing_locations(module), benchmark_path, "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    ("item_key", "count", "finder_name"),
    [("joint_ids", 3, "find_joints"), ("body_ids", 2, "find_bodies")],
)
def test_item_selector_modes_prepare_full_fair_grid_once(item_key: str, count: int, finder_name: str) -> None:
    """Register exact selector types while reusing one fixed environment selector and setup."""
    helpers = _load_selector_helpers()
    calls = 0
    env_ids = torch.arange(4, dtype=torch.int32)
    data = torch.zeros((4, count))

    def base_generator(_config):
        nonlocal calls
        calls += 1
        return {"data": data, "env_ids": env_ids, item_key: torch.arange(count, dtype=torch.int32)}

    benchmark = SimpleNamespace(input_generators={"torch_list": lambda _config: {}, "torch_tensor": base_generator})
    factory = helpers["_register_item_selector_modes"](benchmark, item_key)
    robot = MockArticulation(
        num_instances=4,
        num_joints=3,
        num_bodies=2,
        joint_names=[f"joint_{index}" for index in range(3)],
        body_names=[f"body_{index}" for index in range(2)],
        device="cpu",
    )
    config = SimpleNamespace(articulation=robot, device="cpu")

    assert tuple(benchmark.input_generators) == helpers["ITEM_SELECTOR_MODES"]
    inputs_by_mode = {mode: generator(config) for mode, generator in benchmark.input_generators.items()}

    assert calls == factory.setup_count == 1
    assert all(inputs["env_ids"] is env_ids for inputs in inputs_by_mode.values())
    assert all(inputs["env_ids"].dtype == torch.int32 for inputs in inputs_by_mode.values())
    assert isinstance(inputs_by_mode["torch_list"][item_key], list)
    assert inputs_by_mode["torch_tensor_int32"][item_key].dtype == torch.int32
    assert inputs_by_mode["torch_tensor_int64"][item_key].dtype == torch.int64
    assert isinstance(inputs_by_mode["warp_int32"][item_key], wp.array)
    assert inputs_by_mode["warp_int32"][item_key].dtype == wp.int32
    assert isinstance(inputs_by_mode["warp_int64"][item_key], wp.array)
    assert inputs_by_mode["warp_int64"][item_key].dtype == wp.int64
    proxy = inputs_by_mode["proxy_int32"][item_key]
    assert isinstance(proxy, ProxyArray)
    assert proxy.dtype == wp.int32
    assert proxy is getattr(robot, finder_name)(".*", as_proxy=True)[0]
    assert proxy._torch_cache is None

    repeated = benchmark.input_generators["proxy_int32"](config)
    assert repeated is inputs_by_mode["proxy_int32"]
    assert repeated[item_key] is proxy
    assert calls == factory.setup_count == 1


def test_finder_allocation_and_cached_lookup_are_separate_measurements() -> None:
    """Measure cold allocation separately from a steady cached finder lookup."""
    helpers = _load_selector_helpers()
    robot = MockArticulation(
        num_instances=2,
        num_joints=3,
        num_bodies=2,
        joint_names=[f"joint_{index}" for index in range(3)],
        body_names=[f"body_{index}" for index in range(2)],
        device="cpu",
    )

    results = helpers["_measure_finder_paths"](robot, "find_joints", num_iterations=3, warmup_steps=1)

    assert tuple(results) == ("cold_allocation", "cached_lookup")
    assert all(result["n"] == 3 for result in results.values())
    assert robot.find_joints(".*", as_proxy=True)[0]._torch_cache is None


def test_writer_summary_reports_dispersion_and_both_baseline_ratios() -> None:
    """Report medians, IQR dispersion, and ratios against tensor-int32 and list modes."""
    helpers = _load_selector_helpers()
    results = {
        "write_joint_state_to_sim_torch_list": {"median": 8.0, "iqr": 1.0},
        "write_joint_state_to_sim_torch_tensor_int32": {"median": 4.0, "iqr": 0.5},
        "write_joint_state_to_sim_proxy_int32": {"median": 2.0, "iqr": 0.25},
    }

    summary = helpers["_summarize_writer_results"](results)

    proxy = summary["write_joint_state_to_sim"]["proxy_int32"]
    assert proxy["median_us"] == 2.0
    assert proxy["iqr_us"] == 0.25
    assert proxy["ratio_vs_torch_tensor_int32"] == 0.5
    assert proxy["ratio_vs_torch_list"] == 0.25
