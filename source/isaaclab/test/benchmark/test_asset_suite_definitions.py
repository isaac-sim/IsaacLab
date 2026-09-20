# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for declarative asset micro-benchmark suites."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.benchmark.asset_suites import get_asset_benchmark_suite, resolve_method_benchmarks
from isaaclab.benchmark.method_benchmark import MethodBenchmarkRunnerConfig

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize(
    ("component", "physics", "capabilities", "definition_count", "workload_count"),
    (
        ("articulation", "physx", frozenset({"tensor_fill"}), 30, 190),
        (
            "articulation",
            "newton",
            frozenset({"warp_mask", "tensor_fill", "mask_fill"}),
            50,
            210,
        ),
        ("articulation", "ovphysx", frozenset({"warp_mask"}), 50, 210),
        (
            "rigid_object",
            "physx",
            frozenset({"physx_legacy_state", "tensor_fill"}),
            13,
            77,
        ),
        (
            "rigid_object",
            "newton",
            frozenset({"warp_mask", "tensor_fill", "mask_fill"}),
            15,
            59,
        ),
        ("rigid_object", "ovphysx", frozenset({"warp_mask"}), 15, 59),
        (
            "rigid_object_collection",
            "physx",
            frozenset({"physx_legacy_state", "tensor_fill"}),
            13,
            113,
        ),
        (
            "rigid_object_collection",
            "newton",
            frozenset({"warp_mask", "tensor_fill", "mask_fill"}),
            15,
            75,
        ),
        ("rigid_object_collection", "ovphysx", frozenset({"warp_mask"}), 15, 75),
    ),
)
def test_method_manifests_preserve_backend_workloads(
    component: str,
    physics: str,
    capabilities: frozenset[str],
    definition_count: int,
    workload_count: int,
) -> None:
    """Capability resolution should produce every declared method workload."""
    suite = get_asset_benchmark_suite(component)
    adapter = SimpleNamespace(
        physics=physics,
        capabilities=capabilities,
        generator_overrides={},
    )

    definitions = resolve_method_benchmarks(suite, adapter)

    assert len(definitions) == definition_count
    assert sum(len(definition.input_generators) for definition in definitions) == workload_count


def test_shared_generator_preserves_item_selector_modes_and_shapes() -> None:
    """Joint-indexed writers should cover the selector dtype grid."""
    suite = get_asset_benchmark_suite("articulation")
    adapter = SimpleNamespace(physics="physx", capabilities=frozenset(), generator_overrides={})
    definition = next(
        definition
        for definition in resolve_method_benchmarks(suite, adapter)
        if definition.method_name == "write_joint_state_to_sim"
    )
    config = MethodBenchmarkRunnerConfig(
        num_iterations=1,
        warmup_steps=0,
        num_instances=2,
        num_bodies=3,
        num_joints=4,
        device="cpu",
    )

    assert tuple(definition.input_generators) == (
        "torch_list",
        "torch_tensor_int32",
        "torch_tensor_int64",
        "torch_tensor_int32_int64",
        "torch_tensor_int64_int32",
        "warp_int32",
        "warp_int64",
        "warp_int32_int64",
        "warp_int64_int32",
    )

    inputs_by_mode = {mode: generator(config) for mode, generator in definition.input_generators.items()}
    for inputs in inputs_by_mode.values():
        assert inputs["position"].shape == (2, 4)
        assert inputs["velocity"].shape == (2, 4)

    assert inputs_by_mode["torch_list"]["env_ids"] == [0, 1]
    assert inputs_by_mode["torch_list"]["joint_ids"] == [0, 1, 2, 3]
    for mode, env_dtype, joint_dtype in (
        ("torch_tensor_int32", torch.int32, torch.int32),
        ("torch_tensor_int64", torch.int64, torch.int64),
        ("torch_tensor_int32_int64", torch.int32, torch.int64),
        ("torch_tensor_int64_int32", torch.int64, torch.int32),
        ("warp_int32", wp.int32, wp.int32),
        ("warp_int64", wp.int64, wp.int64),
        ("warp_int32_int64", wp.int32, wp.int64),
        ("warp_int64_int32", wp.int64, wp.int32),
    ):
        assert inputs_by_mode[mode]["env_ids"].dtype is env_dtype
        assert inputs_by_mode[mode]["joint_ids"].dtype is joint_dtype


def test_env_only_generator_preserves_selector_modes() -> None:
    """Environment-only writers should cover every selector representation."""
    suite = get_asset_benchmark_suite("articulation")
    adapter = SimpleNamespace(physics="physx", capabilities=frozenset(), generator_overrides={})
    definition = next(
        definition
        for definition in resolve_method_benchmarks(suite, adapter)
        if definition.method_name == "write_root_state_to_sim"
    )
    config = MethodBenchmarkRunnerConfig(
        num_iterations=1,
        warmup_steps=0,
        num_instances=2,
        num_bodies=1,
        num_joints=0,
        device="cpu",
    )

    assert tuple(definition.input_generators) == (
        "torch_list",
        "torch_tensor_int32",
        "torch_tensor_int64",
        "warp_int32",
        "warp_int64",
    )
    inputs_by_mode = {mode: generator(config) for mode, generator in definition.input_generators.items()}
    assert inputs_by_mode["torch_list"]["env_ids"] == [0, 1]
    for mode, dtype in (
        ("torch_tensor_int32", torch.int32),
        ("torch_tensor_int64", torch.int64),
        ("warp_int32", wp.int32),
        ("warp_int64", wp.int64),
    ):
        assert inputs_by_mode[mode]["env_ids"].dtype is dtype


def test_generator_override_changes_only_selected_workload() -> None:
    """Adapters should override backend-specific data shapes without copying suites."""
    suite = get_asset_benchmark_suite("articulation")

    def newton_coms(config: MethodBenchmarkRunnerConfig) -> dict[str, object]:
        return {"coms": torch.zeros((config.num_instances, config.num_bodies, 3))}

    adapter = SimpleNamespace(
        physics="newton",
        capabilities=frozenset(),
        generator_overrides={("set_coms", "torch_tensor_int32"): newton_coms},
    )
    definition = next(
        definition for definition in resolve_method_benchmarks(suite, adapter) if definition.method_name == "set_coms"
    )
    config = MethodBenchmarkRunnerConfig(
        num_iterations=1,
        warmup_steps=0,
        num_instances=2,
        num_bodies=5,
        num_joints=1,
        device="cpu",
    )

    assert definition.input_generators["torch_tensor_int32"](config)["coms"].shape == (2, 5, 3)
    assert "env_ids" in definition.input_generators["torch_list"](config)
