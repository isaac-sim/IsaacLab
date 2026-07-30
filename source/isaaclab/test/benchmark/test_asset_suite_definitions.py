# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for declarative asset micro-benchmark suites."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.benchmark.asset_suites import get_asset_benchmark_suite, resolve_method_benchmarks
from isaaclab.benchmark.method_benchmark import MethodBenchmarkRunnerConfig

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize(
    ("component", "physics", "capabilities", "definition_count", "workload_count"),
    (
        ("articulation", "physx", frozenset({"tensor_fill"}), 24, 48),
        (
            "articulation",
            "newton",
            frozenset({"warp_mask", "tensor_fill", "mask_fill"}),
            44,
            68,
        ),
        ("articulation", "ovphysx", frozenset({"warp_mask"}), 44, 68),
        (
            "rigid_object",
            "physx",
            frozenset({"physx_legacy_state", "tensor_fill"}),
            13,
            26,
        ),
        (
            "rigid_object",
            "newton",
            frozenset({"warp_mask", "tensor_fill", "mask_fill"}),
            15,
            23,
        ),
        ("rigid_object", "ovphysx", frozenset({"warp_mask"}), 15, 23),
        (
            "rigid_object_collection",
            "physx",
            frozenset({"physx_legacy_state", "tensor_fill"}),
            13,
            26,
        ),
        (
            "rigid_object_collection",
            "newton",
            frozenset({"warp_mask", "tensor_fill", "mask_fill"}),
            15,
            23,
        ),
        ("rigid_object_collection", "ovphysx", frozenset({"warp_mask"}), 15, 23),
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


def test_shared_generator_preserves_index_modes_and_shapes() -> None:
    """One shared generator pair should preserve list and tensor index contracts."""
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

    list_inputs = definition.input_generators["torch_list"](config)
    tensor_inputs = definition.input_generators["torch_tensor"](config)

    assert list_inputs["position"].shape == (2, 4)
    assert list_inputs["velocity"].shape == (2, 4)
    assert list_inputs["env_ids"] == [0, 1]
    assert list_inputs["joint_ids"] == [0, 1, 2, 3]
    assert tensor_inputs["env_ids"].dtype is torch.int32
    assert tensor_inputs["joint_ids"].dtype is torch.int32


def test_generator_override_changes_only_selected_workload() -> None:
    """Adapters should override backend-specific data shapes without copying suites."""
    suite = get_asset_benchmark_suite("articulation")

    def newton_coms(config: MethodBenchmarkRunnerConfig) -> dict[str, object]:
        return {"coms": torch.zeros((config.num_instances, config.num_bodies, 3))}

    adapter = SimpleNamespace(
        physics="newton",
        capabilities=frozenset(),
        generator_overrides={("set_coms", "torch_tensor"): newton_coms},
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

    assert definition.input_generators["torch_tensor"](config)["coms"].shape == (2, 5, 3)
    assert "env_ids" in definition.input_generators["torch_list"](config)
