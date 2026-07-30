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
        ("articulation", "physx", frozenset({"tensor_fill"}), 30, 118),
        (
            "articulation",
            "newton",
            frozenset({"warp_mask", "tensor_fill", "mask_fill"}),
            50,
            138,
        ),
        ("articulation", "ovphysx", frozenset({"warp_mask"}), 50, 138),
        (
            "rigid_object",
            "physx",
            frozenset({"physx_legacy_state", "tensor_fill"}),
            13,
            38,
        ),
        (
            "rigid_object",
            "newton",
            frozenset({"warp_mask", "tensor_fill", "mask_fill"}),
            15,
            35,
        ),
        ("rigid_object", "ovphysx", frozenset({"warp_mask"}), 15, 35),
        (
            "rigid_object_collection",
            "physx",
            frozenset({"physx_legacy_state", "tensor_fill"}),
            13,
            74,
        ),
        (
            "rigid_object_collection",
            "newton",
            frozenset({"warp_mask", "tensor_fill", "mask_fill"}),
            15,
            51,
        ),
        ("rigid_object_collection", "ovphysx", frozenset({"warp_mask"}), 15, 51),
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
    """Joint-indexed writers should compare all selector representations with fixed tensor environment IDs."""
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
        "torch_precast_int32",
        "warp_int32",
        "warp_int64",
    )

    inputs_by_mode = {mode: generator(config) for mode, generator in definition.input_generators.items()}
    for inputs in inputs_by_mode.values():
        assert inputs["position"].shape == (2, 4)
        assert inputs["velocity"].shape == (2, 4)
        assert torch.equal(inputs["env_ids"], torch.tensor([0, 1], dtype=torch.int32))

    assert inputs_by_mode["torch_list"]["joint_ids"] == [0, 1, 2, 3]
    assert inputs_by_mode["torch_tensor_int32"]["joint_ids"].dtype is torch.int32
    assert inputs_by_mode["torch_tensor_int64"]["joint_ids"].dtype is torch.int64
    assert inputs_by_mode["torch_precast_int32"]["joint_ids"].dtype is torch.int64
    assert inputs_by_mode["warp_int32"]["joint_ids"].dtype is wp.int32
    assert inputs_by_mode["warp_int64"]["joint_ids"].dtype is wp.int64
    assert tuple(definition.timed_input_transforms) == ("torch_precast_int32",)


def test_env_only_generator_preserves_legacy_index_modes() -> None:
    """Environment-only writers should retain the consolidated framework's two-mode contract."""
    suite = get_asset_benchmark_suite("articulation")
    adapter = SimpleNamespace(physics="physx", capabilities=frozenset(), generator_overrides={})
    definition = next(
        definition
        for definition in resolve_method_benchmarks(suite, adapter)
        if definition.method_name == "write_root_state_to_sim"
    )

    assert tuple(definition.input_generators) == ("torch_list", "torch_tensor")


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
