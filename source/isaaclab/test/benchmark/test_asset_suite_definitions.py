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
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize(
    ("component", "physics", "capabilities", "definition_count", "workload_count"),
    (
        ("articulation", "physx", frozenset({"tensor_fill"}), 30, 134),
        (
            "articulation",
            "newton",
            frozenset({"warp_mask", "tensor_fill", "mask_fill"}),
            50,
            154,
        ),
        ("articulation", "ovphysx", frozenset({"warp_mask"}), 50, 154),
        (
            "rigid_object",
            "physx",
            frozenset({"physx_legacy_state", "tensor_fill"}),
            13,
            41,
        ),
        (
            "rigid_object",
            "newton",
            frozenset({"warp_mask", "tensor_fill", "mask_fill"}),
            15,
            38,
        ),
        ("rigid_object", "ovphysx", frozenset({"warp_mask"}), 15, 38),
        (
            "rigid_object_collection",
            "physx",
            frozenset({"physx_legacy_state", "tensor_fill"}),
            13,
            86,
        ),
        (
            "rigid_object_collection",
            "newton",
            frozenset({"warp_mask", "tensor_fill", "mask_fill"}),
            15,
            58,
        ),
        ("rigid_object_collection", "ovphysx", frozenset({"warp_mask"}), 15, 58),
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
        "proxy_int32",
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
    assert isinstance(inputs_by_mode["proxy_int32"]["joint_ids"], ProxyArray)
    assert inputs_by_mode["proxy_int32"]["joint_ids"].dtype is wp.int32
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


def test_articulation_suite_declares_cold_and_cached_finder_workloads() -> None:
    """Finder benchmarks should distinguish legacy lists, cold proxy allocation, and cached proxy lookup."""
    suite = get_asset_benchmark_suite("articulation")
    finder_specs = tuple(spec for spec in suite.methods if spec.category == "selector_finder")

    assert tuple(spec.name for spec in finder_specs) == (
        "find_bodies_default",
        "find_bodies_proxy_cold",
        "find_bodies_proxy_cached",
        "find_joints_default",
        "find_joints_proxy_cold",
        "find_joints_proxy_cached",
    )
    config = MethodBenchmarkRunnerConfig(
        num_iterations=1,
        warmup_steps=0,
        num_instances=2,
        num_bodies=3,
        num_joints=4,
        device="cpu",
    )
    assert tuple(spec.method_name for spec in finder_specs) == (
        "find_bodies",
        "find_bodies",
        "find_bodies",
        "find_joints",
        "find_joints",
        "find_joints",
    )
    assert tuple(tuple(spec.input_generators) for spec in finder_specs) == (
        ("default",),
        ("proxy_cold",),
        ("proxy_cached",),
        ("default",),
        ("proxy_cold",),
        ("proxy_cached",),
    )
    assert finder_specs[0].input_generators["default"](config) == {"name_keys": ".*"}
    assert finder_specs[1].input_generators["proxy_cold"](config) == {"name_keys": ".*", "as_proxy": True}
    assert finder_specs[2].input_generators["proxy_cached"](config) == {"name_keys": ".*", "as_proxy": True}
    assert finder_specs[0].prepare_target is None
    assert finder_specs[1].prepare_target is not None
    assert finder_specs[2].prepare_target is None
    assert finder_specs[3].prepare_target is None
    assert finder_specs[4].prepare_target is not None
    assert finder_specs[5].prepare_target is None


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
