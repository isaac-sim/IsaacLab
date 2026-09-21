# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for backend-specific asset benchmark generators."""

import pytest
import torch

from isaaclab.benchmark.asset_suites import generators as asset_generators
from isaaclab.benchmark.asset_suites import (
    get_asset_benchmark_adapter,
    get_asset_benchmark_suite,
    resolve_method_benchmarks,
)
from isaaclab.benchmark.asset_suites.generators import build_fill_benchmarks
from isaaclab.benchmark.method_benchmark import MethodBenchmarkRunnerConfig

pytestmark = pytest.mark.benchmark

_CONFIG = MethodBenchmarkRunnerConfig(
    num_iterations=1,
    warmup_steps=0,
    num_instances=2,
    num_bodies=3,
    num_joints=4,
    device="cpu",
)


@pytest.mark.parametrize(
    ("physics", "component", "expected_width"),
    (
        ("physx", "articulation", 7),
        ("newton_mjwarp", "articulation", 3),
        ("ovphysx", "articulation", 7),
        ("physx", "rigid_object", 7),
        ("newton_mjwarp", "rigid_object", 3),
        ("ovphysx", "rigid_object", 7),
        ("physx", "rigid_object_collection", 7),
        ("newton_mjwarp", "rigid_object_collection", 3),
        ("ovphysx", "rigid_object_collection", 7),
    ),
)
def test_set_coms_preserves_backend_shape_in_every_mode(physics, component, expected_width) -> None:
    """Every list, tensor, and available mask workload should use the backend CoM representation."""
    adapter = get_asset_benchmark_adapter(physics, component)
    definitions = resolve_method_benchmarks(get_asset_benchmark_suite(component), adapter)
    com_definitions = [
        definition for definition in definitions if definition.method_name in {"set_coms", "set_coms_mask"}
    ]

    for definition in com_definitions:
        for generator in definition.input_generators.values():
            assert generator(_CONFIG)["coms"].shape == (2, 3, expected_width)


def test_physx_rigid_object_omits_body_ids_in_every_index_mode() -> None:
    """PhysX rigid-object property setters should avoid advanced body indexing."""
    adapter = get_asset_benchmark_adapter("physx", "rigid_object")
    definitions = resolve_method_benchmarks(get_asset_benchmark_suite("rigid_object"), adapter)

    for method_name in ("set_masses", "set_coms", "set_inertias"):
        definition = next(definition for definition in definitions if definition.method_name == method_name)
        for mode in definition.input_generators:
            assert "body_ids" not in definition.input_generators[mode](_CONFIG)


def test_physx_collection_inertia_uses_flattened_shape_in_every_index_mode() -> None:
    """PhysX collection inertia setters should retain their flattened matrix inputs."""
    adapter = get_asset_benchmark_adapter("physx", "rigid_object_collection")
    definitions = resolve_method_benchmarks(get_asset_benchmark_suite("rigid_object_collection"), adapter)
    definition = next(definition for definition in definitions if definition.method_name == "set_inertias")

    for mode in definition.input_generators:
        assert definition.input_generators[mode](_CONFIG)["inertias"].shape == (2, 3, 9)


def test_tensor_fill_preserves_item_selector_width() -> None:
    """Fill workloads should slice environment data without truncating item selectors."""
    adapter = get_asset_benchmark_adapter("physx", "rigid_object_collection")
    definitions = resolve_method_benchmarks(get_asset_benchmark_suite("rigid_object_collection"), adapter)
    fills = build_fill_benchmarks(definitions, capabilities=adapter.capabilities)
    definition = next(definition for definition in fills if definition.method_name == "write_body_state_to_sim")
    config = MethodBenchmarkRunnerConfig(
        num_iterations=1,
        warmup_steps=0,
        num_instances=4,
        num_bodies=4,
        num_joints=0,
        device="cpu",
    )

    inputs = definition.input_generators["tensor_5pct"](config)

    assert inputs["body_states"].shape == (1, 4, 13)
    assert inputs["env_ids"].shape == (1,)
    assert inputs["body_ids"].shape == (4,)


@pytest.mark.parametrize("physics", ("physx", "newton_mjwarp", "ovphysx"))
def test_articulation_position_limits_preserve_lower_upper_ordering(physics) -> None:
    """Joint-limit inputs should always place non-positive lower bounds before non-negative upper bounds."""
    adapter = get_asset_benchmark_adapter(physics, "articulation")
    definitions = resolve_method_benchmarks(get_asset_benchmark_suite("articulation"), adapter)
    limit_definitions = [
        definition
        for definition in definitions
        if definition.method_name in {"write_joint_position_limit_to_sim", "write_joint_position_limit_to_sim_mask"}
    ]

    for definition in limit_definitions:
        for generator in definition.input_generators.values():
            limits = generator(_CONFIG)["limits"]
            assert isinstance(limits, torch.Tensor)
            assert torch.all(limits[..., 0] <= 0)
            assert torch.all(limits[..., 1] >= 0)


@pytest.mark.parametrize("physics", ("physx", "newton_mjwarp", "ovphysx"))
@pytest.mark.parametrize(
    ("method_name", "field_name", "expected_scale"),
    (
        ("write_joint_stiffness_to_sim", "stiffness", 1.0),
        ("write_joint_damping_to_sim", "damping", 1.0),
        ("write_joint_velocity_limit_to_sim", "limits", 10.0),
        ("write_joint_effort_limit_to_sim", "limits", 100.0),
        ("write_joint_armature_to_sim", "armature", 0.1),
        ("write_joint_friction_coefficient_to_sim", "joint_friction_coeff", 0.5),
    ),
)
def test_articulation_joint_parameter_generators_preserve_workload_scales(
    monkeypatch, physics, method_name, field_name, expected_scale
) -> None:
    """Every backend and available input mode should preserve its workload parameter scale."""

    def ones(*shape, **kwargs):
        return torch.ones(*shape, device=kwargs.get("device"), dtype=kwargs.get("dtype"))

    monkeypatch.setattr(asset_generators.torch, "rand", ones)
    adapter = get_asset_benchmark_adapter(physics, "articulation")
    definitions = resolve_method_benchmarks(get_asset_benchmark_suite("articulation"), adapter)
    matching = [
        definition for definition in definitions if definition.method_name in {method_name, f"{method_name}_mask"}
    ]

    assert matching
    for definition in matching:
        for generator in definition.input_generators.values():
            values = generator(_CONFIG)[field_name]
            assert torch.all(values == expected_scale)
