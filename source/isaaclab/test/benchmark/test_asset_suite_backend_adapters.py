# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for package-local asset micro-benchmark adapter declarations."""

import pytest

from isaaclab.benchmark.asset_suites import get_asset_benchmark_adapter

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize(
    ("physics", "prefix", "capabilities", "property_counts"),
    [
        ("physx", "", {"physx_legacy_state", "tensor_fill"}, (65, 40, 69)),
        ("newton", "newton_", {"warp_mask", "tensor_fill", "mask_fill"}, (64, 40, 78)),
        ("ovphysx", "ovphysx_", {"warp_mask"}, (62, 40, 78)),
    ],
)
def test_backend_provider_preserves_names_capabilities_and_properties(
    physics, prefix, capabilities, property_counts
) -> None:
    """Provider adapters should preserve historical artifacts and backend surfaces."""
    components = ("articulation", "rigid_object", "rigid_object_collection")

    for component, property_count in zip(components, property_counts, strict=True):
        adapter = get_asset_benchmark_adapter(physics, component)
        assert adapter.physics == physics
        assert adapter.component == component
        assert adapter.method_benchmark_name == f"{prefix}{component}_benchmark"
        assert adapter.data_benchmark_name == f"{prefix}{component}_data_benchmark"
        assert adapter.capabilities == frozenset(capabilities)
        assert len(adapter.supported_properties) == property_count


@pytest.mark.parametrize(
    ("physics", "component", "expected_bodies", "expected_joints"),
    [
        ("physx", "articulation", 12, 11),
        ("physx", "rigid_object", 1, 0),
        ("physx", "rigid_object_collection", 4, 0),
        ("newton", "articulation", 12, 11),
        ("newton", "rigid_object", 1, 0),
        ("newton", "rigid_object_collection", 3, 0),
        ("ovphysx", "articulation", 12, 11),
        ("ovphysx", "rigid_object", 1, 0),
        ("ovphysx", "rigid_object_collection", 3, 0),
    ],
)
def test_backend_provider_preserves_workload_defaults(physics, component, expected_bodies, expected_joints) -> None:
    """Adapters should expose the retained scripts' body and joint defaults."""
    adapter = get_asset_benchmark_adapter(physics, component)

    assert adapter.default_num_bodies == expected_bodies
    assert adapter.default_num_joints == expected_joints
