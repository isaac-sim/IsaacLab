# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused PhysX deformable type, material, and kinematic-target unit tests."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from _imports import import_physx_module
from isaaclab_physx.assets.deformable_object.kernels import (
    compute_mean_vec3f_over_vertices,
    set_kinematic_flags_to_one,
    write_nodal_vec3f_to_buffer,
)

pytestmark = pytest.mark.unit


def _module():
    return import_physx_module("isaaclab_physx.assets.deformable_object.deformable_object")


@pytest.mark.parametrize(
    ("material_schemas", "has_tetmesh", "has_mesh", "expected"),
    [
        (("PhysxSurfaceDeformableMaterialAPI",), True, False, "surface"),
        (("PhysxDeformableMaterialAPI",), False, True, "volume"),
        ((), True, False, "volume"),
        ((), False, True, "surface"),
        ((), False, False, None),
    ],
)
def test_deformable_type_prefers_material_schema_then_falls_back_to_topology(
    material_schemas: tuple[str, ...], has_tetmesh: bool, has_mesh: bool, expected: str | None
) -> None:
    """Material schemas must win, while unbound materials fall back to mesh topology."""
    infer = _module()._infer_deformable_type

    assert infer(material_schemas, has_tetmesh=has_tetmesh, has_mesh=has_mesh) == expected


def test_surface_deformable_rejects_kinematic_targets_before_touching_view() -> None:
    """Unsupported surface targets must fail before accessing volume-only buffers or TensorAPI views."""
    deformable = object.__new__(_module().DeformableObject)
    deformable._deformable_type = "surface"
    deformable._root_physx_view = SimpleNamespace()

    with pytest.raises(ValueError, match="volume deformable"):
        deformable.write_nodal_kinematic_target_to_sim_index(wp.zeros((1, 1), dtype=wp.vec4f, device="cpu"))


def test_deformable_tensor_api_float_view_is_cached_over_stable_storage() -> None:
    """Repeated nodal writes must reuse the float wrapper over the stable vector buffer."""
    deformable = object.__new__(_module().DeformableObject)
    positions = wp.zeros((2, 3), dtype=wp.vec3f, device="cpu")
    deformable._data = SimpleNamespace(_nodal_pos_w=SimpleNamespace(data=positions))
    deformable._nodal_pos_w_f32 = None

    first = deformable._get_nodal_pos_w_f32()
    second = deformable._get_nodal_pos_w_f32()

    assert first is second
    assert first.ptr == positions.ptr
    assert first.shape == (2, 3, 3)


def test_nodal_writer_scatter_preserves_unselected_environment() -> None:
    """A compact nodal write must update only the literal selected environment and vertex values."""
    source = wp.array([[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]], dtype=wp.vec3f, device="cpu")
    env_ids = wp.array([1], dtype=wp.int32, device="cpu")
    destination = wp.full((2, 2), value=-1.0, dtype=wp.vec3f, device="cpu")

    wp.launch(
        write_nodal_vec3f_to_buffer,
        dim=(1, 2),
        inputs=[source, env_ids, False],
        outputs=[destination],
        device="cpu",
    )

    np.testing.assert_array_equal(
        destination.numpy(),
        np.asarray([[[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
    )


def test_deformable_reduction_and_kinematic_flag_kernels_use_literal_layout() -> None:
    """Tiny kernels must reduce vector vertices and set only the volume free-node flag."""
    vertices = wp.array([[(1.0, 2.0, 3.0), (3.0, 4.0, 5.0)]], dtype=wp.vec3f, device="cpu")
    mean = wp.zeros(1, dtype=wp.vec3f, device="cpu")
    targets = wp.zeros(2, dtype=wp.vec4f, device="cpu")

    wp.launch(
        compute_mean_vec3f_over_vertices,
        dim=1,
        inputs=[vertices, 2],
        outputs=[mean],
        device="cpu",
    )
    wp.launch(set_kinematic_flags_to_one, dim=2, inputs=[targets], device="cpu")

    np.testing.assert_array_equal(mean.numpy(), [[2.0, 3.0, 4.0]])
    np.testing.assert_array_equal(targets.numpy(), [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
