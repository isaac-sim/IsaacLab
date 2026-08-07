# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the thin OVPhysX deformable body view."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

import warp as wp  # noqa: E402
from isaaclab_ovphysx.assets.deformable_object.views import OvPhysxDeformableBodyView  # noqa: E402
from isaaclab_ovphysx.sim.views import OvPhysxView  # noqa: E402
from ovphysx.types import TensorType  # noqa: E402

wp.init()
wp.set_device("cpu")


@dataclass(frozen=True)
class _FakeDtype:
    code: int
    bits: int
    lanes: int = 1


_FLOAT32 = _FakeDtype(code=2, bits=32)
_INT32 = _FakeDtype(code=0, bits=32)

_SPECS = {
    TensorType.DEFORMABLE_SIM_NODAL_POSITION: ((2, 5, 3), wp.float32),
    TensorType.DEFORMABLE_SIM_ELEMENT_INDICES: ((2, 2, 4), wp.int32),
    TensorType.DEFORMABLE_COLLISION_ELEMENT_INDICES: ((2, 6, 4), wp.int32),
    TensorType.SURFACE_DEFORMABLE_SIM_POSITION: ((2, 4, 3), wp.float32),
    TensorType.SURFACE_DEFORMABLE_SIM_ELEMENT_INDICES: ((2, 2, 3), wp.int32),
}


class _FakeBinding:
    """Minimal tensor binding exposing dimensions and OVPhysX metadata."""

    def __init__(self, tensor_type: TensorType):
        self._tensor_type = tensor_type
        self.shape, dtype = _SPECS[tensor_type]
        self.dtype = _FLOAT32 if dtype == wp.float32 else _INT32
        self.count = self.shape[0]
        self.prim_paths = [f"/World/env_{index}/Soft" for index in range(self.count)]

    def read(self, values: wp.array) -> None:
        if self._tensor_type == TensorType.DEFORMABLE_SIM_ELEMENT_INDICES:
            connectivity = wp.array(
                [
                    [[0, 1, 2, 3], [1, 2, 3, 4]],
                    [[0, 1, 2, 3], [1, 2, 3, 4]],
                ],
                dtype=wp.int32,
                device="cpu",
            )
            wp.copy(values, connectivity)
        elif self._tensor_type == TensorType.DEFORMABLE_COLLISION_ELEMENT_INDICES:
            wp.copy(values, wp.full(self.shape, value=4, dtype=wp.int32, device="cpu"))
        elif self._tensor_type == TensorType.SURFACE_DEFORMABLE_SIM_ELEMENT_INDICES:
            connectivity = wp.array(
                [
                    [[0, 1, 2], [0, 2, 3]],
                    [[0, 1, 2], [0, 2, 3]],
                ],
                dtype=wp.int32,
                device="cpu",
            )
            wp.copy(values, connectivity)

    def write(self, values: wp.array, indices: wp.array | None = None, mask: wp.array | None = None) -> None:
        pass


class _FakePhysX:
    """Fake PhysX instance returning the requested binding type."""

    def create_tensor_binding(self, *, tensor_type: TensorType, pattern: str | None = None) -> _FakeBinding:
        return _FakeBinding(tensor_type)


class _MixedTopologyBinding(_FakeBinding):
    """Simulation connectivity with five nodes in one body and four in another."""

    def read(self, values: wp.array) -> None:
        if self._tensor_type == TensorType.DEFORMABLE_SIM_ELEMENT_INDICES:
            connectivity = wp.array(
                [
                    [[0, 1, 2, 3], [1, 2, 3, 4]],
                    [[0, 1, 2, 3], [0, 0, 0, 0]],
                ],
                dtype=wp.int32,
                device="cpu",
            )
            wp.copy(values, connectivity)
        else:
            super().read(values)


class _MixedTopologyPhysX(_FakePhysX):
    """Fake PhysX instance exposing padded mixed-topology bindings."""

    def create_tensor_binding(self, *, tensor_type: TensorType, pattern: str | None = None) -> _FakeBinding:
        return _MixedTopologyBinding(tensor_type)


def test_volume_view_is_ovphysx_view_with_deformable_dimensions():
    view = OvPhysxDeformableBodyView(
        _FakePhysX(),
        pattern="/World/env_*/Soft",
        device="cpu",
        tensor_types=[
            TensorType.DEFORMABLE_SIM_NODAL_POSITION,
            TensorType.DEFORMABLE_SIM_ELEMENT_INDICES,
            TensorType.DEFORMABLE_COLLISION_ELEMENT_INDICES,
        ],
        eager=True,
        simulation_nodal_position_type=TensorType.DEFORMABLE_SIM_NODAL_POSITION,
        simulation_element_indices_type=TensorType.DEFORMABLE_SIM_ELEMENT_INDICES,
        collision_element_indices_type=TensorType.DEFORMABLE_COLLISION_ELEMENT_INDICES,
    )

    assert isinstance(view, OvPhysxView)
    assert view.count == 2
    assert view.max_simulation_nodes_per_body == 5
    assert view.max_simulation_elements_per_body == 2
    assert view.max_collision_elements_per_body == 6
    assert view.max_collision_nodes_per_body == 5


def test_surface_view_reports_zero_collision_dimensions():
    view = OvPhysxDeformableBodyView(
        _FakePhysX(),
        pattern="/World/env_*/Cloth",
        device="cpu",
        tensor_types=[
            TensorType.SURFACE_DEFORMABLE_SIM_POSITION,
            TensorType.SURFACE_DEFORMABLE_SIM_ELEMENT_INDICES,
        ],
        eager=True,
        simulation_nodal_position_type=TensorType.SURFACE_DEFORMABLE_SIM_POSITION,
        simulation_element_indices_type=TensorType.SURFACE_DEFORMABLE_SIM_ELEMENT_INDICES,
    )

    assert view.max_simulation_nodes_per_body == 4
    assert view.max_simulation_elements_per_body == 2
    assert view.max_collision_elements_per_body == 0
    assert view.max_collision_nodes_per_body == 0


def test_volume_view_rejects_mixed_simulation_node_counts():
    with pytest.raises(
        ValueError,
        match=r"uniform simulation-node counts.*\[5, 4\]",
    ):
        OvPhysxDeformableBodyView(
            _MixedTopologyPhysX(),
            pattern="/World/env_*/Soft",
            device="cpu",
            tensor_types=[
                TensorType.DEFORMABLE_SIM_NODAL_POSITION,
                TensorType.DEFORMABLE_SIM_ELEMENT_INDICES,
                TensorType.DEFORMABLE_COLLISION_ELEMENT_INDICES,
            ],
            eager=True,
            simulation_nodal_position_type=TensorType.DEFORMABLE_SIM_NODAL_POSITION,
            simulation_element_indices_type=TensorType.DEFORMABLE_SIM_ELEMENT_INDICES,
            collision_element_indices_type=TensorType.DEFORMABLE_COLLISION_ELEMENT_INDICES,
        )
