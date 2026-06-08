# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""View adapter for closed-loop (non-articulated) robots in Newton.

Closed-loop robots like DR Legs have cyclic joint graphs that cannot form a
tree-structured articulation. Newton's ``ArticulationView`` requires a tree, so this
module provides ``ClosedLoopView`` -- a duck-typed replacement that accesses the global
Newton ``Model`` / ``State`` / ``Control`` arrays directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from newton import Model, State

logger = logging.getLogger(__name__)


class ClosedLoopView:
    """A view adapter for robots without a formal Newton articulation.

    Implements the read subset of ``newton.selection.ArticulationView``'s interface
    that ``ArticulationData`` depends on, by providing strided views into the global
    Newton arrays.  One "instance" corresponds to one world.

    Args:
        model: Finalized Newton ``Model``.
        pattern: Body-label pattern used to identify the robot (currently
            unused -- all bodies/joints in each world are assumed to belong
            to the robot).
    """

    def __init__(self, model: Model, pattern: str = "*") -> None:
        self.model = model
        self.device = model.device
        self._pattern = pattern

        self._world_count = model.world_count if model.world_count > 0 else 1
        self._count_per_world = 1

        # Per-world counts from the Newton model totals (Kamino's ``size`` struct
        # doesn't expose per-world joint_dof/coord counts).
        nw = max(self._world_count, 1)
        self._bodies_per_world = model.body_count // nw
        self._joints_per_world = model.joint_count // nw
        self._joint_dofs_per_world = getattr(model, "joint_dof_count", model.joint_count) // nw
        self._joint_coords_per_world = getattr(model, "joint_coord_count", model.joint_count) // nw
        self._shapes_per_world = model.shape_count // nw
        # Free-base coord/dof offsets; controllable = per_world - root.
        self._root_coord_count = 0
        self._root_dof_count = 0

        # Extract names from first world only
        self._joint_names = self._extract_names(model.joint_label, self._joints_per_world)
        self._link_names = self._extract_names(model.body_label, self._bodies_per_world)

        # Build per-body shape counts (first world)
        shape_body = model.shape_body.numpy()
        self._link_shapes: list[list[int]] = [[] for _ in range(self._bodies_per_world)]
        for s in range(self._shapes_per_world):
            bid = int(shape_body[s])
            if 0 <= bid < self._bodies_per_world:
                self._link_shapes[bid].append(s)

        # Root base type: FREE/DISTANCE -> joint_q[:root_coord_count];
        # FIXED -> joint_X_p[0]; implicit free base (body 0 has no joint to world,
        # common for closed-loop USDs) -> body_q[0].
        from newton import JointType

        self._is_implicit_free_base = False
        if model.joint_count > 0:
            jtype = int(model.joint_type.numpy()[0])
            self._is_fixed_base = jtype == int(JointType.FIXED)
            self._is_floating_base = jtype in (int(JointType.FREE), int(JointType.DISTANCE))
            if not self._is_floating_base and not self._is_fixed_base:
                joint_child = model.joint_child.numpy()[: self._joints_per_world]
                children = set(int(c) for c in joint_child if int(c) >= 0)
                if 0 not in children:
                    self._is_implicit_free_base = True
            if self._is_floating_base:
                self._root_coord_count = 7 if jtype == int(JointType.FREE) else 1
                self._root_dof_count = 6 if jtype == int(JointType.FREE) else 1
        else:
            self._is_fixed_base = True
            self._is_floating_base = False

        logger.info(
            "ClosedLoopView: %d worlds, %d bodies, %d joints (%d DOFs, %d coords) per world "
            "(fixed_base=%s floating_base=%s implicit_free_base=%s, root_coords=%d root_dofs=%d)",
            self._world_count,
            self._bodies_per_world,
            self._joints_per_world,
            self._joint_dofs_per_world,
            self._joint_coords_per_world,
            self._is_fixed_base,
            self._is_floating_base,
            self._is_implicit_free_base,
            self._root_coord_count,
            self._root_dof_count,
        )

    @staticmethod
    def _extract_names(labels, count: int) -> list[str]:
        """Extract short names from the first ``count`` labels."""
        return [lbl.rsplit("/", 1)[-1] for lbl in labels[:count]]

    # ------------------------------------------------------------------
    # Properties matching ArticulationView interface
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return self._world_count

    @property
    def world_count(self) -> int:
        return self._world_count

    @property
    def count_per_world(self) -> int:
        return self._count_per_world

    @property
    def joint_dof_count(self) -> int:
        return self._joint_dofs_per_world - self._root_dof_count

    @property
    def joint_coord_count(self) -> int:
        return self._joint_coords_per_world - self._root_coord_count

    @property
    def link_count(self) -> int:
        return self._bodies_per_world

    @property
    def tendon_count(self) -> int:
        """Number of fixed tendons (always 0 for closed-loop assets).

        Closed-loop robots are accessed via the global Newton model arrays and do not
        expose MuJoCo-style fixed tendons, so ``ArticulationData`` allocates empty tendon
        bindings for them.
        """
        return 0

    @property
    def articulation_ids(self) -> wp.array:
        """Empty ``(world_count, 0)`` int32 array (closed-loop assets have no tree articulations).

        Sizing buffers from ``articulation_ids`` (e.g.
        ``ArticulationData._create_jacobian_buffers``) yields zero rows, while
        :meth:`Articulation._get_root_view_articulation_ids` treats the empty
        array as "no articulations" so reset scoping falls back to ``env_ids`` /
        ``env_mask``.
        """
        return wp.zeros((self._world_count, 0), dtype=wp.int32, device=self.device)

    @property
    def is_fixed_base(self) -> bool:
        return self._is_fixed_base

    @property
    def is_floating_base(self) -> bool:
        return self._is_floating_base

    @property
    def joint_dof_names(self) -> list[str]:
        if self._is_floating_base:
            return self._joint_names[1:]
        return self._joint_names

    @property
    def link_names(self) -> list[str]:
        return self._link_names

    @property
    def body_names(self) -> list[str]:
        return self._link_names

    @property
    def body_shapes(self) -> list[list[int]]:
        return self._link_shapes

    @property
    def link_shapes(self) -> list[list[int]]:
        return self._link_shapes

    # ------------------------------------------------------------------
    # Core array access
    # ------------------------------------------------------------------

    def _get_strided_view(
        self,
        attrib: wp.array,
        items_per_world: int,
        _slice: slice | int | None = None,
    ) -> wp.array:
        """Create a strided view into a global Newton array shaped
        ``(world_count, count_per_world=1, val_count, ...)``. A non-zero slice start
        is folded into the view's base pointer. Passing ``_slice`` as an ``int`` drops
        the ``val_count`` axis (NumPy-style), matching ``articulation_data.py``'s
        shape convention for root transforms.
        """
        value_stride = attrib.strides[0]
        trailing_shape = attrib.shape[1:]
        trailing_strides = attrib.strides[1:]

        squeeze_val = False
        if _slice is None:
            val_count = items_per_world
            val_offset = 0
        elif isinstance(_slice, int):
            val_count = 1
            val_offset = _slice
            squeeze_val = True
        elif isinstance(_slice, slice):
            start = _slice.start or 0
            stop = _slice.stop if _slice.stop is not None else items_per_world
            val_count = stop - start
            val_offset = start
        else:
            val_count = items_per_world
            val_offset = 0

        if squeeze_val:
            shape = (self._world_count, self._count_per_world, *trailing_shape)
            strides = (
                items_per_world * value_stride,  # stride between worlds
                items_per_world * value_stride,  # stride within worlds (1 robot)
                *trailing_strides,
            )
        else:
            shape = (self._world_count, self._count_per_world, val_count, *trailing_shape)
            strides = (
                items_per_world * value_stride,  # stride between worlds
                items_per_world * value_stride,  # stride within worlds (1 robot)
                value_stride,  # stride between items
                *trailing_strides,
            )

        if attrib.ptr is None:
            result = wp.empty(shape, dtype=attrib.dtype, device=attrib.device)
            result.ptr = None
            return result

        base_ptr = int(attrib.ptr) + val_offset * int(value_stride)
        result = wp.array(
            ptr=base_ptr,
            dtype=attrib.dtype,
            shape=shape,
            strides=strides,
            device=attrib.device,
            copy=False,
        )
        return result

    def _resolve_frequency(self, name: str) -> tuple[int, int]:
        """Return (items-per-world, per-world-offset) for an attribute. Offset skips
        free-base DOF/coord rows to match isaaclab ArticulationData's controllable-DOF view.
        """
        freq = self.model.get_attribute_frequency(name)
        if isinstance(freq, str):
            return self._joints_per_world, 0
        freq_name = freq.name.lower() if hasattr(freq, "name") else str(freq).lower()
        if "body" in freq_name or "link" in freq_name:
            return self._bodies_per_world, 0
        if "joint_coord" in freq_name:
            return self._joint_coords_per_world, self._root_coord_count
        if "joint_dof" in freq_name:
            return self._joint_dofs_per_world, self._root_dof_count
        if "joint" in freq_name:
            return self._joints_per_world, (1 if self._is_floating_base else 0)
        if "shape" in freq_name:
            return self._shapes_per_world, 0
        return self._joints_per_world, 0

    # ------------------------------------------------------------------
    # Read API (matches the ArticulationView subset ArticulationData uses)
    # ------------------------------------------------------------------

    def get_attribute(self, name: str, source) -> wp.array:
        """Get a strided view of a model/state/control attribute."""
        if "." in name:
            parts = name.split(".")
            attrib = source
            for part in parts:
                attrib = getattr(attrib, part)
            freq_name = ":".join(parts)
        else:
            attrib = getattr(source, name)
            freq_name = name

        items, offset = self._resolve_frequency(freq_name)
        if offset > 0:
            return self._get_strided_view(attrib, items, _slice=slice(offset, items))
        return self._get_strided_view(attrib, items)

    def get_root_transforms(self, source: Model | State) -> wp.array:
        if self.is_floating_base:
            attrib = self._get_strided_view(
                source.joint_q, self._joint_coords_per_world, _slice=slice(0, self._root_coord_count)
            )
        elif self._is_implicit_free_base and hasattr(source, "body_q"):
            attrib = self._get_strided_view(source.body_q, self._bodies_per_world, _slice=0)
        else:
            attrib = self._get_strided_view(self.model.joint_X_p, self._joints_per_world, _slice=0)
        if attrib.dtype is wp.transformf:
            return attrib
        return wp.array(attrib, dtype=wp.transformf, device=self.device, copy=False)

    def get_root_velocities(self, source: Model | State):
        if self.is_floating_base:
            attrib = self._get_strided_view(
                source.joint_qd, self._joint_dofs_per_world, _slice=slice(0, self._root_dof_count)
            )
            if attrib.dtype is wp.spatial_vectorf:
                return attrib
            return wp.array(attrib, dtype=wp.spatial_vectorf, device=self.device, copy=False)
        if self._is_implicit_free_base and hasattr(source, "body_qd"):
            attrib = self._get_strided_view(source.body_qd, self._bodies_per_world, _slice=0)
            if attrib.dtype is wp.spatial_vectorf:
                return attrib
            return wp.array(attrib, dtype=wp.spatial_vectorf, device=self.device, copy=False)
        return None

    def get_link_transforms(self, source: Model | State) -> wp.array:
        return self._get_strided_view(source.body_q, self._bodies_per_world)

    def get_link_velocities(self, source: Model | State) -> wp.array:
        return self._get_strided_view(source.body_qd, self._bodies_per_world)

    def _dof_slice(self, is_coord: bool) -> slice:
        """Slice of joint_q/joint_qd for controllable DOFs (excludes free-base coords/DOFs)."""
        offset = self._root_coord_count if is_coord else self._root_dof_count
        per_world = self._joint_coords_per_world if is_coord else self._joint_dofs_per_world
        return slice(offset, per_world)

    def get_dof_positions(self, source: Model | State) -> wp.array:
        return self._get_strided_view(source.joint_q, self._joint_coords_per_world, _slice=self._dof_slice(True))

    def get_dof_velocities(self, source: Model | State) -> wp.array:
        return self._get_strided_view(source.joint_qd, self._joint_dofs_per_world, _slice=self._dof_slice(False))
