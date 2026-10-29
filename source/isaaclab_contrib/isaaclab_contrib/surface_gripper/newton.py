# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bind Newton's experimental suction seal to replicated robot tools and box parcels."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton import GeoType, ShapeFlags, State
from newton.surface_gripper import (
    SurfaceGripper as NewtonSurfaceGripper,
)
from newton.surface_gripper import (
    SurfaceGripperBuilder,
    attach_seal,
    evaluate_gripper_force,
)

from isaaclab.physics import PhysicsEvent


@wp.func
def _box_seal_fits(seal: wp.transform, box: wp.transform, half_extents: wp.vec3, radius: float, gap: float) -> bool:
    relative = wp.transform_inverse(box) * seal
    point = wp.transform_get_translation(relative)
    axis = wp.transform_vector(relative, wp.vec3(0.0, 0.0, 1.0))
    normal_axis = int(0)
    for component in range(1, 3):
        if wp.abs(axis[component]) > wp.abs(axis[normal_axis]):
            normal_axis = component
    sign = float(1.0)
    if axis[normal_axis] > 0.0:
        sign = -1.0
    separation = sign * point[normal_axis] - half_extents[normal_axis]
    fits = wp.abs(axis[normal_axis]) > 0.94 and separation >= -0.002 and separation <= gap
    for component in range(3):
        if component != normal_axis and wp.abs(point[component]) + radius > half_extents[component]:
            fits = False
    return fits


@wp.kernel
def _reset_pads(
    world_ids: wp.array[wp.int64],
    previous: wp.array[wp.vec2i],
    current: wp.array[wp.vec2i],
    vacuum: wp.array[wp.float32],
    load: wp.array[wp.vec4],
    unclamped_load: wp.array[wp.vec4],
):
    first = int(world_ids[wp.tid()]) * 4
    for offset in range(4):
        pad = first + offset
        previous[pad] = wp.vec2i(-1, -1)
        current[pad] = wp.vec2i(-1, -1)
        vacuum[pad] = 0.0
        load[pad] = wp.vec4(0.0)
        unclamped_load[pad] = wp.vec4(0.0)


@wp.kernel
def _update_engagement(
    enabled: wp.array[wp.bool],
    target_ids: wp.array[wp.int32],
    target_body: wp.array2d[wp.int32],
    target_shape: wp.array2d[wp.int32],
    tool_body: wp.array[wp.int32],
    tool_transform: wp.array[wp.transform],
    pad_transform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    anchor: wp.array[wp.transform],
    previous: wp.array[wp.vec2i],
    radius: wp.float32,
    gap: wp.float32,
    break_distance: wp.float32,
    current: wp.array[wp.vec2i],
    vacuum: wp.array[wp.float32],
    engaged: wp.array[wp.bool],
    seal_error: wp.array[wp.float32],
):
    world = wp.tid()
    first = world * 4
    body = previous[first][0]
    shape = previous[first][1]
    active = enabled[world]
    error = float(0.0)
    tool = body_q[tool_body[world]] * tool_transform[world]
    if body >= 0:
        for offset in range(4):
            pad = first + offset
            expected = body_q[body] * anchor[pad]
            actual = tool * pad_transform[pad]
            error = wp.max(
                error, wp.length(wp.transform_get_translation(expected) - wp.transform_get_translation(actual))
            )
        active = active and error <= break_distance
    else:
        target = target_ids[world]
        active = active and target >= 0 and target < target_body.shape[1]
        if active:
            body = target_body[world, target]
            shape = target_shape[world, target]
            box = body_q[body] * shape_transform[shape]
            for offset in range(4):
                active = active and _box_seal_fits(
                    tool * pad_transform[first + offset], box, shape_scale[shape], radius, gap
                )
    engaged[world] = active
    seal_error[world] = error
    for offset in range(4):
        pad = first + offset
        pair = wp.vec2i(-1, -1)
        if active:
            pair = wp.vec2i(body, shape)
        current[pad] = pair
        vacuum[pad] = float(active)


class SurfaceGripper:
    """Own one four-pad Newton suction tool per environment.

    The caller supplies a binary vacuum command and a within-environment target index.
    A seal forms only when all four pad perimeters fit a nearby, aligned box face.
    Newton's upstream spring-damper model then applies capped equal-and-opposite
    wrenches; this adapter never writes either body's pose or velocity.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        manager: NewtonManager,
        tool_body_pattern: str,
        target_body_pattern: str,
        target_count_per_env: int,
        tool_offset: tuple[float, float, float] = (0.0, 0.0, 0.105),
        pad_spacing: float = 0.007,
        pad_radius: float = 0.004,
        contact_gap: float = 0.006,
        break_distance: float = 0.025,
        force_per_pad: float = 4.0,
    ) -> None:
        """Register before the simulation's first reset; bind before graph capture.

        Body patterns match full Newton labels. Target indices follow the trailing
        parcel number in those labels, independently in each replicated world.
        Tool and pad local +Z point toward the parcel face.
        """
        self._manager = manager
        self._num_envs = num_envs
        self._tool_pattern = re.compile(tool_body_pattern)
        self._target_pattern = re.compile(target_body_pattern)
        self._target_count = target_count_per_env
        self._tool_offset = tool_offset
        self._pad_spacing = pad_spacing
        self._pad_radius = pad_radius
        self._contact_gap = contact_gap
        self._break_distance = break_distance
        self._force_per_pad = force_per_pad
        self._closed = False
        self._model = None
        self._ready_handle = manager.register_callback(
            self._queue_binding, PhysicsEvent.PHYSICS_READY, name="surface_gripper_solver_init"
        )

    def set_command(self, enabled: torch.Tensor, target_ids: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Write per-environment vacuum commands and target parcel indices."""
        if env_ids is None:
            self._enabled.copy_(enabled)
            self._target_ids.copy_(target_ids)
        else:
            self._enabled[env_ids] = enabled
            self._target_ids[env_ids] = target_ids.to(dtype=self._target_ids.dtype)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Release selected tools without moving their robot or parcel."""
        ids = slice(None) if env_ids is None else env_ids
        self._enabled[ids] = False
        self._target_ids[ids] = -1
        self.engaged[ids] = False
        self.seal_error[ids] = 0.0
        world_ids = self._world_ids[ids]
        wp.launch(
            _reset_pads,
            dim=world_ids.numel(),
            inputs=[wp.from_torch(world_ids)],
            outputs=[
                self._previous.pad_engaged_bs,
                self._current.pad_engaged_bs,
                self._control.pad_grip_control,
                self._output.pad_seal_load,
                self._output.pad_seal_load_unclamped,
            ],
            device=self._model.device,
        )

    def close(self) -> None:
        """Detach callbacks before the model is released; safe to call twice."""
        if self._closed:
            return
        self._ready_handle.deregister()
        self._manager.unregister_solver_init_callback(self._bind)
        self._manager.unregister_state_force_callback(self._apply)
        self._model = None
        self._closed = True

    def _queue_binding(self, _event: Any) -> None:
        self._manager.register_solver_init_callback(self._bind)

    def _bind(self) -> None:
        model = self._manager.backend.model
        body_world = model.body_world.numpy()
        tool_bodies = [[] for _ in range(self._num_envs)]
        targets = [[] for _ in range(self._num_envs)]
        for body, label in enumerate(model.body_label):
            world = int(body_world[body])
            if not 0 <= world < self._num_envs:
                continue
            if self._tool_pattern.fullmatch(label):
                tool_bodies[world].append(body)
            if self._target_pattern.fullmatch(label):
                targets[world].append(body)
        if any(len(bodies) != 1 for bodies in tool_bodies):
            raise ValueError("A surface gripper requires exactly one matching tool body per world.")
        if any(len(bodies) != self._target_count for bodies in targets):
            raise ValueError(f"A surface gripper requires {self._target_count} matching parcels per world.")
        for bodies in targets:
            bodies.sort(key=lambda body: int(re.search(r"([0-9]+)(?:/[^/]*)?$", model.body_label[body])[1]))
        shape_body, shape_type, shape_flags = (
            model.shape_body.numpy(),
            model.shape_type.numpy(),
            model.shape_flags.numpy(),
        )
        collision_shapes = {}
        for shape, (body, kind, flags) in enumerate(zip(shape_body, shape_type, shape_flags, strict=True)):
            if kind == int(GeoType.BOX) and flags & int(ShapeFlags.COLLIDE_SHAPES):
                collision_shapes.setdefault(int(body), []).append(shape)
        target_shapes = []
        for bodies in targets:
            shapes = []
            for body in bodies:
                candidates = collision_shapes.get(body, [])
                if len(candidates) != 1:
                    raise ValueError(f"Suction parcel {model.body_label[body]!r} must have one box collision shape.")
                shapes.append(candidates[0])
            target_shapes.append(shapes)
        builder = SurfaceGripperBuilder()
        for world, bodies in enumerate(tool_bodies):
            gripper = NewtonSurfaceGripper(bodies[0], wp.transform(self._tool_offset, wp.quat_identity()), world=world)
            # Four compliant pads share each payload. These fixed hardware gains
            # keep the 30 g / 40 mm parcel stable at 120 Hz; the 200 g parcel's
            # static deflection is about 4 mm (m*g / (4*120 N/m)). The upstream
            # damper still uses each actual body's mass and inertia every step.
            gripper.set_stiffness_damping(
                120.0,
                0.6,
                self._force_per_pad,
                50.0,
                0.4,
                50.0,
                0.4,
                self._force_per_pad,
                0.001,
                0.00003,
                0.001,
                0.00003,
                0.15,
                0.0005,
                0.00002,
                0.08,
            )
            for x, y in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                gripper.add_pad(
                    wp.transform((x * self._pad_spacing, y * self._pad_spacing, 0.0), wp.quat_identity()),
                    self._pad_radius,
                    0.0,
                )
            builder.add_gripper(gripper)
        self._model = model
        self._gripper = builder.finalize(device=model.device)
        self._previous, self._current = self._gripper.state_input(), self._gripper.state_input()
        self._output, self._control = self._gripper.state_output(), self._gripper.control()
        self._target_body = wp.array(np.asarray(targets, dtype=np.int32), dtype=wp.int32, device=model.device)
        self._target_shape = wp.array(np.asarray(target_shapes, dtype=np.int32), dtype=wp.int32, device=model.device)
        self._shape_mesh_id = wp.zeros(model.shape_count, dtype=wp.uint64, device=model.device)
        device = str(model.device)
        self._world_ids = torch.arange(self._num_envs, device=device)
        self._enabled = torch.zeros(self._num_envs, dtype=torch.bool, device=device)
        self._target_ids = torch.full((self._num_envs,), -1, dtype=torch.int32, device=device)
        self.engaged = torch.zeros(self._num_envs, dtype=torch.bool, device=device)
        """Whether all pads currently hold a sealed parcel, shape ``(num_envs,)``."""
        self.seal_error = torch.zeros(self._num_envs, device=device)
        """Largest pad-anchor displacement [m], shape ``(num_envs,)``."""
        self._manager.register_state_force_callback(self._apply)

    def _apply(self, state: State) -> None:
        gripper = self._gripper
        wp.launch(
            _update_engagement,
            dim=self._num_envs,
            inputs=[
                wp.from_torch(self._enabled),
                wp.from_torch(self._target_ids),
                self._target_body,
                self._target_shape,
                gripper.gripper_body_id,
                gripper.gripper_xform,
                gripper.pad_xform,
                state.body_q,
                self._model.shape_transform,
                self._model.shape_scale,
                self._output.pad_anchor_b,
                self._previous.pad_engaged_bs,
                self._pad_radius,
                self._contact_gap,
                self._break_distance,
            ],
            outputs=[
                self._current.pad_engaged_bs,
                self._control.pad_grip_control,
                wp.from_torch(self.engaged),
                wp.from_torch(self.seal_error),
            ],
            device=self._model.device,
        )
        attach_seal(self._model, state, gripper, self._previous, self._output, self._current, self._shape_mesh_id)
        evaluate_gripper_force(
            self._model,
            state,
            gripper,
            self._current,
            self._output,
            self._control,
            self._manager.backend.solver_dt,
        )
        # Copy instead of Python buffer swapping: captured graphs must see the
        # same previous/current pointers on every replay, including odd substeps.
        wp.copy(self._previous.pad_engaged_bs, self._current.pad_engaged_bs)
