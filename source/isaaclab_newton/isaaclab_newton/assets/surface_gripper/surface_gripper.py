# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bind Newton's surface-gripper example model to replicated robot tools and box parcels."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import warp as wp
from newton import GeoType, ShapeFlags, State
from newton.examples.surface_gripper.surface_gripper import (
    SurfaceGripper as NewtonSurfaceGripper,
)
from newton.examples.surface_gripper.surface_gripper import (
    SurfaceGripperBuilder,
    attach_seal,
    evaluate_gripper_force,
)

from isaaclab.assets.surface_gripper import BaseSurfaceGripper
from isaaclab.cloner.cloner_cfg import DEFAULT_ENV_TEMPLATE, expand_env_regex_ns
from isaaclab.physics import PhysicsEvent
from isaaclab.sim import SimulationContext
from isaaclab.utils import clone, validate
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.memory import make_masked_copy, resolve_1d_indices, resolve_1d_mask, resolve_ids

from isaaclab_newton.physics import NewtonManager

if TYPE_CHECKING:
    from isaaclab.assets.surface_gripper import SurfaceGripperCfg


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
    mask: wp.array[wp.bool],
    enabled: wp.array[wp.bool],
    targets: wp.array[wp.int32],
    engaged: wp.array[wp.bool],
    command: wp.array[wp.float32],
    state: wp.array[wp.float32],
    previous: wp.array[wp.vec2i],
    current: wp.array[wp.vec2i],
    vacuum: wp.array[wp.float32],
    load: wp.array[wp.vec4],
    unclamped_load: wp.array[wp.vec4],
):
    world = wp.tid()
    if not mask[world]:
        return
    enabled[world] = False
    targets[world] = -1
    engaged[world] = False
    command[world] = 0.0
    state[world] = -1.0
    for offset in range(4):
        pad = world * 4 + offset
        previous[pad] = wp.vec2i(-1, -1)
        current[pad] = wp.vec2i(-1, -1)
        vacuum[pad] = 0.0
        load[pad] = wp.vec4(0.0)
        unclamped_load[pad] = wp.vec4(0.0)


@wp.kernel
def _stage_vacuum(
    enabled: wp.array[wp.bool],
    world_ids: wp.array[wp.int32],
    vacuum: wp.array[wp.bool],
    command: wp.array[wp.float32],
):
    i = wp.tid()
    world = world_ids[i]
    vacuum[world] = enabled[i]
    command[world] = wp.where(enabled[i], 1.0, -1.0)


@wp.kernel
def _apply_commands(command: wp.array[wp.float32], enabled: wp.array[wp.bool]):
    world = wp.tid()
    if command[world] > 0.3:
        enabled[world] = True
    elif command[world] < -0.3:
        enabled[world] = False


@wp.kernel
def _update_engagement(
    mask: wp.array[wp.bool],
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
    gripper_state: wp.array[wp.float32],
):
    world = wp.tid()
    if mask and not mask[world]:
        return
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
        requested = target_ids[world]
        for target in range(target_body.shape[1]):
            if active and (requested == -1 or requested == target):
                candidate_body = target_body[world, target]
                candidate_shape = target_shape[world, target]
                box = body_q[candidate_body] * shape_transform[candidate_shape]
                fits = True
                for offset in range(4):
                    fits = fits and _box_seal_fits(
                        tool * pad_transform[first + offset], box, shape_scale[candidate_shape], radius, gap
                    )
                if fits:
                    body = candidate_body
                    shape = candidate_shape
                    break
        active = active and body >= 0
    engaged[world] = active
    gripper_state[world] = wp.where(active, 1.0, wp.where(enabled[world], 0.0, -1.0))
    for offset in range(4):
        pad = first + offset
        pair = wp.vec2i(-1, -1)
        if active:
            pair = wp.vec2i(body, shape)
        current[pad] = pair
        vacuum[pad] = float(active)


class SurfaceGripper(BaseSurfaceGripper):
    """One compliant four-pad suction tool per Newton world.

    A closing tool automatically acquires the first eligible box whose face fits
    every pad. :meth:`set_target_ids` optionally restricts acquisition to one
    naturally ordered payload index. No payload pose or velocity is modified.
    """

    is_graphable = True

    def __init__(self, cfg: SurfaceGripperCfg, *, sim: SimulationContext | None = None) -> None:
        """Register native binding before the first simulation reset captures solver graphs."""
        validate(cfg)
        self.cfg = clone(cfg)
        sim = SimulationContext.instance() if sim is None else sim
        self._manager: NewtonManager = sim.backend_manager
        for name in ("max_grip_distance", "coaxial_force_limit", "shear_force_limit", "retry_interval"):
            if getattr(cfg, name) is not None:
                raise NotImplementedError(f"Newton surface grippers do not support the PhysX property {name!r}.")
        for name in ("pad_spacing", "pad_radius", "contact_gap", "break_distance", "force_per_pad"):
            value = getattr(cfg, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Surface-gripper {name} must be finite and positive.")
        if len(cfg.tool_offset) != 3 or not all(math.isfinite(value) for value in cfg.tool_offset):
            raise ValueError("Surface-gripper tool_offset must contain three finite values.")
        if cfg.spawn is not None:
            raise NotImplementedError(
                "Newton surface grippers attach to existing tool bodies; configure robot spawning separately."
            )
        plan = sim.get_clone_plan()
        template = DEFAULT_ENV_TEMPLATE if plan is None else plan.env_template
        self._tool_pattern = re.compile(expand_env_regex_ns(cfg.prim_path, template))
        self._target_pattern = re.compile(expand_env_regex_ns(cfg.target_prim_path, template))
        self._tool_offset = cfg.tool_offset
        self._pad_spacing = cfg.pad_spacing
        self._pad_radius = cfg.pad_radius
        self._contact_gap = cfg.contact_gap
        self._break_distance = cfg.break_distance
        self._force_per_pad = cfg.force_per_pad
        self._model = None
        self._ready_handle = self._manager.register_callback(
            self._queue_binding, PhysicsEvent.PHYSICS_READY, name="surface_gripper_solver_init"
        )

    @property
    def num_instances(self) -> int:
        return self._num_envs

    @property
    def device(self) -> str:
        return str(self._model.device)

    @property
    def is_initialized(self) -> bool:
        return self._model is not None

    @property
    def state(self) -> wp.array:
        return self._state.warp

    @property
    def command(self) -> wp.array:
        return self._command.warp

    @property
    def engaged(self) -> torch.Tensor:
        """Whether all pads currently hold a sealed parcel, shape ``(num_envs,)``."""
        return self._engaged.torch

    @property
    def target_ids(self) -> torch.Tensor:
        """Acquisition restrictions; -1 permits any eligible payload."""
        return self._target_ids.torch

    def _indices(self, env_ids) -> wp.array:
        return resolve_1d_indices(resolve_ids(ids=env_ids, all_ids=self._world_ids))

    def set_target_ids(self, target_ids: torch.Tensor | wp.array, env_ids=None) -> None:
        """Restrict selected tools to payload IDs, or pass -1 to enable automatic acquisition."""
        ids = self._indices(env_ids)
        values = wp.from_torch(target_ids.to(dtype=torch.int32)) if isinstance(target_ids, torch.Tensor) else target_ids
        wp.copy(wp.indexedarray(self._target_ids.warp, [ids]), values)

    def set_grippers_command_index(self, states, env_ids=None, full_data: bool = False) -> None:
        """Stage compact commands, or full-size commands when ``full_data`` is true."""
        ids = self._indices(env_ids)
        if isinstance(states, ProxyArray):
            states = states.warp
        values = wp.from_torch(states.to(dtype=torch.float32)) if isinstance(states, torch.Tensor) else states
        if full_data:
            values = wp.indexedarray(values, [ids])
        wp.copy(wp.indexedarray(self._command.warp, [ids]), values)

    def set_grippers_command_mask(self, states, env_mask=None) -> None:
        """Stage full-size commands for the masked worlds without compacting CUDA masks."""
        if isinstance(states, ProxyArray):
            states = states.warp
        values = wp.from_torch(states.to(dtype=torch.float32)) if isinstance(states, torch.Tensor) else states
        mask = resolve_1d_mask(mask=env_mask, all_mask=self._all_mask, scratch_mask=self._selection_mask)
        wp.launch(
            make_masked_copy(wp.float32, 1),
            dim=self._num_envs,
            inputs=[values, mask, None],
            outputs=[self._command.warp],
            device=self._model.device,
        )

    def set_grippers_command(self, states, indices=None) -> None:
        """Stage commands for all or selected worlds."""
        self.set_grippers_command_index(states, indices)

    def set_command(self, enabled: torch.Tensor | wp.array, target_ids: torch.Tensor | wp.array, env_ids=None) -> None:
        """Apply vacuum and target restrictions together, preserving the task reset interface."""
        ids = self._indices(env_ids)
        wp.launch(
            _stage_vacuum,
            dim=ids.shape[0],
            inputs=[wp.from_torch(enabled) if isinstance(enabled, torch.Tensor) else enabled, ids],
            outputs=[self._enabled.warp, self._command.warp],
            device=self._model.device,
        )
        self.set_target_ids(target_ids, ids)

    def reset(self, env_ids=None, env_mask=None) -> None:
        """Release selected tools without changing physical poses or unselected worlds."""
        mask = resolve_1d_mask(ids=env_ids, mask=env_mask, all_mask=self._all_mask, scratch_mask=self._selection_mask)
        wp.launch(
            _reset_pads,
            dim=self._num_envs,
            inputs=[mask],
            outputs=[
                self._enabled.warp,
                self._target_ids.warp,
                self._engaged.warp,
                self._command.warp,
                self._state.warp,
                self._previous.pad_engaged_bs,
                self._current.pad_engaged_bs,
                self._control.pad_grip_control,
                self._output.pad_seal_load,
                self._output.pad_seal_load_unclamped,
            ],
            device=self._model.device,
        )

    def write_data_to_sim(self) -> None:
        """Apply staged close/open commands; values inside [-0.3, 0.3] retain the current vacuum."""
        wp.launch(
            _apply_commands,
            dim=self._num_envs,
            inputs=[self._command.warp],
            outputs=[self._enabled.warp],
            device=self._model.device,
        )

    def update(self, dt: float) -> None:
        """Native solver callbacks already publish the current seal state."""

    def refresh(self, env_ids: torch.Tensor | None = None) -> None:
        """Rebuild selected seals from restored geometry without stepping physics or applying forces.

        Call after restoring robot and parcel states and setting their vacuum commands. Attachment
        still requires every pad to fit the actual parcel face; this does not force a grasp.
        Unselected worlds retain their attachment state.
        """
        self._manager.forward()
        mask = resolve_1d_mask(ids=env_ids, all_mask=self._all_mask, scratch_mask=self._selection_mask)
        self._update_seals(self._manager.backend.state_0, mask)
        wp.copy(self._previous.pad_engaged_bs, self._current.pad_engaged_bs)

    def close(self) -> None:
        """Detach callbacks before the model is released; safe to call twice."""
        self._ready_handle.deregister()
        self._manager.unregister_solver_init_callback(self._bind)
        self._manager.unregister_state_force_callback(self._apply)
        self._model = None

    def _queue_binding(self, _event: Any) -> None:
        self._manager.register_solver_init_callback(self._bind)

    def _bind(self) -> None:
        model = self._manager.backend.model
        self._num_envs = model.world_count
        body_world = model.body_world.numpy()
        shape_body, shape_type, shape_flags = (
            model.shape_body.numpy(),
            model.shape_type.numpy(),
            model.shape_flags.numpy(),
        )
        collision_shapes = {}
        for shape, (body, kind, flags) in enumerate(zip(shape_body, shape_type, shape_flags, strict=True)):
            if kind == int(GeoType.BOX) and flags & int(ShapeFlags.COLLIDE_SHAPES):
                collision_shapes.setdefault(int(body), []).append(shape)
        tool_bodies = [[] for _ in range(self._num_envs)]
        targets = [[] for _ in range(self._num_envs)]
        for body, label in enumerate(model.body_label):
            world = int(body_world[body])
            if not 0 <= world < self._num_envs:
                continue
            if self._tool_pattern.fullmatch(label):
                tool_bodies[world].append(body)
            elif self._target_pattern.fullmatch(label) and len(collision_shapes.get(body, ())) == 1:
                targets[world].append(body)
        if any(len(bodies) != 1 for bodies in tool_bodies):
            raise ValueError("A surface gripper requires exactly one matching tool body per world.")
        if not targets[0] or any(len(bodies) != len(targets[0]) for bodies in targets):
            raise ValueError(
                "Surface-gripper targets must select the same positive number of single-box bodies per world."
            )
        for bodies in targets:
            bodies.sort(
                key=lambda body: [
                    int(part) if part.isdigit() else part for part in re.split(r"(\d+)", model.body_label[body])
                ]
            )
        target_shapes = [[collision_shapes[body][0] for body in bodies] for bodies in targets]
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
        self._world_ids = wp.array(np.arange(self._num_envs), dtype=wp.int32, device=model.device)
        self._all_mask = wp.full(self._num_envs, True, dtype=wp.bool, device=model.device)
        self._selection_mask = wp.empty_like(self._all_mask)
        self._enabled = ProxyArray(wp.zeros(self._num_envs, dtype=wp.bool, device=model.device))
        self._target_ids = ProxyArray(wp.full(self._num_envs, -1, dtype=wp.int32, device=model.device))
        self._command = ProxyArray(wp.zeros(self._num_envs, dtype=wp.float32, device=model.device))
        self._state = ProxyArray(wp.full(self._num_envs, -1.0, dtype=wp.float32, device=model.device))
        self._engaged = ProxyArray(wp.zeros(self._num_envs, dtype=wp.bool, device=model.device))
        self._manager.register_state_force_callback(self._apply)

    def _update_seals(self, state: State, mask: wp.array | None = None) -> None:
        gripper = self._gripper
        wp.launch(
            _update_engagement,
            dim=self._num_envs,
            inputs=[
                mask,
                self._enabled.warp,
                self._target_ids.warp,
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
                self._engaged.warp,
                self._state.warp,
            ],
            device=self._model.device,
        )
        attach_seal(self._model, state, gripper, self._previous, self._output, self._current, self._shape_mesh_id)

    def _apply(self, state: State) -> None:
        self._update_seals(state)
        evaluate_gripper_force(
            self._model,
            state,
            self._gripper,
            self._current,
            self._output,
            self._control,
            self._manager.backend.solver_dt,
        )
        # Copy instead of Python buffer swapping: captured graphs must see the
        # same previous/current pointers on every replay, including odd substeps.
        wp.copy(self._previous.pad_engaged_bs, self._current.pad_engaged_bs)
