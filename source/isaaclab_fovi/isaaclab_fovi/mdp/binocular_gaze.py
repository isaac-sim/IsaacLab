# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Binocular gaze for two foveated cameras on one body: a shared version and a vergence.

The version is the shared line of sight, (tilt, pan) as for a single camera's gaze; the vergence is the angle between
the two eyes' lines of sight. Each eye turns from the version by half the vergence, toward the other eye, in the plane
that holds the version and the baseline between the eyes, so with any vergence above zero the two lines of sight
meet; at zero they are parallel. Both are in field radii, half the retina's field of view.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING, ClassVar

import torch
import warp as wp

from isaaclab.managers import ActionTerm, ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.utils.warp import ProxyArray

from ..sensors.gaze_kernels import DEG_TO_RAD, CameraParams, project_direction, record_launch, warp_stream
from .gaze import (
    HYBRID,
    PHYSICAL,
    RAD_TO_DEG,
    clamp_unit,
    gaze_direction,
    gaze_of,
    head_and_target_kernel,
    head_angles,
    rig_gaze,
    target_body_index,
    within,
    write_head_targets,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from ..sensors import FoveatedCamera
    from .gaze_cfg import BinocularGazeActionCfg

SOFTWARE = 0
"""Code of software gaze for :func:`make_binocular_gaze_step`; the rig modes share the single camera's codes."""


@cache
def make_binocular_gaze_step(mode: int) -> wp.Kernel:
    """Return the binocular gaze step of one mode: :data:`SOFTWARE`, :data:`~isaaclab_fovi.mdp.gaze.PHYSICAL` or
    :data:`~isaaclab_fovi.mdp.gaze.HYBRID`."""
    if mode not in (SOFTWARE, PHYSICAL, HYBRID):
        raise ValueError(f"Unknown binocular gaze mode {mode!r}.")

    @wp.kernel(enable_backward=False, module="unique")
    def binocular_gaze_step(
        actions: wp.array2d(dtype=wp.float32),
        joint_pos: wp.array2d(dtype=wp.float32),
        limits: wp.array2d(dtype=wp.vec2f),
        joints: wp.vec2i,
        signs: wp.vec2f,
        mounts: wp.array(dtype=wp.quatf),
        baseline: wp.vec3f,
        step_size: wp.float32,
        gaze_limit: wp.float32,
        vergence_limit: wp.float32,
        head_step: wp.float32,
        step_dt: wp.float32,
        field_radius_deg: wp.float32,
        held: wp.array(dtype=wp.vec3f),
        held_valid: wp.array(dtype=wp.int32),
        version: wp.array(dtype=wp.vec2f),
        left_deg: wp.array(dtype=wp.vec2f),
        right_deg: wp.array(dtype=wp.vec2f),
        state: wp.array2d(dtype=wp.float32),
        targets: wp.array2d(dtype=wp.float32),
    ):
        """Move each pair of eyes by its actions (version tilt, version pan, vergence) and aim both retinas.

        The version moves as a single camera's gaze does in the same mode (see
        :func:`~isaaclab_fovi.mdp.gaze.make_rig_gaze_step`); the vergence always moves the retinas. ``baseline`` is
        the unit direction from the left eye to the right in the upright frame, and ``mounts`` the eyes' shared mount.
        Software gaze reads no joints, and only hybrid gaze holds a fixation, so the arrays a mode leaves alone may be
        empty. Writes both eyes' gaze angles ``left_deg`` and ``right_deg``, the joint ``targets`` (pan, tilt) and
        ``state``: the version's direction from the rig's zero pose, the vergence, which the state alone holds, and
        their velocities.
        """
        i = wp.tid()
        radius_rad = field_radius_deg * wp.float32(DEG_TO_RAD)
        soft = version[i]
        action = clamp_unit(wp.vec2f(actions[i, 0], actions[i, 1]))
        head = wp.vec2f(0.0, 0.0)
        if wp.static(mode == SOFTWARE):
            soft = within(soft + action * step_size, gaze_limit)
        if wp.static(mode != SOFTWARE):
            head = head_angles(joint_pos, i, joints, signs)
            # Commanded head steps add to the last targets; hybrid gaze steps from where the head is.
            base = wp.vec2f(targets[i, 0], targets[i, 1])
            turn = action * step_size
            if wp.static(mode == HYBRID):
                if held_valid[i] != 0:
                    # The head turned since the last step: aim the version back at the fixation it holds.
                    soft = gaze_of(wp.quat_rotate_inv(mounts[i], held[i]), field_radius_deg)
                soft = within(soft + action * step_size, gaze_limit)
                turn = wp.vec2f(wp.clamp(soft[0], -head_step, head_step), wp.clamp(soft[1], -head_step, head_step))
                held[i] = wp.quat_rotate(mounts[i], gaze_direction(soft, field_radius_deg))
                held_valid[i] = 1
                base = wp.vec2f(joint_pos[i, joints[0]], joint_pos[i, joints[1]])
            write_head_targets(targets, limits, i, joints, signs, base, turn * radius_rad)
        verge = wp.clamp(state[i, 2] + wp.clamp(actions[i, 2], -1.0, 1.0) * step_size, 0.0, vergence_limit)
        # Each eye turns by half the vergence toward the other, in the plane of the version and the baseline.
        d = gaze_direction(soft, field_radius_deg)
        toward_right = wp.normalize(baseline - wp.dot(baseline, d) * d)
        half = 0.5 * verge * radius_rad
        left_deg[i] = gaze_of(wp.cos(half) * d + wp.sin(half) * toward_right, field_radius_deg) * field_radius_deg
        right_deg[i] = gaze_of(wp.cos(half) * d - wp.sin(half) * toward_right, field_radius_deg) * field_radius_deg
        version[i] = soft
        gaze = rig_gaze(head, soft, field_radius_deg)
        state[i, 3] = (gaze[0] - state[i, 0]) / step_dt
        state[i, 4] = (gaze[1] - state[i, 1]) / step_dt
        state[i, 5] = (verge - state[i, 2]) / step_dt
        state[i, 0] = gaze[0]
        state[i, 1] = gaze[1]
        state[i, 2] = verge

    return binocular_gaze_step


@wp.kernel(enable_backward=False)
def binocular_gaze_reset_kernel(
    ids: wp.array(dtype=wp.int32),
    use_ids: wp.int32,
    default_joint_pos: wp.array2d(dtype=wp.float32),
    joints: wp.vec2i,
    signs: wp.vec2f,
    field_radius_deg: wp.float32,
    has_rig: wp.int32,
    clear_held: wp.int32,
    targets: wp.array2d(dtype=wp.float32),
    held_valid: wp.array(dtype=wp.int32),
    version: wp.array(dtype=wp.vec2f),
    left_deg: wp.array(dtype=wp.vec2f),
    right_deg: wp.array(dtype=wp.vec2f),
    state: wp.array2d(dtype=wp.float32),
):
    """Put each reset environment's version on the optical axes, its vergence at zero and, with a rig (``has_rig``),
    its joint targets at their defaults; the state starts at the default head's angles, at rest."""
    k = wp.tid()
    i = k
    if use_ids != 0:
        i = ids[k]
    version[i] = wp.vec2f(0.0, 0.0)
    left_deg[i] = wp.vec2f(0.0, 0.0)
    right_deg[i] = wp.vec2f(0.0, 0.0)
    head = wp.vec2f(0.0, 0.0)
    if has_rig != 0:
        targets[i, 0] = default_joint_pos[i, joints[0]]
        targets[i, 1] = default_joint_pos[i, joints[1]]
        head = head_angles(default_joint_pos, i, joints, signs) / (field_radius_deg * wp.float32(DEG_TO_RAD))
    if clear_held != 0:
        held_valid[i] = 0
    state[i, 0] = head[0]
    state[i, 1] = head[1]
    for column in range(2, 6):
        state[i, column] = 0.0


@wp.kernel(enable_backward=False)
def binocular_teacher_kernel(
    points: wp.array(dtype=wp.vec3f),
    left_positions: wp.array(dtype=wp.vec3f),
    right_positions: wp.array(dtype=wp.vec3f),
    mounts: wp.array(dtype=wp.quatf),
    version: wp.array(dtype=wp.vec2f),
    state: wp.array2d(dtype=wp.float32),
    heads: wp.array(dtype=wp.vec4f),
    held: wp.array(dtype=wp.vec3f),
    held_valid: wp.array(dtype=wp.int32),
    mode: wp.int32,
    frame: wp.mat33f,
    field_radius_deg: wp.float32,
    step_size: wp.float32,
    gaze_limit: wp.float32,
    vergence_limit: wp.float32,
    left_camera: CameraParams,
    right_camera: CameraParams,
    labels: wp.array2d(dtype=wp.float32),
):
    """Labels (version tilt, version pan, vergence, valid) that fixate each point with both eyes.

    The version that fixates a point bisects the two eyes' lines of sight to it, and the vergence is the angle between
    them; ``state`` holds the current vergence in its third column. A point is a valid target when each eye's camera
    sees it and the fixation is within the limits. The version is labelled from where the next step starts: in hybrid
    gaze the version re-aimed at the fixation it holds in ``held``, once ``held_valid`` is set. In physical gaze the
    version actions turn the head, so they are labelled from ``heads``, the head's measured and commanded (tilt, pan)
    in field radii, and never repeat a turn the head has yet to make. ``mode`` is the mode's code, and the arrays it
    does not use may be empty.
    """
    i = wp.tid()
    mount = mounts[i]
    left = wp.quat_rotate_inv(mount, wp.normalize(points[i] - left_positions[i]))
    right = wp.quat_rotate_inv(mount, wp.normalize(points[i] - right_positions[i]))
    desired = gaze_of(wp.transpose(frame) * wp.normalize(left + right), field_radius_deg)
    angle = wp.atan2(wp.length(wp.cross(left, right)), wp.dot(left, right)) * wp.float32(RAD_TO_DEG)
    verge = angle / field_radius_deg
    ok = (
        project_direction(left, left_camera)[2] > 0.5
        and project_direction(right, right_camera)[2] > 0.5
        and wp.length(desired) <= gaze_limit
        and verge <= vergence_limit
    )
    step = wp.vec3f(0.0, 0.0, 0.0)
    if ok:
        start = version[i]
        if mode == HYBRID:
            if held_valid[i] != 0:
                start = gaze_of(wp.transpose(frame) * wp.quat_rotate_inv(mount, held[i]), field_radius_deg)
        turn = desired - start
        if mode == PHYSICAL:
            head = heads[i]
            measured = wp.vec2f(head[0], head[1]) * field_radius_deg * wp.float32(DEG_TO_RAD)
            turn = rig_gaze(measured, desired, field_radius_deg) - wp.vec2f(head[2], head[3])
        turn = clamp_unit(turn / step_size)
        step = wp.vec3f(turn[0], turn[1], wp.clamp((verge - state[i, 2]) / step_size, -1.0, 1.0))
    labels[i, 0] = step[0]
    labels[i, 1] = step[1]
    labels[i, 2] = step[2]
    labels[i, 3] = wp.where(ok, 1.0, 0.0)


@wp.kernel(enable_backward=False)
def offset_points_kernel(
    positions: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.quatf),
    offset: wp.vec3f,
    points: wp.array(dtype=wp.vec3f),
):
    """World points at ``offset`` in each frame."""
    i = wp.tid()
    points[i] = positions[i] + wp.quat_rotate(orientations[i], offset)


def binocular_gaze_state(env: ManagerBasedEnv, action_name: str = "gaze") -> torch.Tensor:
    """Version, vergence and their velocities, in field radii and field radii per second, shape (num_envs, 6)."""
    term: BinocularGazeAction = env.action_manager.get_term(action_name)
    return term.state


class BinocularGazeAction(ActionTerm):
    """Moves two foveated cameras' gazes together: three actions step the version (tilt, pan) and the vergence.

    Actions, limits and the observed gaze are in field radii. An action of 1 changes the version's coordinates and
    the vergence by ``step_size``; the version stays within ``gaze_limit`` of the eyes' optical axes, and the vergence
    between zero, parallel lines of sight, and ``vergence_limit``.

    Both eyes ride one body with parallel optical axes and the same retina, on any renderer.
    :attr:`~isaaclab_fovi.mdp.BinocularGazeActionCfg.mode` sets what the version actions
    turn, as for a single camera's :class:`~isaaclab_fovi.mdp.GazeAction`: the retinas (``"software"``), the rig's
    joints (``"physical"``), or the retinas at once while the rig follows (``"hybrid"``). The vergence always turns
    the retinas. The modes that turn a rig need eyes without image roll, whose tilt and pan follow the rig's joints.

    The term owns the version and the vergence and sets both eyes' gaze angles from them each step, so a gaze set on
    an eye directly lasts until the next step. Each episode starts with the version on the optical axes and the
    vergence at zero. :attr:`state`, the observation, holds the version, with a rig its direction from the rig's zero
    pose, the head's turn composed with the version, then the vergence, and their velocities.
    """

    cfg: BinocularGazeActionCfg

    is_graphable: ClassVar[bool] = True
    """The gaze moves in :meth:`process_actions`; :meth:`apply_actions` at most writes a rig's joint targets."""

    def __init__(self, cfg: BinocularGazeActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._right: FoveatedCamera = env.scene[cfg.right_asset_name]
        left, right = self.left, self.right
        if left.cfg.prim_path.rpartition("/")[0] != right.cfg.prim_path.rpartition("/")[0]:
            # The version, the vergence and the baseline assume one rigid head that carries both eyes.
            raise ValueError(
                "Both eyes of a binocular gaze must be on one body: their prims need the same parent, got"
                f" {left.cfg.prim_path!r} and {right.cfg.prim_path!r}."
            )
        if cfg.mode != "software" and left.cfg.image_rotation_deg != 0:
            raise ValueError(
                f"BinocularGazeActionCfg.mode={cfg.mode!r} turns the eyes with a rig's pan and tilt joints, which a"
                f" rolled image's tilt and pan do not follow; got image_rotation_deg={left.cfg.image_rotation_deg!r}."
            )
        self.field_radius_deg = left.sampling_coords.fov / 2
        """Half the retina's field of view [deg]."""
        if not math.isclose(right.sampling_coords.fov / 2, self.field_radius_deg, rel_tol=1e-6):
            raise ValueError("Both eyes of a binocular gaze need the same retina field of view.")
        if _rows(right.image_frame) != _rows(left.image_frame):
            raise ValueError("Both eyes of a binocular gaze need the same image rotation.")
        mount_left, mount_right = left.update_mount().torch, right.update_mount().torch
        # The quaternions' dot product is +-1 for one orientation.
        if (1 - (mount_left * mount_right).sum(-1).abs()).amax() > 1e-5:
            raise ValueError("Both eyes of a binocular gaze need parallel optical axes.")
        self.baseline = self._baseline(mount_left[0], left.data.pos_w.torch[0], right.data.pos_w.torch[0])
        """Unit direction from the left eye to the right, in the upright image frame."""
        self._raw: torch.Tensor | None = None
        self._version = ProxyArray(wp.zeros(self.num_envs, dtype=wp.vec2f, device=self.device))
        self._state = ProxyArray(wp.zeros((self.num_envs, 6), dtype=wp.float32, device=self.device))
        """Version, vergence and their velocities; the vergence lives here alone."""
        self._mode_code = {"software": SOFTWARE, "physical": PHYSICAL, "hybrid": HYBRID}[cfg.mode]
        # Only hybrid gaze holds a fixation; the other modes' kernels never touch these, so they stay empty.
        held = self.num_envs if self._mode_code == HYBRID else 0
        self._held = ProxyArray(wp.zeros(held, dtype=wp.vec3f, device=self.device))
        self._held_valid = ProxyArray(wp.zeros(held, dtype=wp.int32, device=self.device))
        self._heads = ProxyArray(wp.zeros(self.num_envs, dtype=wp.vec4f, device=self.device))
        """The head's measured and commanded (tilt, pan) [field radii], which the teacher labels physical gaze from."""
        self._step_launch: tuple[tuple, wp.Launch] | None = None
        self._teacher_launch: tuple[tuple, wp.Launch] | None = None
        self._heads_launch: tuple[wp.array, wp.Launch] | None = None
        self._reset_ids = ProxyArray(wp.zeros(self.num_envs, dtype=wp.int32, device=self.device))
        self._reset_launches: dict[bool, wp.Launch] = {}
        self.rig = None
        """The articulation that turns both eyes, for physical and hybrid gaze."""
        if cfg.mode == "software":
            # Software gaze reads no joints; its launches take empty arrays.
            self._joint_ids = [0, 1]
            self._targets = ProxyArray(wp.zeros((0, 2), dtype=wp.float32, device=self.device))
            self._no_joints = wp.zeros((0, 2), dtype=wp.float32, device=self.device)
            self._no_limits = wp.zeros((0, 2), dtype=wp.vec2f, device=self.device)
            self._signs = wp.vec2f(1.0, 1.0)
        else:
            self.rig = env.scene[cfg.rig.asset_name]
            self._joint_ids = list(self.rig.find_joints(list(cfg.rig.joint_names), preserve_order=True)[0])
            self._signs = wp.vec2f(*cfg.rig.joint_signs)
            # Indexing the selected joints copies them, so the targets own their storage.
            self._targets = ProxyArray(wp.from_torch(self.rig.data.default_joint_pos.torch[:, self._joint_ids]))
        self._joints = wp.vec2i(*self._joint_ids)
        self.reset()

    @property
    def left(self) -> FoveatedCamera:
        """The left eye."""
        return self._asset

    @property
    def right(self) -> FoveatedCamera:
        """The right eye."""
        return self._right

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        if self._raw is not None:
            return self._raw
        return torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._state.torch[:, :3]

    @property
    def version(self) -> torch.Tensor:
        """The software version (tilt, pan) from the eyes' optical axes, in field radii, shape (num_envs, 2)."""
        return self._version.torch

    @property
    def vergence(self) -> torch.Tensor:
        """The vergence, the angle between the lines of sight, in field radii, shape (num_envs,)."""
        return self._state.torch[:, 2]

    @property
    def state(self) -> torch.Tensor:
        """Version, vergence and their velocities, in field radii and field radii per second, shape (num_envs, 6)."""
        return self._state.torch

    def process_actions(self, actions: torch.Tensor):
        """Move the version and vergence by the supplied actions, shape (num_envs, 3), and aim both retinas."""
        if actions.dtype != torch.float32:
            actions = actions.float()
        left, right = self.left, self.right
        if self.rig is None:
            joint_pos, limits = self._no_joints, self._no_limits
        else:
            # Reading the properties refreshes the rig's joint buffers.
            joint_pos, limits = self.rig.data.joint_pos.warp, self.rig.data.soft_joint_pos_limits.warp
        # Hybrid gaze aims from the mount as the head stands now, which a lazily updated camera has not read yet.
        mounts = left.update_mount() if self._mode_code == HYBRID else left.mount_quat_w_ros
        key = (actions.data_ptr(), actions.stride(), actions.device)
        arrays = (joint_pos, limits, mounts)
        cached = self._step_launch
        if cached is None or cached[0][0] != key or any(a is not b for a, b in zip(cached[0][1], arrays)):
            launch = record_launch(
                make_binocular_gaze_step(self._mode_code),
                self.num_envs,
                [
                    wp.from_torch(actions),
                    joint_pos,
                    limits,
                    self._joints,
                    self._signs,
                    mounts,
                    self.baseline,
                    self.cfg.step_size,
                    self.cfg.gaze_limit,
                    self.cfg.vergence_limit,
                    (self.cfg.rig.max_step_deg if self.rig is not None else 0.0) / self.field_radius_deg,
                    self._env.step_dt,
                    self.field_radius_deg,
                ],
                [
                    self._held,
                    self._held_valid,
                    self._version,
                    left.gaze_angles,
                    right.gaze_angles,
                    self._state,
                    self._targets,
                ],
                self.device,
            )
            self._step_launch = ((key, arrays), launch)
        self._raw = actions
        self._step_launch[1].launch(warp_stream(self.device))
        left.set_gaze_angles(left.gaze_angles.torch)
        right.set_gaze_angles(right.gaze_angles.torch)

    def apply_actions(self):
        if self.rig is not None:
            self.rig.actuators.target_command.set_position_index(value=self._targets.torch, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        # The action manager zeroes the reset environments' actions itself.
        indexed = env_ids is not None
        count = self.num_envs
        if indexed:
            if isinstance(env_ids, slice):
                env_ids = torch.arange(*env_ids.indices(self.num_envs), device=self.device)
            count = len(env_ids)
            if count == 0:
                return
            self._reset_ids.torch[:count] = torch.as_tensor(env_ids, device=self.device)
        left, right = self.left, self.right
        launch = self._reset_launches.get(indexed)
        if launch is None:
            default = self._no_joints if self.rig is None else self.rig.data.default_joint_pos.warp
            launch = self._reset_launches[indexed] = record_launch(
                binocular_gaze_reset_kernel,
                self.num_envs,
                [
                    self._reset_ids,
                    int(indexed),
                    default,
                    self._joints,
                    self._signs,
                    self.field_radius_deg,
                    int(self.rig is not None),
                    int(self._mode_code == HYBRID),
                ],
                [
                    self._targets,
                    self._held_valid,
                    self._version,
                    left.gaze_angles,
                    right.gaze_angles,
                    self._state,
                ],
                self.device,
            )
        launch.set_dim(count)
        launch.launch(warp_stream(self.device))
        left.set_gaze_angles(left.gaze_angles.torch)
        right.set_gaze_angles(right.gaze_angles.torch)

    def head_and_target(self) -> ProxyArray:
        """The head's measured (tilt, pan) and the (tilt, pan) its joint targets command, in field radii, shape
        (num_envs,) of ``vec4f``; zero but in physical gaze, the one mode whose version actions turn the head."""
        if self._mode_code == PHYSICAL:
            # Reading the property refreshes the rig's joint buffer.
            joint_pos = self.rig.data.joint_pos.warp
            if self._heads_launch is None or self._heads_launch[0] is not joint_pos:
                launch = record_launch(
                    head_and_target_kernel,
                    self.num_envs,
                    [joint_pos, self._targets, self._joints, self._signs, self.field_radius_deg],
                    [self._heads],
                    self.device,
                )
                self._heads_launch = (joint_pos, launch)
            self._heads_launch[1].launch(warp_stream(self.device))
        return self._heads

    def teacher_labels(self, points: wp.array, labels: wp.array) -> None:
        """Write the teacher's labels (version tilt, version pan, vergence, valid) toward world ``points`` [m].

        Args:
            points: One world point per environment, ``vec3f``.
            labels: The output, shape (num_envs, 4).
        """
        left, right = self.left, self.right
        # Both eyes read their poses, without capturing an image.
        mounts = left.update_mount()
        right.update_mount()
        heads = self.head_and_target()
        left_positions, right_positions = left.mount_pos_w.warp, right.mount_pos_w.warp
        sources = (points, labels, mounts, left_positions, right_positions)
        if self._teacher_launch is None or any(a is not b for a, b in zip(self._teacher_launch[0], sources)):
            launch = record_launch(
                binocular_teacher_kernel,
                self.num_envs,
                [
                    points,
                    left_positions,
                    right_positions,
                    mounts,
                    self._version,
                    self._state,
                    heads,
                    self._held,
                    self._held_valid,
                    self._mode_code,
                    left.image_frame,
                    self.field_radius_deg,
                    self.cfg.step_size,
                    self.cfg.gaze_limit,
                    self.cfg.vergence_limit,
                    left.projection_params,
                    right.projection_params,
                ],
                [labels],
                self.device,
            )
            self._teacher_launch = (sources, launch)
        self._teacher_launch[1].launch(warp_stream(self.device))

    def _baseline(self, mount: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> wp.vec3f:
        """The unit direction from the left eye to the right in the upright frame; it must cross the optical axes."""
        local = quat_apply_inverse(mount[None], (right - left)[None])[0]
        upright = torch.tensor(_rows(self.left.image_frame), device=mount.device).T @ local
        length = float(upright.norm())
        if length < 1e-6 or abs(float(upright[2])) > 0.1 * length:
            raise ValueError("The eyes of a binocular gaze need a baseline across their optical axes.")
        return wp.vec3f(*(upright / length).tolist())


class binocular_gaze_teacher_labels(ManagerTermBase):
    """The teacher's binocular gaze action toward a target point, and its validity, shape (num_envs, 4).

    The target is a point fixed to an asset (``target_cfg``), or any world point each environment supplies
    (``target_w``), as for :class:`~isaaclab_fovi.mdp.gaze_teacher_labels`.

    A training label, never a policy input.

    Args:
        target_cfg: The asset, and optionally one body by ``body_names``; give this or ``target_w``.
        offset: The point in the root's or the body's frame [m]; along the world axes from ``target_w``.
        action_name: The binocular gaze action term.
        target_w: A function of the environment that returns each environment's target point in the world [m], shape
            (num_envs, 3); give this or ``target_cfg``.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._gaze: BinocularGazeAction = env.action_manager.get_term(cfg.params.get("action_name", "gaze"))
        self._points = wp.zeros(env.num_envs, dtype=wp.vec3f, device=env.device)
        self._labels = ProxyArray(wp.zeros((env.num_envs, 4), dtype=wp.float32, device=env.device))

    def __call__(
        self,
        env: ManagerBasedEnv,
        target_cfg: SceneEntityCfg | None = None,
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        action_name: str = "gaze",
        target_w: Callable[[ManagerBasedEnv], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if (target_cfg is None) == (target_w is None):
            raise ValueError("binocular_gaze_teacher_labels needs one target: target_cfg or target_w.")
        if target_w is not None:
            points = target_w(env)
            if tuple(points.shape) != (env.num_envs, 3):
                raise ValueError(
                    f"target_w must return points of shape ({env.num_envs}, 3), got {tuple(points.shape)}."
                )
            wp.to_torch(self._points).copy_(points + points.new_tensor(offset))
            self._gaze.teacher_labels(self._points, self._labels.warp)
            return self._labels.torch
        data = env.scene[target_cfg.name].data
        body = target_body_index(target_cfg)
        if body is None:
            positions, orientations = data.root_pos_w.warp, data.root_quat_w.warp
        else:
            positions = wp.from_torch(data.body_pos_w.torch[:, body], dtype=wp.vec3f)
            orientations = wp.from_torch(data.body_quat_w.torch[:, body], dtype=wp.quatf)
        wp.launch(
            offset_points_kernel,
            env.num_envs,
            [positions, orientations, wp.vec3f(*offset)],
            [self._points],
            device=str(env.device),
            stream=warp_stream(env.device),
        )
        self._gaze.teacher_labels(self._points, self._labels.warp)
        return self._labels.torch


def _rows(matrix: wp.mat33f) -> list[list[float]]:
    """A 3 x 3 Warp matrix as nested lists of rows."""
    return [[float(matrix[row][col]) for col in range(3)] for row in range(3)]
