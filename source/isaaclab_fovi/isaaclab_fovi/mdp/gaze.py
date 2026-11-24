# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Active gaze for a foveated camera: a gaze action, its observation, and a teacher's gaze labels.

Gaze angles are (tilt, pan) [deg] from the physical camera's optical axis: positive tilt looks up and positive pan
looks left, as for :meth:`~isaaclab_fovi.sensors.FoveatedCamera.set_gaze_angles`. The policy moves and observes the
gaze in field radii, half the retina's field of view. Camera-frame directions follow the ROS (OpenCV) convention:
x right, y down, z forward.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING, ClassVar

import torch
import warp as wp

from isaaclab.managers import ActionTerm, ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.warp import ProxyArray

from ..sensors.gaze_kernels import (
    DEG_TO_RAD,
    CameraParams,
    project_direction,
    project_directions,
    record_launch,
    warp_stream,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from ..sensors import FoveatedCamera
    from .gaze_cfg import GazeActionCfg

RAD_TO_DEG = 180.0 / math.pi

PHYSICAL, HYBRID, DUAL = 1, 2, 3
"""Codes of the gaze modes that turn a rig, for :func:`make_rig_gaze_step`."""

RIG_MODES = {"physical": PHYSICAL, "hybrid": HYBRID, "dual": DUAL}
"""The rig modes' codes by :attr:`~isaaclab_fovi.mdp.GazeActionCfg.mode`."""


@wp.func
def clamp_unit(a: wp.vec2f) -> wp.vec2f:
    """Each coordinate of an action clamped to [-1, 1]."""
    return wp.vec2f(wp.clamp(a[0], -1.0, 1.0), wp.clamp(a[1], -1.0, 1.0))


@wp.func
def within(g: wp.vec2f, limit: wp.float32) -> wp.vec2f:
    """A gaze scaled back onto the circle of radius ``limit`` when it leaves it."""
    return g * (limit / wp.max(wp.length(g), limit))


@wp.func
def gaze_direction(g: wp.vec2f, field_radius_deg: wp.float32) -> wp.vec3f:
    """Upright-frame unit direction of a gaze (tilt, pan) in field radii."""
    tilt = g[0] * field_radius_deg * wp.float32(DEG_TO_RAD)
    pan = g[1] * field_radius_deg * wp.float32(DEG_TO_RAD)
    return wp.vec3f(-wp.cos(tilt) * wp.sin(pan), -wp.sin(tilt), wp.cos(tilt) * wp.cos(pan))


@wp.func
def gaze_of(u: wp.vec3f, field_radius_deg: wp.float32) -> wp.vec2f:
    """Gaze (tilt, pan) in field radii of an upright-frame direction."""
    tilt = -wp.atan2(u[1], wp.sqrt(u[0] * u[0] + u[2] * u[2])) * wp.float32(RAD_TO_DEG)
    pan = -wp.atan2(u[0], u[2]) * wp.float32(RAD_TO_DEG)
    return wp.vec2f(tilt, pan) / field_radius_deg


@wp.func
def head_angles(joint_pos: wp.array2d(dtype=wp.float32), i: wp.int32, joints: wp.vec2i, signs: wp.vec2f) -> wp.vec2f:
    """The head's (tilt, pan) [rad], left and up positive, from the rig's (pan, tilt) ``joints``."""
    return wp.vec2f(signs[1] * joint_pos[i, joints[1]], signs[0] * joint_pos[i, joints[0]])


@wp.func
def rig_gaze(head_rad: wp.vec2f, soft: wp.vec2f, field_radius_deg: wp.float32) -> wp.vec2f:
    """Gaze (tilt, pan) in field radii, from the rig's zero pose, of a software gaze ``soft`` [field radii] seen
    through a head turned by ``head_rad`` (tilt, pan) [rad].

    The head pans and then tilts, as the retina does within it, so their rotations compose rather than add.
    """
    d = gaze_direction(soft, field_radius_deg)
    c, s = wp.cos(head_rad[0]), wp.sin(head_rad[0])
    d = wp.vec3f(d[0], c * d[1] - s * d[2], s * d[1] + c * d[2])
    c, s = wp.cos(head_rad[1]), wp.sin(head_rad[1])
    d = wp.vec3f(c * d[0] - s * d[2], d[1], s * d[0] + c * d[2])
    return gaze_of(d, field_radius_deg)


@wp.func
def write_head_targets(
    targets: wp.array2d(dtype=wp.float32),
    limits: wp.array2d(dtype=wp.vec2f),
    i: wp.int32,
    joints: wp.vec2i,
    signs: wp.vec2f,
    base: wp.vec2f,
    turn_rad: wp.vec2f,
):
    """Set the rig's joint targets (pan, tilt) to ``base`` (pan, tilt) [rad] turned by ``turn_rad`` (tilt, pan)
    [rad], within the joints' limits."""
    pan_limit = limits[i, joints[0]]
    tilt_limit = limits[i, joints[1]]
    targets[i, 0] = wp.clamp(base[0] + signs[0] * turn_rad[1], pan_limit[0], pan_limit[1])
    targets[i, 1] = wp.clamp(base[1] + signs[1] * turn_rad[0], tilt_limit[0], tilt_limit[1])


@wp.func
def teach(
    point: wp.vec3f,
    position: wp.vec3f,
    mount: wp.quatf,
    g: wp.vec2f,
    start: wp.vec2f,
    head: wp.vec4f,
    physical: wp.int32,
    frame: wp.mat33f,
    field_radius_deg: wp.float32,
    step_size: wp.float32,
    gaze_limit: wp.float32,
    camera: CameraParams,
) -> wp.vec4f:
    """The teacher's action (x, y) toward a point, 1 for a valid target or else 0, and the gaze's error [deg].

    ``g`` is the software gaze in field radii as the camera looks now, from which the error is measured, and ``start``
    the software gaze the next step moves from: ``g`` itself, or in hybrid gaze ``g`` re-aimed at its held fixation
    for the head's turn. In physical gaze (``physical``) the action moves the head instead: ``head`` holds its
    measured and commanded (tilt, pan) in field radii, and the action takes the commanded head to the pose that faces
    the point, so it does not repeat a turn the head has yet to make. ``frame`` carries the upright image frame, in
    which gazes are measured, into the physical camera's frame.
    """
    d = wp.normalize(wp.quat_rotate_inv(mount, point - position))
    # The fixation reaches the physical camera at frame * gaze * z, so the gaze looks along frame^T d.
    u = wp.transpose(frame) * d
    desired = gaze_of(u, field_radius_deg)
    ok = project_direction(d, camera)[2] > 0.5 and wp.length(desired) <= gaze_limit
    step = wp.vec2f(0.0, 0.0)
    if ok:
        if physical != 0:
            measured = wp.vec2f(head[0], head[1]) * field_radius_deg * wp.float32(DEG_TO_RAD)
            facing = rig_gaze(measured, desired, field_radius_deg)
            step = clamp_unit((facing - wp.vec2f(head[2], head[3])) / step_size)
        else:
            step = clamp_unit((desired - start) / step_size)
    looking = gaze_direction(g, field_radius_deg)
    # atan2 stays accurate near perfect alignment, unlike acos of the dot product.
    error = wp.atan2(wp.length(wp.cross(looking, u)), wp.dot(looking, u)) * wp.float32(RAD_TO_DEG)
    return wp.vec4f(step[0], step[1], wp.where(ok, 1.0, 0.0), error)


@wp.kernel(enable_backward=False)
def teacher_labels_kernel(
    target_poses: wp.array(dtype=wp.transformf),
    offset: wp.vec3f,
    positions: wp.array(dtype=wp.vec3f),
    mounts: wp.array(dtype=wp.quatf),
    angles_deg: wp.array(dtype=wp.vec2f),
    heads: wp.array(dtype=wp.vec4f),
    held: wp.array(dtype=wp.vec3f),
    held_valid: wp.array(dtype=wp.int32),
    mode: wp.int32,
    frame: wp.mat33f,
    field_radius_deg: wp.float32,
    step_size: wp.float32,
    gaze_limit: wp.float32,
    camera: CameraParams,
    labels: wp.array(dtype=wp.vec3f),
    log: wp.array(dtype=wp.float32),
):
    """Labels (action x, action y, valid) toward the point ``offset`` fixed to each target's frame; and the mean gaze
    error to valid targets [deg] with the fraction of valid targets.

    ``mounts`` holds each camera's mount (ROS convention); ``angles_deg`` the camera's stored gaze angles [deg];
    ``heads`` the head's measured and commanded (tilt, pan) in field radii, which physical gaze labels from; ``held``
    the world fixation hybrid gaze holds, once ``held_valid`` is set, which the next step re-aims the gaze at first.
    ``mode`` is the rig mode's code, or 0 without a rig; the arrays a mode does not use may be empty. One block labels
    every environment and reduces the logs itself, so the launch needs no zeroed accumulators.
    """
    _block, lane = wp.tid()
    count = target_poses.shape[0]
    error_sum = wp.float32(0.0)
    valid_count = wp.float32(0.0)
    i = lane
    while i < count:
        point = wp.transform_point(target_poses[i], offset)
        gaze = angles_deg[i] / field_radius_deg
        start = gaze
        if mode == HYBRID:
            if held_valid[i] != 0:
                start = gaze_of(wp.transpose(frame) * wp.quat_rotate_inv(mounts[i], held[i]), field_radius_deg)
        head = wp.vec4f(0.0, 0.0, 0.0, 0.0)
        if mode == PHYSICAL:
            head = heads[i]
        result = teach(
            point,
            positions[i],
            mounts[i],
            gaze,
            start,
            head,
            wp.where(mode == PHYSICAL, 1, 0),
            frame,
            field_radius_deg,
            step_size,
            gaze_limit,
            camera,
        )
        labels[i] = wp.vec3f(result[0], result[1], result[2])
        if result[2] > 0.5:
            error_sum += result[3]
            valid_count += 1.0
        i += wp.block_dim()
    errors = wp.tile_sum(wp.tile(error_sum))
    valid = wp.tile_sum(wp.tile(valid_count))
    if lane == 0:
        log[0] = errors[0] / wp.max(valid[0], 1.0)
        log[1] = valid[0] / wp.float32(count)


@wp.kernel(enable_backward=False)
def gaze_reset_kernel(
    ids: wp.array(dtype=wp.int32),
    use_ids: wp.int32,
    start: wp.array(dtype=wp.vec2f),
    use_start: wp.int32,
    field_radius_deg: wp.float32,
    angles_deg: wp.array(dtype=wp.vec2f),
    state: wp.array(dtype=wp.vec4f),
):
    """Put each reset environment's gaze at its start, or on the optical axis, at rest.

    ``state`` holds each gaze and its velocity, in field radii and field radii per second.
    """
    k = wp.tid()
    i = k
    if use_ids != 0:
        i = ids[k]
    g = wp.vec2f(0.0, 0.0)
    if use_start != 0:
        g = start[k]
    angles_deg[i] = g * field_radius_deg
    state[i] = wp.vec4f(g[0], g[1], 0.0, 0.0)


@wp.kernel(enable_backward=False)
def gaze_step_kernel(
    actions: wp.array(dtype=wp.vec2f),
    state: wp.array(dtype=wp.vec4f),
    angles_deg: wp.array(dtype=wp.vec2f),
    step_size: wp.float32,
    gaze_limit: wp.float32,
    step_dt: wp.float32,
    field_radius_deg: wp.float32,
):
    """Move each gaze by its clamped action and keep it within the limit; record the gaze, velocity and angles.

    The step starts from the camera's stored ``angles_deg``, which own the gaze. ``state`` records each gaze and its
    velocity, in field radii and field radii per second.
    """
    i = wp.tid()
    a = actions[i]
    previous = angles_deg[i] / field_radius_deg
    current = within(previous + clamp_unit(a) * step_size, gaze_limit)
    rate = (current - previous) / step_dt
    angles_deg[i] = current * field_radius_deg
    state[i] = wp.vec4f(current[0], current[1], rate[0], rate[1])


@cache
def make_rig_gaze_step(mode: int) -> wp.Kernel:
    """Return the gaze step of one rig mode: :data:`PHYSICAL`, :data:`HYBRID` or :data:`DUAL`."""
    if mode not in (PHYSICAL, HYBRID, DUAL):
        raise ValueError(f"Unknown rig gaze mode {mode!r}.")

    @wp.kernel(enable_backward=False, module="unique")
    def rig_gaze_step(
        actions: wp.array2d(dtype=wp.float32),
        joint_pos: wp.array2d(dtype=wp.float32),
        limits: wp.array2d(dtype=wp.vec2f),
        joints: wp.vec2i,
        signs: wp.vec2f,
        mounts: wp.array(dtype=wp.quatf),
        step_size: wp.float32,
        gaze_limit: wp.float32,
        head_step: wp.float32,
        step_dt: wp.float32,
        field_radius_deg: wp.float32,
        held: wp.array(dtype=wp.vec3f),
        held_valid: wp.array(dtype=wp.int32),
        angles_deg: wp.array(dtype=wp.vec2f),
        state: wp.array(dtype=wp.vec4f),
        targets: wp.array2d(dtype=wp.float32),
    ):
        """Move each camera's gaze with its rig's joints and, in hybrid and dual gaze, its software gaze.

        Gazes are (tilt, pan) in field radii, in the camera's frame, which needs no image roll; the rig's ``joints``
        are (pan, tilt) with ``signs`` that turn the camera left and up. ``mounts`` holds each camera's mount (ROS
        convention) as of its last pose update. ``head_step`` bounds the hybrid head target's offset from the measured
        joints, in field radii. Hybrid gaze holds each fixation as a world direction in ``held`` once ``held_valid``
        is set; the other modes leave those arrays alone, so they may be empty.
        Writes the software gaze ``angles_deg``, the joint ``targets`` (pan, tilt) and ``state``: the fixation's
        direction from the rig's zero pose and its velocity.
        """
        i = wp.tid()
        radius_rad = field_radius_deg * wp.float32(DEG_TO_RAD)
        head = head_angles(joint_pos, i, joints, signs)
        soft = angles_deg[i] / field_radius_deg
        # Commanded head steps add to the last target, so a lagging head still reaches every commanded step.
        base = wp.vec2f(targets[i, 0], targets[i, 1])
        turn = wp.vec2f(0.0, 0.0)
        if wp.static(mode == PHYSICAL):
            turn = clamp_unit(wp.vec2f(actions[i, 0], actions[i, 1])) * step_size
        if wp.static(mode == DUAL):
            turn = clamp_unit(wp.vec2f(actions[i, 0], actions[i, 1])) * step_size
            soft = within(soft + clamp_unit(wp.vec2f(actions[i, 2], actions[i, 3])) * step_size, gaze_limit)
        if wp.static(mode == HYBRID):
            if held_valid[i] != 0:
                # The head turned since the last step: aim the software gaze back at the fixation it holds in the world.
                soft = gaze_of(wp.quat_rotate_inv(mounts[i], held[i]), field_radius_deg)
            soft = within(soft + clamp_unit(wp.vec2f(actions[i, 0], actions[i, 1])) * step_size, gaze_limit)
            turn = wp.vec2f(wp.clamp(soft[0], -head_step, head_step), wp.clamp(soft[1], -head_step, head_step))
            held[i] = wp.quat_rotate(mounts[i], gaze_direction(soft, field_radius_deg))
            held_valid[i] = 1
            base = wp.vec2f(joint_pos[i, joints[0]], joint_pos[i, joints[1]])
        write_head_targets(targets, limits, i, joints, signs, base, turn * radius_rad)
        angles_deg[i] = soft * field_radius_deg
        gaze = rig_gaze(head, soft, field_radius_deg)
        previous = wp.vec2f(state[i][0], state[i][1])
        rate = (gaze - previous) / step_dt
        state[i] = wp.vec4f(gaze[0], gaze[1], rate[0], rate[1])

    return rig_gaze_step


@wp.kernel(enable_backward=False)
def rig_gaze_reset_kernel(
    ids: wp.array(dtype=wp.int32),
    use_ids: wp.int32,
    default_joint_pos: wp.array2d(dtype=wp.float32),
    joints: wp.vec2i,
    signs: wp.vec2f,
    field_radius_deg: wp.float32,
    clear_held: wp.int32,
    targets: wp.array2d(dtype=wp.float32),
    held_valid: wp.array(dtype=wp.int32),
    angles_deg: wp.array(dtype=wp.vec2f),
    state: wp.array(dtype=wp.vec4f),
):
    """Put each reset environment's joint targets at their defaults and its software gaze on the optical axis.

    The gaze state starts at the default head's angles, at rest; hybrid gaze (``clear_held``) drops its fixation.
    """
    k = wp.tid()
    i = k
    if use_ids != 0:
        i = ids[k]
    targets[i, 0] = default_joint_pos[i, joints[0]]
    targets[i, 1] = default_joint_pos[i, joints[1]]
    if clear_held != 0:
        held_valid[i] = 0
    angles_deg[i] = wp.vec2f(0.0, 0.0)
    head = head_angles(default_joint_pos, i, joints, signs) / (field_radius_deg * wp.float32(DEG_TO_RAD))
    state[i] = wp.vec4f(head[0], head[1], 0.0, 0.0)


@wp.kernel(enable_backward=False)
def head_and_target_kernel(
    joint_pos: wp.array2d(dtype=wp.float32),
    targets: wp.array2d(dtype=wp.float32),
    joints: wp.vec2i,
    signs: wp.vec2f,
    field_radius_deg: wp.float32,
    heads: wp.array(dtype=wp.vec4f),
):
    """Each head's measured (tilt, pan) and the (tilt, pan) its joint targets (pan, tilt) command, in field radii."""
    i = wp.tid()
    radius_rad = field_radius_deg * wp.float32(DEG_TO_RAD)
    head = head_angles(joint_pos, i, joints, signs) / radius_rad
    target = head_angles(targets, i, wp.vec2i(0, 1), signs) / radius_rad
    heads[i] = wp.vec4f(head[0], head[1], target[0], target[1])


def target_body_index(target_cfg: SceneEntityCfg) -> int | None:
    """The body a teacher's point is fixed to, or None for the asset's root.

    :class:`~isaaclab.managers.SceneEntityCfg` resolves ``body_ids`` to a slice when the named bodies cover every body
    of the asset, as naming the only body of a rigid object does; the slice starts at the first of them.
    """
    if target_cfg.body_names is None:
        return None
    ids = target_cfg.body_ids
    if isinstance(ids, slice):
        return ids.start or 0
    return int(ids[0])


def gaze_state(env: ManagerBasedEnv, action_name: str = "gaze") -> torch.Tensor:
    """Gaze and its velocity, in field radii and field radii per second, shape (num_envs, 4)."""
    term: GazeAction = env.action_manager.get_term(action_name)
    return term.state


class GazeAction(ActionTerm):
    """Moves a foveated camera's gaze by bounded steps in (tilt, pan) order.

    Actions, the limit and the observed gaze are in field radii. An action of 1 changes a coordinate by
    ``step_size``, and the software gaze stays within ``gaze_limit`` of the optical axis. Gazes are measured in the
    upright image frame, which :attr:`~isaaclab_fovi.sensors.FoveatedCameraCfg.image_rotation_deg` rolls against a
    rolled mount.

    :attr:`~isaaclab_fovi.mdp.GazeActionCfg.mode` sets what the actions turn, two (tilt, pan) actions in every mode
    but dual gaze, which takes four:

    * ``"software"``: the retina, at once, relative to the camera's optical axis; the camera itself may move with a
      tracked mount.
    * ``"physical"``: the joints of the rig that carries the camera, by moving their targets from the last targets;
      the retina stays on the optical axis, and the gaze follows as the joints reach their targets.
    * ``"hybrid"``: the retina at once, while the rig's joints follow, by at most
      :attr:`~isaaclab_fovi.mdp.GazeRigCfg.max_step_deg` per step. Each step the software gaze is re-aimed at the
      fixation it holds in the world, so as the head catches up the retina returns toward the optical axis and the
      fixation stays put.
    * ``"dual"`` (experimental): the rig's joints as in physical gaze and then the retina as in software gaze, each
      moved independently by the policy.

    The modes that turn a rig need a camera without image roll, whose tilt and pan follow the rig's joints.

    The software gaze starts each episode on the optical axis, or, in software gaze, anywhere the physical camera sees
    within the limit when ``random_reset`` is set.

    With ``mount="fixed"`` the camera keeps the orientation it had when the term was created, and the teacher uses
    that mount; turning the camera's mount afterwards raises. With ``mount="tracked"`` the camera may ride a moving
    body, for example a robot's head, and the teacher reads each camera's current mount, so software gaze adds to
    hardware gaze.

    The camera's stored gaze angles own the software gaze: each step and the teacher start from them, so a gaze set
    directly with :meth:`~isaaclab_fovi.sensors.FoveatedCamera.set_gaze_angles` carries on, and in hybrid gaze starts a
    new fixation. :attr:`state`, the observation, is the gaze and velocity as of the last step or reset; with a rig,
    the gaze is the fixation's direction from the rig's zero pose, the head's turn composed with the software gaze.
    """

    cfg: GazeActionCfg

    is_graphable: ClassVar[bool] = True
    """The gaze moves in :meth:`process_actions`, once per environment step; :meth:`apply_actions` at most writes a
    rig's joint targets from a fixed buffer, so physics may record its decimation loop into a CUDA graph."""

    def __init__(self, cfg: GazeActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        camera = self.camera
        self.tracked_mount = cfg.mount == "tracked"
        if not self.tracked_mount and camera.cfg.prim_path.rpartition("/")[0] not in (
            "{ENV_REGEX_NS}",
            env.scene.env_regex_ns,
        ):
            # A body's motion turns its camera without set_world_poses, which the fixed mount could not detect.
            raise ValueError(
                f"A fixed gaze mount needs a camera mounted on the environment, got {camera.cfg.prim_path!r};"
                " use mount='tracked' for a camera on a moving body."
            )
        # The camera's own geometry, so the teacher and the sensor mask share one projection.
        frame = camera.image_frame
        self._frame_torch = torch.tensor(
            [[frame[row][col] for col in range(3)] for row in range(3)], device=self.device
        )
        self.field_radius_deg = camera.sampling_coords.fov / 2
        """Half the retina's field of view [deg]."""
        self._fixed_mount = None
        if not self.tracked_mount:
            # The fixed-mount teacher keeps the camera's mount as it is now; only later turns invalidate it.
            mount = camera.ungazed_quat_w_ros()
            self._fixed_mount = ProxyArray(wp.from_torch(mount.contiguous(), dtype=wp.quatf))
            self._mount_turns = camera.mount_turns
        self._raw: torch.Tensor | None = None
        """Gaze actions of the last step, as supplied to :meth:`process_actions`."""
        self._state = ProxyArray(wp.zeros(self.num_envs, dtype=wp.vec4f, device=self.device))
        """Gaze and its velocity as of the last step or reset, the gaze observation."""
        self._velocity = self._state.torch[:, 2:]
        self._step_launch: tuple[tuple, wp.Launch] | None = None
        # Each reset copies its environments, and their start gazes if random, into these and replays one launch.
        self._reset_ids = ProxyArray(wp.zeros(self.num_envs, dtype=wp.int32, device=self.device))
        self._reset_start = ProxyArray(wp.zeros(self.num_envs, dtype=wp.vec2f, device=self.device))
        self._reset_launches: dict[tuple[bool, bool], wp.Launch] = {}
        self._heads = ProxyArray(wp.zeros(self.num_envs, dtype=wp.vec4f, device=self.device))
        """The head's measured and commanded (tilt, pan) [field radii], which the teacher labels physical gaze from."""
        self._own_gaze_writes = 0
        """The camera's :attr:`~isaaclab_fovi.sensors.FoveatedCamera.gaze_writes` after this term's last write."""
        # Without a rig there is no fixation to hold; _init_rig replaces these for the rig modes.
        self._mode_code = 0
        self._held = ProxyArray(wp.zeros(0, dtype=wp.vec3f, device=self.device))
        self._held_valid = ProxyArray(wp.zeros(0, dtype=wp.int32, device=self.device))
        self.rig = None
        """The articulation that turns the camera, for every mode but software gaze."""
        if cfg.mode != "software":
            if camera.cfg.image_rotation_deg != 0:
                raise ValueError(
                    f"GazeActionCfg.mode={cfg.mode!r} turns the camera with a rig's pan and tilt joints, which a rolled"
                    f" image's tilt and pan do not follow; got image_rotation_deg={camera.cfg.image_rotation_deg!r}."
                )
            self._init_rig(env)

    @property
    def action_dim(self) -> int:
        return 4 if self.cfg.mode == "dual" else 2

    @property
    def camera(self) -> FoveatedCamera:
        """The foveated camera whose gaze the term moves."""
        return self._asset

    @property
    def raw_actions(self) -> torch.Tensor:
        if self._raw is not None:
            return self._raw
        return torch.zeros(self.num_envs, self.action_dim, dtype=torch.float32, device=self.device)

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.gaze

    @property
    def gaze(self) -> torch.Tensor:
        """Current gaze in field radii, from the camera's stored angles, shape (num_envs, 2)."""
        return self.angles_deg / self.field_radius_deg

    @property
    def velocity(self) -> torch.Tensor:
        """Gaze velocity in field radii per second, shape (num_envs, 2)."""
        return self._velocity

    @property
    def state(self) -> torch.Tensor:
        """Gaze and its velocity, in field radii and field radii per second, shape (num_envs, 4)."""
        return self._state.torch

    @property
    def angles_deg(self) -> torch.Tensor:
        """Current gaze angles [deg], the camera's stored ones, shape (num_envs, 2)."""
        return self.camera.gaze_angles.torch

    def process_actions(self, actions: torch.Tensor):
        """Move the gaze by the supplied actions, shape (num_envs, :attr:`action_dim`): 2, or 4 in dual gaze.

        The launch reads ``actions`` in place. The action manager hands the term the same slice of its buffer each
        step, so the launch is recorded once; another buffer records a launch that reads it.
        """
        if actions.dtype != torch.float32:
            actions = actions.float()
        if self.rig is not None:
            self._process_rig_actions(actions)
            return
        key = (actions.data_ptr(), actions.stride(), actions.device)
        if self._step_launch is None or self._step_launch[0] != key:
            launch = record_launch(
                gaze_step_kernel,
                self.num_envs,
                [
                    wp.from_torch(actions, dtype=wp.vec2f),
                    self._state,
                    self.camera.gaze_angles,
                    self.cfg.step_size,
                    self.cfg.gaze_limit,
                    self._env.step_dt,
                    self.field_radius_deg,
                ],
                device=self.device,
            )
            self._step_launch = (key, launch)
        self._raw = actions
        self._step_launch[1].launch(warp_stream(self.device))
        self.camera.set_gaze_angles(self.camera.gaze_angles.torch)

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
        if self.rig is not None:
            self._reset_rig(indexed, count)
            return
        random_start = self.cfg.random_reset
        if random_start:
            # A square root gives uniform area density in the gaze disc.
            radius = self.cfg.gaze_limit * torch.rand(count, device=self.device).sqrt()
            angle = 2 * math.pi * torch.rand(count, device=self.device)
            gaze = torch.stack((radius * angle.sin(), radius * angle.cos()), dim=-1)
            tilt, pan = (gaze * self.field_radius_deg).deg2rad().unbind(-1)
            direction = torch.stack((-tilt.cos() * pan.sin(), -tilt.sin(), tilt.cos() * pan.cos()), -1)
            visible = self.visible(direction)
            self._reset_start.torch[:count] = torch.where(visible[:, None], gaze, 0.0)
        angles = self.camera.gaze_angles
        launch = self._reset_launches.get((indexed, random_start))
        if launch is None:
            launch = self._reset_launches[(indexed, random_start)] = record_launch(
                gaze_reset_kernel,
                self.num_envs,
                [self._reset_ids, int(indexed), self._reset_start, int(random_start), self.field_radius_deg],
                [angles, self._state],
                self.device,
            )
        launch.set_dim(count)
        launch.launch(warp_stream(self.device))
        # The camera keeps reading the same buffer, which every other environment already holds.
        self.camera.set_gaze_angles(angles.torch)

    def visible(self, direction: torch.Tensor) -> torch.Tensor:
        """Whether directions in the upright gaze frame fall inside the physical camera's image."""
        return project_directions(direction @ self._frame_torch.T, self.camera.projection_params)[1]

    def teacher_mounts(self) -> ProxyArray:
        """Each camera's mount (x, y, z, w), ROS convention, that the teacher aims from: the current one when
        tracked, else the one captured when the term was created.

        Raises:
            RuntimeError: The camera's mount turned while the gaze assumes a fixed mount.
        """
        if self.tracked_mount:
            # The mount as the camera stands now, though no term has read its image yet.
            return self.camera.update_mount()
        if self.camera.mount_turns != self._mount_turns:
            raise RuntimeError(
                "The gaze camera's mount turned, so the fixed-mount teacher would aim from its first orientation;"
                " use GazeActionCfg.mount='tracked'."
            )
        return self._fixed_mount

    def held_fixation(self) -> tuple[ProxyArray, ProxyArray]:
        """The world fixation hybrid gaze holds (ROS world directions) and whether it holds one, per environment; both
        empty but in hybrid gaze. The next step re-aims the software gaze at it, so the teacher labels from there."""
        return self._held, self._held_valid

    def head_and_target(self) -> ProxyArray:
        """The head's measured (tilt, pan) and the (tilt, pan) its joint targets command, in field radii, shape
        (num_envs,) of ``vec4f``; zero but in physical gaze, the one mode whose actions turn the head.

        The teacher labels physical gaze from them, so it does not command a turn again while the head catches up.
        """
        if self.rig is not None and self._mode_code == PHYSICAL:
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

    def _init_rig(self, env: ManagerBasedEnv) -> None:
        """Find the rig's joints and allocate the joint targets and, in hybrid gaze, the fixation it holds."""
        rig_cfg = self.cfg.rig
        self.rig = env.scene[rig_cfg.asset_name]
        joint_ids, _ = self.rig.find_joints(list(rig_cfg.joint_names), preserve_order=True)
        self._joint_ids = list(joint_ids)
        self._joints = wp.vec2i(*self._joint_ids)
        self._signs = wp.vec2f(*rig_cfg.joint_signs)
        self._mode_code = RIG_MODES[self.cfg.mode]
        default = self.rig.data.default_joint_pos.torch[:, self._joint_ids]
        # Indexing the selected joints copies them, so the targets own their storage.
        self._targets = ProxyArray(wp.from_torch(default.contiguous()))
        """Joint position targets (pan, tilt) [rad] the rig drives to."""
        # Only hybrid gaze holds a fixation; the other modes' kernels never touch these, so they stay empty.
        held = self.num_envs if self._mode_code == HYBRID else 0
        self._held = ProxyArray(wp.zeros(held, dtype=wp.vec3f, device=self.device))
        """The world fixation hybrid gaze holds (ROS world directions), once ``_held_valid`` is set."""
        self._held_valid = ProxyArray(wp.zeros(held, dtype=wp.int32, device=self.device))
        self._rig_launch: tuple[tuple, wp.Launch] | None = None
        self._rig_reset_launches: dict[bool, wp.Launch] = {}
        self._heads_launch: tuple[wp.array, wp.Launch] | None = None

    def _process_rig_actions(self, actions: torch.Tensor) -> None:
        """Move the gaze with the rig's joints and the software gaze, in one launch."""
        data = self.rig.data
        # Reading the properties refreshes the rig's joint buffers. Hybrid gaze aims from the mount as the head stands
        # now, which a lazily updated camera has not read yet.
        joint_pos, limits = data.joint_pos.warp, data.soft_joint_pos_limits.warp
        mounts = self.camera.update_mount() if self._mode_code == HYBRID else self.camera.mount_quat_w_ros
        if self._mode_code == HYBRID and self.camera.gaze_writes != self._own_gaze_writes:
            # A gaze set on the camera since this term's last write, even to the same angles, starts a new fixation.
            self._held_valid.torch.zero_()
        key = (actions.data_ptr(), actions.stride(), actions.device)
        arrays = (joint_pos, limits, mounts)
        cached = self._rig_launch
        if cached is None or cached[0][0] != key or any(a is not b for a, b in zip(cached[0][1], arrays)):
            launch = record_launch(
                make_rig_gaze_step(self._mode_code),
                self.num_envs,
                [
                    wp.from_torch(actions),
                    joint_pos,
                    limits,
                    self._joints,
                    self._signs,
                    mounts,
                    self.cfg.step_size,
                    self.cfg.gaze_limit,
                    self.cfg.rig.max_step_deg / self.field_radius_deg,
                    self._env.step_dt,
                    self.field_radius_deg,
                ],
                [self._held, self._held_valid, self.camera.gaze_angles, self._state, self._targets],
                self.device,
            )
            self._rig_launch = ((key, arrays), launch)
        self._raw = actions
        self._rig_launch[1].launch(warp_stream(self.device))
        self._write_gaze()

    def _reset_rig(self, indexed: bool, count: int) -> None:
        """Put the reset environments, the first ``count`` of :attr:`_reset_ids` if ``indexed``, at the rig's default
        joint targets with the software gaze on the optical axis.

        The articulation resets its own joints; the gaze state starts at the joints' default angles, at rest.
        """
        launch = self._rig_reset_launches.get(indexed)
        if launch is None:
            launch = self._rig_reset_launches[indexed] = record_launch(
                rig_gaze_reset_kernel,
                self.num_envs,
                [
                    self._reset_ids,
                    int(indexed),
                    self.rig.data.default_joint_pos.warp,
                    self._joints,
                    self._signs,
                    self.field_radius_deg,
                    int(self._mode_code == HYBRID),
                ],
                [self._targets, self._held_valid, self.camera.gaze_angles, self._state],
                self.device,
            )
        launch.set_dim(count)
        launch.launch(warp_stream(self.device))
        self._write_gaze()

    def _write_gaze(self) -> None:
        """Publish the camera's stored gaze, which the kernels wrote, and note the write as this term's own."""
        self.camera.set_gaze_angles(self.camera.gaze_angles.torch)
        self._own_gaze_writes = self.camera.gaze_writes


class gaze_teacher_labels(ManagerTermBase):
    """The teacher's gaze action toward a target point, and its validity, shape (num_envs, 3).

    The target is a point fixed to an asset (``target_cfg``), or any world point each environment supplies
    (``target_w``), for example the object a command picks among several.

    A training label, never a policy input. One launch labels every environment and logs the mean gaze error to valid
    targets and the fraction of valid targets.

    Args:
        target_cfg: The asset, and optionally one body by ``body_names``; give this or ``target_w``.
        offset: The point in the root's or the body's frame [m]; along the world axes from ``target_w``.
        action_name: The gaze action term.
        target_w: A function of the environment that returns each environment's target point in the world [m], shape
            (num_envs, 3); give this or ``target_cfg``.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._gaze: GazeAction = env.action_manager.get_term(cfg.params.get("action_name", "gaze"))
        self._labels = ProxyArray(wp.zeros(env.num_envs, dtype=wp.vec3f, device=env.device))
        self._log = ProxyArray(wp.zeros(2, dtype=wp.float32, device=env.device))
        """Mean gaze error to valid targets [deg] and the fraction of valid targets."""
        self._launch: tuple[tuple, wp.Launch] | None = None
        self._point_poses: ProxyArray | None = None
        """Poses at the points ``target_w`` returns, unturned, which the launch reads as it reads an asset's."""

    def __call__(
        self,
        env: ManagerBasedEnv,
        target_cfg: SceneEntityCfg | None = None,
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        action_name: str = "gaze",
        target_w: Callable[[ManagerBasedEnv], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if (target_cfg is None) == (target_w is None):
            raise ValueError("gaze_teacher_labels needs one target: target_cfg or target_w.")
        if target_w is not None:
            body_id = None
            poses = self._poses_at(target_w(env))
        else:
            data = env.scene[target_cfg.name].data
            # Reading the property refreshes the producer's poses.
            body_id = target_body_index(target_cfg)
            poses = data.root_pose_w.warp if body_id is None else data.body_pose_w.warp
        point = tuple(float(value) for value in offset)
        gaze = self._gaze
        mounts = gaze.teacher_mounts()
        heads = gaze.head_and_target()
        held, held_valid = gaze.held_fixation()
        camera_positions = gaze.camera.data.pos_w.warp
        # Warp retains the selected view and its backing array in the recorded launch.
        key = (id(poses), body_id, id(camera_positions), id(mounts), id(heads), point)
        if self._launch is None or self._launch[0] != key:
            # Warp runs CPU blocks as one thread.
            block = 1024 if torch.device(env.device).type == "cuda" else 1
            launch = record_launch(
                teacher_labels_kernel,
                (1, block),
                [
                    poses if body_id is None else poses[:, body_id],
                    wp.vec3f(*point),
                    camera_positions,
                    mounts,
                    gaze.camera.gaze_angles,
                    heads,
                    held,
                    held_valid,
                    RIG_MODES.get(gaze.cfg.mode, 0),
                    gaze.camera.image_frame,
                    gaze.field_radius_deg,
                    gaze.cfg.step_size,
                    gaze.cfg.gaze_limit,
                    gaze.camera.projection_params,
                ],
                [self._labels, self._log],
                env.device,
                block_dim=block,
            )
            self._launch = (key, launch)
        self._launch[1].launch(warp_stream(env.device))
        # Each step's log must survive the next write to this buffer.
        error, fraction = self._log.torch.clone()
        log = env.extras.setdefault("log", {})
        log["Gaze/target_error_deg"] = error
        log["Gaze/valid_target_fraction"] = fraction
        return self._labels.torch

    def _poses_at(self, points: torch.Tensor) -> wp.array:
        """World points as unturned poses in one buffer, shape (num_envs,) of ``transformf``, so the recorded launch
        reads them as it reads an asset's poses."""
        if tuple(points.shape) != (self.num_envs, 3):
            raise ValueError(f"target_w must return points of shape ({self.num_envs}, 3), got {tuple(points.shape)}.")
        if self._point_poses is None:
            self._point_poses = ProxyArray(wp.zeros(self.num_envs, dtype=wp.transformf, device=self.device))
            # Identity orientations (x, y, z, w).
            self._point_poses.torch[:, 6] = 1.0
        self._point_poses.torch[:, :3] = points
        return self._point_poses.warp
