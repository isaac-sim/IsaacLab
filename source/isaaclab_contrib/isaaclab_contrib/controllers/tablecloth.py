# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""GPU state machine and Newton IK for the fixed-base tablecloth example."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import newton
import numpy as np
import warp as wp
from isaaclab_newton.controllers.ik import (
    NewtonIKJointLimitObjectiveCfg,
    NewtonIKPoseObjectiveCfg,
    NewtonIKSolver,
    NewtonIKSolverCfg,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

from isaaclab_contrib.assets.tablecloth import (
    FINGER_JOINT_NAMES,
    H2_HAND_OFFSETS,
    H2_OPEN_JOINT_POS,
    H2_PINCH_JOINT_POS,
    HAND_OFFSETS,
)

PULL_DISTANCE = 0.40
H2_PULL_DISTANCE = 0.45
PULL_TIME_FACTOR = 1.875
POSE_COMPONENTS = 21

FINGERS_OPEN = (0.0, 0.0, 0.0, 0.0, 0.0)
# Tuck unused fingers while above the table so they do not obstruct the pinch.
FINGERS_CURLED = (0.0, 0.0, 0.0, 0.0, 1.50)
FINGERS_INSERTED = (0.0, 0.0, 0.75, 0.75, 1.50)
FINGERS_PREPINCHED = (0.0, 0.0, 0.737080, 0.713855, 1.50)
FINGERS_PINCHED = (1.0, 1.0, 0.737080, 0.713855, 1.50)

PHASE_DURATIONS = (0.50, 0.80, 0.60, 0.50, 0.60, 0.60, 0.30, 0.30)
H2_PHASE_DURATIONS = (0.08, 0.05, 0.05, 0.05, 0.35, 0.15, 0.30, 0.30)
PHASE_KEYFRAMES = ((0, 0), (0, 2), (2, 4), (4, 6), (6, 8), (8, 8), (8, 10), (10, 12))
H2_PHASE_KEYFRAMES = ((0, 0), (0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 12))
PHASE_FINGERS = (
    (FINGERS_OPEN, FINGERS_OPEN),
    (FINGERS_OPEN, FINGERS_CURLED),
    (FINGERS_CURLED, FINGERS_INSERTED),
    (FINGERS_INSERTED, FINGERS_INSERTED),
    (FINGERS_INSERTED, FINGERS_PREPINCHED),
    (FINGERS_PREPINCHED, FINGERS_PINCHED),
    (FINGERS_PINCHED, FINGERS_PINCHED),
    (FINGERS_PINCHED, FINGERS_PINCHED),
)
STATE_PULL = wp.constant(len(PHASE_DURATIONS))
STATE_HOLD = wp.constant(len(PHASE_DURATIONS) + 1)

# Positions are expressed in the fixed pelvis frame.
HAND_KEYFRAMES = (
    (0.27, 0.24, 0.140),
    (0.27, -0.24, 0.140),
    (0.43, 0.361, 0.060),
    (0.43, -0.372, 0.060),
    (0.43, 0.361, -0.0352),
    (0.43, -0.372, -0.0292),
    (0.530, 0.361, -0.0352),
    (0.530, -0.372, -0.0292),
    (0.530, 0.361, -0.0352),
    (0.530, -0.372, -0.0292),
    (0.500, 0.361, -0.0352),
    (0.500, -0.372, -0.0292),
    (0.500, 0.361, -0.0052),
    (0.500, -0.372, -0.0052),
)
HAND_ROTATIONS = (
    (-0.09022585, 0.46115433, 0.03007528, 0.88220828),
    (0.09023000, 0.46114998, -0.03008000, 0.88220997),
)
# Feasible corner-grasp attitudes for H1's five-DOF arms, calibrated with forward kinematics.
GRASP_HAND_ROTATIONS = (
    (-0.0679863393, 0.3806174397, 0.0725343823, 0.9193731546),
    (0.0912486315, 0.3623600602, -0.0816145539, 0.9239633083),
)
FINGER_CLOSED_VALUES = (
    1.273907,
    0.160957,
    0.369535,
    0.892908,
    1.2,
    1.2,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.192278,
    0.195421,
    0.400690,
    0.679765,
    1.2,
    1.2,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
)
FINGER_GROUPS = (0, 0, 0, 0, 2, 2, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 3, 3, 4, 4, 4, 4, 4, 4)
# Start with the native index fingertips beneath the hem, close the thumbs, then lift 2 cm.
H2_HAND_KEYFRAMES = (
    (0.5628056417, 0.2475379991, 0.1070655547),
    (0.5628101729, -0.2475441579, 0.1070715950),
    (0.5628056417, 0.2475379991, 0.1070655547),
    (0.5628101729, -0.2475441579, 0.1070715950),
    (0.5628056417, 0.2475379991, 0.1070655547),
    (0.5628101729, -0.2475441579, 0.1070715950),
    (0.5628056417, 0.2475379991, 0.1070655547),
    (0.5628101729, -0.2475441579, 0.1070715950),
    (0.5628056417, 0.2475379991, 0.1070655547),
    (0.5628101729, -0.2475441579, 0.1070715950),
    (0.5628056417, 0.2475379991, 0.1270655547),
    (0.5628101729, -0.2475441579, 0.1270715950),
    (0.5628056417, 0.2475379991, 0.1270655547),
    (0.5628101729, -0.2475441579, 0.1270715950),
)
# Control the complete wrist pose: a free roll can put the proximal fingers inside the table.
H2_READY_HAND_ROTATIONS = (
    (-0.9239897494, 0.0197845272, -0.3665561809, 0.1071824696),
    (0.9231279041, 0.0512161522, 0.3652593359, 0.1086158183),
)
H2_GRASP_HAND_ROTATIONS = H2_READY_HAND_ROTATIONS
H2_PULL_HAND_ROTATIONS = (
    (-0.9524211656, 0.0290508089, -0.2846303110, 0.1050502728),
    (0.9514495724, 0.0604877516, 0.2834135157, 0.1037387205),
)
# Keep the native pinch orientation fixed until the pull begins.
H2_HAND_ROTATIONS = H2_GRASP_HAND_ROTATIONS * 7
# The index joints are identical in the asset's open and closed poses; only the thumbs close.
H2_OPEN = (0.0, 0.0, 0.0, 0.0, 1.0)
H2_CLOSED = (1.0, 1.0, 1.0, 1.0, 1.0)
H2_PHASE_FINGERS = (
    (H2_OPEN, H2_CLOSED),
    *((H2_CLOSED, H2_CLOSED),) * 7,
)


@wp.func
def _smoothstep(value: float) -> float:
    value = wp.clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


@wp.func
def _minimum_jerk(value: float) -> float:
    value = wp.clamp(value, 0.0, 1.0)
    return value * value * value * (10.0 + value * (-15.0 + 6.0 * value))


@wp.func
def _pull_offset(distance: float) -> wp.vec3:
    # This arc preserves the grasp attitude within the arms' reachable pose manifold.
    radius = 0.3072
    center = 0.2248
    height = wp.sqrt(radius * radius - center * center) - wp.sqrt(
        radius * radius - (distance - center) * (distance - center)
    )
    return wp.vec3(-distance, 0.0, height)


@wp.func
def _interpolate_rotation(start: wp.vec4, end: wp.vec4, alpha: float) -> wp.vec4:
    # Quaternion signs describe the same pose; always take the shorter rotation.
    if wp.dot(start, end) < 0.0:
        end = -end
    return wp.normalize(wp.lerp(start, end, alpha))


@wp.kernel
def _infer_state_machine(
    dt: float,
    pull_speed: wp.array(dtype=float),
    keyframes: wp.array(dtype=wp.vec3),
    hand_rotations: wp.array(dtype=wp.vec4),
    pull_hand_rotations: wp.array(dtype=wp.vec4),
    rotate_during_pull: int,
    torso_pose: wp.array2d(dtype=float),
    phase_durations: wp.array(dtype=float),
    phase_start_keyframes: wp.array(dtype=wp.int32),
    phase_end_keyframes: wp.array(dtype=wp.int32),
    phase_start_fingers: wp.array(dtype=float),
    phase_end_fingers: wp.array(dtype=float),
    finger_open_values: wp.array(dtype=float),
    finger_closed_values: wp.array(dtype=float),
    pull_arc_scale: float,
    final_keyframe: int,
    pull_limit: float,
    finger_groups: wp.array(dtype=wp.int32),
    finger_fractions: wp.array2d(dtype=float),
    state: wp.array(dtype=wp.int32),
    state_time: wp.array(dtype=float),
    pull_distance: wp.array(dtype=float),
    commands: wp.array2d(dtype=float),
):
    env_id = wp.tid()
    current_state = state[env_id]
    elapsed = state_time[env_id] + dt

    if current_state < STATE_PULL and elapsed >= phase_durations[current_state]:
        current_state += 1
        elapsed = 0.0

    left = keyframes[0]
    right = keyframes[1]
    left_rotation = hand_rotations[0]
    right_rotation = hand_rotations[1]
    if current_state < STATE_PULL:
        alpha = _smoothstep(elapsed / phase_durations[current_state])
        start_keyframe = phase_start_keyframes[current_state]
        end_keyframe = phase_end_keyframes[current_state]
        left = wp.lerp(keyframes[start_keyframe], keyframes[end_keyframe], alpha)
        right = wp.lerp(keyframes[start_keyframe + 1], keyframes[end_keyframe + 1], alpha)
        left_rotation = _interpolate_rotation(hand_rotations[start_keyframe], hand_rotations[end_keyframe], alpha)
        right_rotation = _interpolate_rotation(
            hand_rotations[start_keyframe + 1], hand_rotations[end_keyframe + 1], alpha
        )
        for group in range(5):
            index = current_state * 5 + group
            finger_fractions[env_id, group] = wp.lerp(phase_start_fingers[index], phase_end_fingers[index], alpha)
    elif current_state == STATE_PULL:
        # Zero endpoint velocity and acceleration prevent a sudden load on the
        # pinch. The quintic's peak derivative is 1.875, preserving --speeds.
        duration = PULL_TIME_FACTOR * pull_limit / pull_speed[env_id]
        distance = pull_limit * _minimum_jerk(elapsed / duration)
        pull_distance[env_id] = distance
        offset = _pull_offset(distance)
        offset[2] *= pull_arc_scale
        left = keyframes[final_keyframe] + offset
        right = keyframes[final_keyframe + 1] + offset
        left_rotation = hand_rotations[final_keyframe]
        right_rotation = hand_rotations[final_keyframe + 1]
        if rotate_during_pull != 0:
            fraction = distance / pull_limit
            left_rotation = _interpolate_rotation(left_rotation, pull_hand_rotations[0], fraction)
            right_rotation = _interpolate_rotation(right_rotation, pull_hand_rotations[1], fraction)
        for group in range(5):
            finger_fractions[env_id, group] = phase_end_fingers[(STATE_PULL - 1) * 5 + group]
        if distance >= pull_limit:
            current_state = STATE_HOLD
    else:
        distance = pull_distance[env_id]
        offset = _pull_offset(distance)
        offset[2] *= pull_arc_scale
        left = keyframes[final_keyframe] + offset
        right = keyframes[final_keyframe + 1] + offset
        left_rotation = hand_rotations[final_keyframe]
        right_rotation = hand_rotations[final_keyframe + 1]
        if rotate_during_pull != 0:
            left_rotation = pull_hand_rotations[0]
            right_rotation = pull_hand_rotations[1]
        for group in range(5):
            finger_fractions[env_id, group] = phase_end_fingers[(STATE_PULL - 1) * 5 + group]

    for axis in range(3):
        commands[env_id, axis] = left[axis]
        commands[env_id, 7 + axis] = right[axis]
    for axis in range(4):
        commands[env_id, 3 + axis] = left_rotation[axis]
        commands[env_id, 10 + axis] = right_rotation[axis]
    for axis in range(7):
        commands[env_id, 14 + axis] = torso_pose[env_id, axis]

    for finger in range(finger_closed_values.shape[0]):
        fraction = finger_fractions[env_id, finger_groups[finger]]
        commands[env_id, POSE_COMPONENTS + finger] = wp.lerp(
            finger_open_values[finger], finger_closed_values[finger], fraction
        )

    state[env_id] = current_state
    state_time[env_id] = elapsed


class _GraspStateMachine:
    """Advance calibrated hand poses and finger targets using only Warp buffers."""

    def __init__(
        self, dt: float, torso_pose: wp.array, speeds: tuple[float, ...], device: str, robot_type: str
    ) -> None:
        self.num_envs = len(speeds)
        self._warp_device = device
        self.dt = dt
        self.pull_speed = wp.array(speeds, dtype=float, device=device)
        if robot_type == "h2":
            durations = H2_PHASE_DURATIONS
            fingers = H2_PHASE_FINGERS
            keyframes = H2_HAND_KEYFRAMES
            open_values = tuple(H2_OPEN_JOINT_POS.values())
            closed_values = tuple(H2_PINCH_JOINT_POS.get(name, value) for name, value in H2_OPEN_JOINT_POS.items())
            groups_by_finger = {"left_thumb": 0, "right_thumb": 1, "left_index": 2, "right_index": 3}
            groups = tuple(groups_by_finger.get("_".join(name.split("_")[:2]), 4) for name in H2_OPEN_JOINT_POS)
            self._pull_arc_scale = 0.0
            self._pull_limit = H2_PULL_DISTANCE
            phase_keyframes = H2_PHASE_KEYFRAMES
            rotations = H2_HAND_ROTATIONS
        else:
            durations = PHASE_DURATIONS
            fingers = PHASE_FINGERS
            keyframes = HAND_KEYFRAMES
            open_values = (0.0,) * len(FINGER_CLOSED_VALUES)
            closed_values = FINGER_CLOSED_VALUES
            groups = FINGER_GROUPS
            self._pull_arc_scale = 1.0
            self._pull_limit = PULL_DISTANCE
            phase_keyframes = PHASE_KEYFRAMES
            rotations = (*(HAND_ROTATIONS * 2), *(GRASP_HAND_ROTATIONS * 5))
        if len(rotations) != len(keyframes):
            raise ValueError("Each hand position keyframe requires a matching rotation.")
        pull_rotations = H2_PULL_HAND_ROTATIONS if robot_type == "h2" else rotations[-2:]
        self._rotate_during_pull = int(tuple(pull_rotations) != tuple(rotations[-2:]))
        self._final_keyframe = len(keyframes) - 2
        phases = list(zip(durations, phase_keyframes, fingers, strict=True))
        self._keyframes = wp.array(keyframes, dtype=wp.vec3, device=self._warp_device)
        self._hand_rotations = wp.array(rotations, dtype=wp.vec4, device=self._warp_device)
        self._pull_hand_rotations = wp.array(pull_rotations, dtype=wp.vec4, device=self._warp_device)
        self._torso_pose = torso_pose
        self._phase_durations = wp.array([phase[0] for phase in phases], dtype=float, device=self._warp_device)
        self._phase_start_keyframes = wp.array(
            [phase[1][0] for phase in phases], dtype=wp.int32, device=self._warp_device
        )
        self._phase_end_keyframes = wp.array(
            [phase[1][1] for phase in phases], dtype=wp.int32, device=self._warp_device
        )
        self._phase_start_fingers = wp.array(
            [value for phase in phases for value in phase[2][0]], dtype=float, device=self._warp_device
        )
        self._phase_end_fingers = wp.array(
            [value for phase in phases for value in phase[2][1]], dtype=float, device=self._warp_device
        )
        self._finger_open_values = wp.array(open_values, dtype=float, device=self._warp_device)
        self._finger_closed_values = wp.array(closed_values, dtype=float, device=self._warp_device)
        self._finger_groups = wp.array(groups, dtype=wp.int32, device=self._warp_device)
        self._finger_fractions = wp.zeros((self.num_envs, 5), dtype=float, device=self._warp_device)
        self._state = wp.zeros(self.num_envs, dtype=wp.int32, device=self._warp_device)
        self._state_time = wp.zeros(self.num_envs, dtype=float, device=self._warp_device)
        self._pull_distance = wp.zeros(self.num_envs, dtype=float, device=self._warp_device)
        self._commands = wp.zeros((self.num_envs, POSE_COMPONENTS + len(closed_values)), dtype=float, device=device)

    def compute(self) -> None:
        """Advance one frame without allocating buffers or reading device state on the host."""
        wp.launch(
            _infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.dt,
                self.pull_speed,
                self._keyframes,
                self._hand_rotations,
                self._pull_hand_rotations,
                self._rotate_during_pull,
                self._torso_pose,
                self._phase_durations,
                self._phase_start_keyframes,
                self._phase_end_keyframes,
                self._phase_start_fingers,
                self._phase_end_fingers,
                self._finger_open_values,
                self._finger_closed_values,
                self._pull_arc_scale,
                self._final_keyframe,
                self._pull_limit,
                self._finger_groups,
                self._finger_fractions,
                self._state,
                self._state_time,
                self._pull_distance,
                self._commands,
            ],
            device=self._warp_device,
        )


@wp.kernel
def _seed_coordinates(current: wp.array2d(dtype=float), indices: wp.array(dtype=int), seed: wp.array2d(dtype=float)):
    lane, joint = wp.tid()
    if indices[joint] >= 0:
        seed[lane, indices[joint]] = current[lane, joint]


@wp.kernel
def _clamp_ik_coordinates(
    coordinates: wp.array2d(dtype=float), lower: wp.array(dtype=float), upper: wp.array(dtype=float)
):
    lane, joint = wp.tid()
    coordinates[lane, joint] = wp.clamp(coordinates[lane, joint], lower[joint], upper[joint])


@wp.kernel
def _set_pose_targets(
    poses: wp.array2d(dtype=float),
    offset: int,
    root: wp.transform,
    positions: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.vec4),
):
    lane = wp.tid()
    position = wp.vec3(poses[lane, offset], poses[lane, offset + 1], poses[lane, offset + 2])
    rotation = wp.quat(
        poses[lane, offset + 3], poses[lane, offset + 4], poses[lane, offset + 5], poses[lane, offset + 6]
    )
    positions[lane] = wp.transform_point(root, position)
    result = wp.transform_get_rotation(root) * rotation
    rotations[lane] = wp.vec4(result[0], result[1], result[2], result[3])


@wp.kernel
def _set_elbow_position(
    body_poses: wp.array(dtype=wp.transform),
    roots: wp.array(dtype=int),
    elbows: wp.array(dtype=int),
    native_root: wp.transform,
    guard_y: float,
    states: wp.array(dtype=int),
    times: wp.array(dtype=float),
    durations: wp.array(dtype=float),
    targets: wp.array(dtype=wp.vec3),
):
    lane = wp.tid()
    root_inverse = wp.transform_inverse(body_poses[roots[lane]])
    position = wp.transform_point(root_inverse, wp.transform_get_translation(body_poses[elbows[lane]]))
    blend = float(1.0)
    if states[lane] == 0:
        blend = _smoothstep(times[lane] / durations[0])
    outward_y = wp.min(position[1], guard_y)
    if guard_y > 0.0:
        outward_y = wp.max(position[1], guard_y)
    position[1] = wp.lerp(position[1], outward_y, blend)
    targets[lane] = wp.transform_point(native_root, position)


@wp.kernel
def _gather_targets(
    solved: wp.array2d(dtype=float),
    poses: wp.array2d(dtype=float),
    indices: wp.array(dtype=int),
    arms: wp.array(dtype=int),
    fingers: wp.array(dtype=int),
    defaults: wp.array2d(dtype=float),
    velocity_limits: wp.array2d(dtype=float),
    previous: wp.array2d(dtype=float),
    dt: float,
    velocity_feedforward: int,
    targets: wp.array2d(dtype=float),
    velocities: wp.array2d(dtype=float),
):
    lane, joint = wp.tid()
    value = defaults[lane, joint]
    if arms[joint] != 0:
        value = solved[lane, indices[joint]]
    if fingers[joint] >= 0:
        value = poses[lane, POSE_COMPONENTS + fingers[joint]]
    velocity = float(0.0)
    if arms[joint] != 0 and velocity_feedforward != 0:
        # Match moving arm targets without exciting the quasi-massless fingers.
        limit = wp.min(velocity_limits[lane, joint], 20.0)
        velocity = wp.clamp((value - previous[lane, joint]) / dt, -limit, limit)
    previous[lane, joint] = value
    targets[lane, joint] = value
    velocities[lane, joint] = velocity


@wp.kernel(enable_backward=False)
def _read_arm_positions(
    positions: wp.array(dtype=float),
    coordinates: wp.array2d(dtype=int),
    selected: wp.array2d(dtype=float),
):
    lane, arm = wp.tid()
    selected[lane, arm] = positions[coordinates[lane, arm]]


@wp.kernel(enable_backward=False)
def _stage_arm_targets(
    targets: wp.array(dtype=float),
    coordinates: wp.array2d(dtype=int),
    start: wp.array2d(dtype=float),
    end: wp.array2d(dtype=float),
    counters: wp.array(dtype=int),
):
    lane = wp.tid()
    for arm in range(coordinates.shape[1]):
        start[lane, arm] = end[lane, arm]
        end[lane, arm] = targets[coordinates[lane, arm]]
    counters[lane] = 0


@wp.kernel(enable_backward=False)
def _interpolate_arm_targets(
    coordinates: wp.array2d(dtype=int),
    start: wp.array2d(dtype=float),
    end: wp.array2d(dtype=float),
    counters: wp.array(dtype=int),
    substeps: int,
    targets: wp.array(dtype=float),
):
    lane = wp.tid()
    tick = wp.min(counters[lane] + 1, substeps)
    for arm in range(coordinates.shape[1]):
        value = end[lane, arm]
        if tick < substeps:
            fraction = float(tick) / float(substeps)
            value = start[lane, arm] + fraction * (end[lane, arm] - start[lane, arm])
        targets[coordinates[lane, arm]] = value
    counters[lane] = tick


class TableclothController:
    """Batched GPU IK and implicit joint drives for physically grasping the cloth.

    Register :meth:`compute` as a Newton post-step callback before the first
    simulation step. The physics graph then includes the state machine, IK, and
    actuator submission. There are no Torch operations or host state checks in
    the callback. Cloth motion is entirely contact-driven: no particle targets
    or artificial grasp attachments are used.
    """

    def __init__(
        self, robot: Articulation, usd_path: str, dt: float, speeds: tuple[float, ...], robot_type: str = "h1"
    ) -> None:
        from pxr import Gf, Usd, UsdPhysics  # noqa: PLC0415

        self.robot = robot
        self.device = robot.device
        self.num_lanes = len(speeds)
        self.num_joints = robot.num_joints
        self._use_command_seed = robot_type == "h2"
        if robot.num_instances != self.num_lanes:
            raise ValueError("The example requires one identical fixed-base robot per environment.")
        if robot_type not in ("h1", "h2"):
            raise ValueError(f"Unsupported tablecloth robot: {robot_type}")
        if robot_type == "h2":
            self.hand_body_names = ("left_hand_C_MC", "right_hand_C_MC")
            self.hand_offsets = H2_HAND_OFFSETS
            arm_expr = [r"(left|right)_(shoulder|elbow|wrist)_.*"]
            finger_names = list(H2_OPEN_JOINT_POS)
        else:
            self.hand_body_names = ("left_hand_link", "right_hand_link")
            self.hand_offsets = HAND_OFFSETS
            arm_expr = ["torso_joint", "(left|right)_(shoulder_.*|elbow|hand)_joint"]
            finger_names = FINGER_JOINT_NAMES
        arm_ids, arm_names = robot.find_joints(arm_expr)
        self._arm_joint_names = arm_names
        self._arm_control: newton.Control | None = None

        # Lock non-IK coordinates in an anonymous session layer only. The
        # simulation retains every articulated finger; IK solves just the arms.
        # This also keeps LM's tiled solve within the GPU shared-memory budget.
        stage = Usd.Stage.Open(usd_path)
        stage.SetEditTarget(stage.GetSessionLayer())
        fixed_positions = (
            dict(zip(robot.joint_names, robot.data.default_joint_pos.warp.numpy()[0], strict=True))
            if robot_type == "h2"
            else {}
        )
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() not in arm_names:
                # A fixed joint must retain the configured posture, especially
                # a forward waist bend. Freezing the authored zero angle would
                # give IK a different arm chain from the simulated robot.
                angle = float(fixed_positions.get(prim.GetName(), 0.0))
                if angle != 0.0:
                    joint = UsdPhysics.RevoluteJoint(prim)
                    axis = Gf.Vec3f(
                        *{"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}[joint.GetAxisAttr().Get()]
                    )
                    rotation = Gf.Quatf(math.cos(0.5 * angle), math.sin(0.5 * angle) * axis)
                    joint.GetLocalRot0Attr().Set(joint.GetLocalRot0Attr().Get() * rotation)
                prim.SetTypeName("PhysicsFixedJoint")
        builder = newton.ModelBuilder()
        builder.add_usd(
            stage,
            floating=False if robot_type == "h2" else None,
            enable_self_collisions=False,
            load_visual_shapes=False,
            load_static_visual_shapes=False,
        )
        model = builder.finalize(device=self.device)
        if robot_type == "h2":
            # Reserve room for compliant tracking without commanding physical hard stops.
            model.joint_limit_lower.assign(model.joint_limit_lower.numpy() + 0.08)
            model.joint_limit_upper.assign(model.joint_limit_upper.numpy() - 0.08)
            if model.joint_coord_count != model.joint_dof_count:
                raise ValueError("H2 arm IK requires scalar joint coordinates.")
        self._ik_lower = model.joint_limit_lower
        self._ik_upper = model.joint_limit_upper
        starts = model.joint_q_start.numpy()
        coordinates = {
            label.split("/")[-1]: int(start)
            for label, start, end in zip(model.joint_label, starts[:-1], starts[1:], strict=True)
            if end > start
        }
        links = {label.split("/")[-1]: index for index, label in enumerate(model.body_label)}
        self._indices = wp.array(
            [coordinates.get(name, -1) for name in robot.joint_names], dtype=int, device=self.device
        )
        self._seed = wp.zeros((self.num_lanes, model.joint_coord_count), dtype=float, device=self.device)
        self._defaults = wp.clone(robot.data.default_joint_pos.warp)
        self._previous_targets = wp.clone(self._defaults)
        self._velocity_limits = wp.clone(robot.data.joint_vel_limits.warp)
        self._targets = wp.zeros((self.num_lanes, self.num_joints), dtype=float, device=self.device)
        self._velocity_targets = wp.zeros_like(self._targets)
        self._joint_positions = robot.data.joint_pos.warp
        if self._use_command_seed:
            wp.launch(
                _seed_coordinates,
                dim=(self.num_lanes, self.num_joints),
                inputs=[self._defaults, self._indices, self._seed],
                device=self.device,
            )
        finger_ids, _ = robot.find_joints(finger_names, preserve_order=True)
        finger_indices = {joint: index for index, joint in enumerate(finger_ids)}
        self._arms = wp.array([int(i in arm_ids) for i in range(self.num_joints)], dtype=int, device=self.device)
        self._fingers = wp.array(
            [finger_indices.get(i, -1) for i in range(self.num_joints)], dtype=int, device=self.device
        )
        if robot_type == "h2":
            # USD body transforms still describe the authored zero posture.
            # Evaluate the configured READY seed before constructing posture targets.
            initial_state = model.state()
            wp.copy(model.joint_q, self._seed, count=model.joint_coord_count)
            newton.eval_fk(model, model.joint_q, model.joint_qd, initial_state)
            body_poses = initial_state.body_q.numpy()
        else:
            body_poses = model.body_q.numpy()
        self._root = wp.transform(body_poses[links["pelvis"]][:3], body_poses[links["pelvis"]][3:])
        torso = wp.transform(body_poses[links["torso_link"]][:3], body_poses[links["torso_link"]][3:])
        torso = wp.transform_inverse(self._root) * torso
        torso_pose = wp.array([list(torso)] * self.num_lanes, dtype=float, device=self.device)
        objectives = [
            NewtonIKPoseObjectiveCfg(
                body_name=body,
                name=name + "_hand",
                body_offset_pos=offset,
                use_relative_mode=False,
                position_weight=500.0 if robot_type == "h2" else 5.0,
                rotation_weight=50.0 if robot_type == "h2" else 0.2,
            )
            for name, body, offset in zip(("left", "right"), self.hand_body_names, self.hand_offsets, strict=True)
        ]
        objectives.extend(
            [
                NewtonIKPoseObjectiveCfg(
                    body_name="torso_link",
                    name="torso",
                    use_relative_mode=False,
                    position_weight=50.0,
                    rotation_weight=50.0,
                ),
                NewtonIKJointLimitObjectiveCfg(weight=100.0 if robot_type == "h2" else 1.0),
            ]
        )
        if robot_type == "h2":
            objectives.extend(
                NewtonIKPoseObjectiveCfg(
                    body_name=f"{side}_elbow_link",
                    name=f"{side}_elbow",
                    command_type="position",
                    use_relative_mode=False,
                    position_weight=5.0,
                    rotation_weight=0.0,
                )
                for side in ("left", "right")
            )
        self._solver = NewtonIKSolver(
            NewtonIKSolverCfg(iterations=32 if robot_type == "h2" else 24, lambda_initial=0.1),
            model=model,
            num_envs=self.num_lanes,
            device=self.device,
            objectives=objectives,
            link_resolver=lambda name: links[name],
        )
        self._poses = [self._solver.objectives_by_name[name] for name in ("left_hand", "right_hand", "torso")]
        if robot_type == "h2":
            self._elbow_positions = []
            for side in ("left", "right"):
                positions = self._solver.objectives_by_name[f"{side}_elbow"].position_objective.target_positions
                positions.assign(np.repeat(body_poses[links[f"{side}_elbow_link"], :3][None], self.num_lanes, axis=0))
                self._elbow_positions.append(positions)
        self._machine = _GraspStateMachine(dt, torso_pose, speeds, self.device, robot_type)
        self._scene_state: newton.State | None = None

    def bind_arm_substeps(
        self,
        model: newton.Model,
        state: newton.State,
        control: newton.Control,
        env_paths: Sequence[str],
        substeps: int = 16,
    ) -> None:
        """Bind H2 arm-position interpolation before physics graph capture.

        Call after the first :meth:`compute`, then register
        :meth:`interpolate_arm_targets` after the native actuator adapter on
        every inner physics tick. The caller keeps IK at the outer-frame
        cadence and advances exactly ``substeps`` inner ticks per frame.
        Only the fourteen arm position targets are rewritten; fingers and
        velocity targets retain their controller values.

        Args:
            model: Native scene model containing one H2 per environment.
            state: Initial native joint state, used as the first ramp's start.
            control: Native control whose position targets are updated in place.
            env_paths: Environment prim paths, ordered like the controller lanes.
            substeps: Number of inner physics ticks per controller update.
        """
        if not self._use_command_seed or len(env_paths) != self.num_lanes:
            raise ValueError("Arm interpolation requires one H2 per controller lane.")
        if self._arm_control is not None:
            raise RuntimeError("Arm interpolation is already bound.")
        if not isinstance(substeps, int) or substeps < 1:
            raise ValueError("Arm interpolation requires a positive integer substep count.")
        if len(self._arm_joint_names) != 14 or len(set(self._arm_joint_names)) != 14:
            raise ValueError("H2 interpolation requires fourteen distinct arm joints.")
        if state.joint_q is None or control.joint_target_q is None or model.joint_target_q_start is None:
            raise ValueError("Arm interpolation requires native joint positions and position targets.")
        if state.joint_q.device != control.joint_target_q.device or state.joint_q.device != wp.get_device(self.device):
            raise ValueError("Arm state, position targets, and controller must share a device.")

        q_starts = model.joint_q_start.numpy()
        target_starts = model.joint_target_q_start.numpy()
        q_coordinates = np.empty((self.num_lanes, 14), dtype=np.int32)
        target_coordinates = np.empty_like(q_coordinates)
        for lane, env_path in enumerate(env_paths):
            prefix = env_path.rstrip("/") + "/Robot/"
            selected = {}
            for joint, label in enumerate(model.joint_label):
                name = label.rsplit("/", 1)[-1]
                if label.startswith(prefix) and name in self._arm_joint_names:
                    if name in selected:
                        raise ValueError(f"Duplicate H2 arm joint {name!r} in {env_path!r}.")
                    if (
                        q_starts[joint + 1] - q_starts[joint] != 1
                        or target_starts[joint + 1] - target_starts[joint] != 1
                    ):
                        raise ValueError(f"H2 arm joint {name!r} must have scalar state and target coordinates.")
                    selected[name] = (int(q_starts[joint]), int(target_starts[joint]))
            if set(selected) != set(self._arm_joint_names):
                raise ValueError(f"Incomplete H2 arm selection in {env_path!r}: {sorted(selected)}")
            for arm, name in enumerate(self._arm_joint_names):
                q_coordinates[lane, arm], target_coordinates[lane, arm] = selected[name]
        for coordinates, size in (
            (q_coordinates, state.joint_q.size),
            (target_coordinates, control.joint_target_q.size),
        ):
            if coordinates.min() < 0 or coordinates.max() >= size or np.unique(coordinates).size != coordinates.size:
                raise ValueError("Each H2 arm coordinate must be valid and distinct across lanes.")

        self._arm_target_coordinates = wp.array(target_coordinates, dtype=int, device=self.device)
        self._arm_initial = wp.empty(q_coordinates.shape, dtype=float, device=self.device)
        wp.launch(
            _read_arm_positions,
            dim=q_coordinates.shape,
            inputs=[state.joint_q, wp.array(q_coordinates, dtype=int, device=self.device), self._arm_initial],
            device=self.device,
        )
        self._arm_start = wp.clone(self._arm_initial)
        self._arm_end = wp.clone(self._arm_initial)
        self._arm_counters = wp.zeros(self.num_lanes, dtype=int, device=self.device)
        self._arm_substeps = substeps
        self._arm_control = control
        self._stage_arm_targets()

    def interpolate_arm_targets(self) -> None:
        """Apply the next linear arm-position sample after the native actuator adapter."""
        if self._arm_control is None:
            return
        wp.launch(
            _interpolate_arm_targets,
            dim=self.num_lanes,
            inputs=[
                self._arm_target_coordinates,
                self._arm_start,
                self._arm_end,
                self._arm_counters,
                self._arm_substeps,
                self._arm_control.joint_target_q,
            ],
            device=self.device,
        )

    def reset_arm_targets(self) -> None:
        """Restart the ramp from the initial state after the reset's existing :meth:`compute`."""
        if self._arm_control is None:
            return
        wp.copy(self._arm_end, self._arm_initial)
        self._stage_arm_targets()

    def observe_scene(self, model: newton.Model, state: newton.State, env_prim_paths: list[str]) -> None:
        """Bind native pelvis and elbow observations for H2.

        Resolve body labels once before the first update. Elbow posture guards
        run inside the physics CUDA graph without changing the cloth.
        """
        if not self._use_command_seed or len(env_prim_paths) != self.num_lanes:
            raise ValueError("Scene observation requires one H2 per environment.")
        roots, elbows = [], ([], [])
        for env_path in env_prim_paths:
            resolved = []
            for body in ("pelvis", "left_elbow_link", "right_elbow_link"):
                matches = [
                    i
                    for i, name in enumerate(model.body_label)
                    if name.startswith(env_path + "/Robot/") and name.rsplit("/", 1)[-1] == body
                ]
                if len(matches) != 1:
                    raise ValueError(f"Expected one H2 {body} in {env_path}, found {len(matches)}")
                resolved.append(matches[0])
            roots.append(resolved[0])
            elbows[0].append(resolved[1])
            elbows[1].append(resolved[2])
        self._scene_state = state
        self._scene_roots = wp.array(roots, dtype=int, device=self.device)
        self._scene_elbows = [wp.array(side, dtype=int, device=self.device) for side in elbows]

    @staticmethod
    def duration(speeds: tuple[float, ...], dt: float, robot_type: str = "h1") -> float:
        """Include per-phase frame rounding, the slowest pull, and a one-second hold."""
        durations = H2_PHASE_DURATIONS if robot_type == "h2" else PHASE_DURATIONS
        distance = H2_PULL_DISTANCE if robot_type == "h2" else PULL_DISTANCE
        return sum(durations) + len(durations) * dt + PULL_TIME_FACTOR * distance / min(speeds) + 1.0

    def compute(self) -> None:
        """Advance the GPU controller and submit the next frame's joint targets."""
        self._machine.compute()
        for i, pose in enumerate(self._poses):
            wp.launch(
                _set_pose_targets,
                dim=self.num_lanes,
                inputs=[
                    self._machine._commands,
                    7 * i,
                    self._root,
                    pose.position_objective.target_positions,
                    pose.rotation_objective.target_rotations,
                ],
                device=self.device,
            )
        if self._scene_state is not None:
            # Measured native torso/elbow clearance guards. Keep current X/Z
            # free to follow the hand trajectory rather than fixing an elbow pose.
            for elbows, positions, guard in zip(
                self._scene_elbows, self._elbow_positions, (0.159, -0.180), strict=True
            ):
                wp.launch(
                    _set_elbow_position,
                    dim=self.num_lanes,
                    inputs=[
                        self._scene_state.body_q,
                        self._scene_roots,
                        elbows,
                        self._root,
                        guard,
                        self._machine._state,
                        self._machine._state_time,
                        self._machine._phase_durations,
                        positions,
                    ],
                    device=self.device,
                )
        if not self._use_command_seed:
            wp.launch(
                _seed_coordinates,
                dim=(self.num_lanes, self.num_joints),
                inputs=[self._joint_positions, self._indices, self._seed],
                device=self.device,
            )
        solved = self._solver.solve(self._seed)
        if self._use_command_seed:
            # The IK limit objective is soft. Never ask physical drives to
            # follow an out-of-range solution when a contact shifts the hem.
            wp.launch(
                _clamp_ik_coordinates,
                dim=solved.shape,
                inputs=[solved, self._ik_lower, self._ik_upper],
                device=self.device,
            )
            # Contact tracking errors must not steer the redundant arm coordinates.
            wp.copy(self._seed, solved)
        wp.launch(
            _gather_targets,
            dim=(self.num_lanes, self.num_joints),
            inputs=[
                solved,
                self._machine._commands,
                self._indices,
                self._arms,
                self._fingers,
                self._defaults,
                self._velocity_limits,
                self._previous_targets,
                self._machine.dt,
                int(self._use_command_seed),
                self._targets,
                self._velocity_targets,
            ],
            device=self.device,
        )
        self.robot.actuators.target_command.set_position_index(value=self._targets)
        self.robot.actuators.target_command.set_velocity_index(value=self._velocity_targets)
        self.robot.write_data_to_sim()
        if self._arm_control is not None:
            self._stage_arm_targets()

    def _stage_arm_targets(self) -> None:
        wp.launch(
            _stage_arm_targets,
            dim=self.num_lanes,
            inputs=[
                self._arm_control.joint_target_q,
                self._arm_target_coordinates,
                self._arm_start,
                self._arm_end,
                self._arm_counters,
            ],
            device=self.device,
        )


@wp.kernel
def _measure_grasp(
    mesh: wp.uint64,
    body_poses: wp.array(dtype=wp.transform),
    hands: wp.array2d(dtype=int),
    offsets: wp.array(dtype=wp.vec3),
    states: wp.array(dtype=int),
    faces: wp.array2d(dtype=int),
    coordinates: wp.array2d(dtype=wp.vec2),
    initial_points: wp.array2d(dtype=wp.vec3),
    initial_gaps: wp.array2d(dtype=float),
    maximum_gaps: wp.array2d(dtype=float),
    travel: wp.array2d(dtype=float),
):
    lane, side = wp.tid()
    if states[lane] >= STATE_PULL:
        hand = wp.transform_point(body_poses[hands[lane, side]], offsets[side])
        if faces[lane, side] == -1:
            # Try once at pull onset; a later accidental pickup is not a successful grasp.
            faces[lane, side] = -2
            query = wp.mesh_query_point_no_sign(mesh, hand, 0.2)
            if query.result:
                faces[lane, side] = query.face
                coordinates[lane, side] = wp.vec2(query.u, query.v)
                point = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
                initial_points[lane, side] = point
                initial_gaps[lane, side] = wp.length(point - hand)
        if faces[lane, side] >= 0:
            uv = coordinates[lane, side]
            point = wp.mesh_eval_position(mesh, faces[lane, side], uv[0], uv[1])
            maximum_gaps[lane, side] = wp.max(maximum_gaps[lane, side], wp.length(point - hand))
            travel[lane, side] = initial_points[lane, side][0] - point[0]


class GraspMonitor:
    """Track fixed cloth material points on the GPU, rather than just commanded motion.

    Sample after each physics frame inside the graph. Only :meth:`report`,
    called after the complete run, copies results to the host. This monitor
    observes material motion; it never changes the cloth or attaches it to a hand.
    """

    def __init__(
        self, model: newton.Model, state: newton.State, controller: TableclothController, env_prim_paths: list[str]
    ) -> None:
        device = controller.device
        count = controller.num_lanes
        if len(env_prim_paths) != count:
            raise ValueError("Expected one environment path per robot instance.")
        self._state = state
        self._states = controller._machine._state
        hand_ids = []
        for env_path in env_prim_paths:
            robot_path = env_path + "/Robot"
            pair = []
            for side, body in zip(("left", "right"), controller.hand_body_names, strict=True):
                matches = [
                    i
                    for i, name in enumerate(model.body_label)
                    if name.startswith(robot_path + "/") and name.rsplit("/", 1)[-1] == body
                ]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {side} hand for {robot_path}, found {len(matches)}")
                pair.append(matches[0])
            hand_ids.append(pair)
        self._hands = wp.array(hand_ids, dtype=int, device=device)
        self._offsets = wp.array(controller.hand_offsets, dtype=wp.vec3, device=device)
        self._mesh = wp.Mesh(points=wp.clone(state.particle_q), indices=model.tri_indices.flatten())
        self._faces = wp.full((count, 2), -1, dtype=int, device=device)
        self._coordinates = wp.zeros((count, 2), dtype=wp.vec2, device=device)
        self._initial_points = wp.zeros((count, 2), dtype=wp.vec3, device=device)
        self._initial_gaps = wp.full((count, 2), 1.0, dtype=float, device=device)
        self._maximum_gaps = wp.zeros((count, 2), dtype=float, device=device)
        self._travel = wp.zeros((count, 2), dtype=float, device=device)

    def compute(self) -> None:
        """Refit the cloth surface and measure acquisition, retention, and withdrawal."""
        wp.copy(self._mesh.points, self._state.particle_q)
        self._mesh.refit()
        wp.launch(
            _measure_grasp,
            dim=self._faces.shape,
            inputs=[
                self._mesh.id,
                self._state.body_q,
                self._hands,
                self._offsets,
                self._states,
                self._faces,
                self._coordinates,
                self._initial_points,
                self._initial_gaps,
                self._maximum_gaps,
                self._travel,
            ],
            device=self._faces.device,
        )

    def report(self) -> list[dict[str, object]]:
        """Return end-of-run physical grip results, in lane order."""
        initial = self._initial_gaps.numpy()
        maximum = self._maximum_gaps.numpy()
        travel = self._travel.numpy()
        return [
            {
                "retained": bool((gap < 0.005).all() and (peak < 0.015).all() and (distance > 0.30).all()),
                "initial_gap_m": gap.tolist(),
                "maximum_gap_m": peak.tolist(),
                "travel_m": distance.tolist(),
            }
            for gap, peak, distance in zip(initial, maximum, travel, strict=True)
        ]
