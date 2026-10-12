# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""GPU state machine and Newton IK for the fixed-base tablecloth example."""

from __future__ import annotations

import math
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
    H2_HAND_OFFSETS,
    H2_OPEN_JOINT_POS,
    H2_PINCH_JOINT_POS,
)

H2_PULL_DISTANCE = 0.45
PULL_TIME_FACTOR = 1.875
POSE_COMPONENTS = 14

H2_PHASE_DURATIONS = (0.20, 0.40, 0.40, 0.40, 0.40, 0.30, 0.40, 0.40)
H2_PHASE_KEYFRAMES = ((0, 0), (0, 2), (2, 4), (4, 6), (6, 8), (8, 8), (8, 10), (10, 12))
STATE_PULL = wp.constant(len(H2_PHASE_DURATIONS))
STATE_HOLD = wp.constant(len(H2_PHASE_DURATIONS) + 1)

# Positions are expressed in the fixed pelvis frame.
# Ready, clearance, precontact, insertion, pinch, withdrawal, and lift poses [m].
H2_HAND_KEYFRAMES = (
    (0.45, 0.2475379991, 0.16),
    (0.45, -0.2475441579, 0.16),
    (0.55, 0.2475379991, 0.15),
    (0.55, -0.2475441579, 0.15),
    (0.55, 0.2475379991, 0.065),
    (0.55, -0.2475441579, 0.065),
    (0.577, 0.2475379991, 0.070),
    (0.577, -0.2475441579, 0.070),
    (0.578403, 0.247525, 0.078011),
    (0.577649, -0.247555, 0.078554),
    (0.558403, 0.247525, 0.087011),
    (0.557649, -0.247555, 0.087554),
    (0.5628056417, 0.2475379991, 0.1270655547),
    (0.5628101729, -0.2475441579, 0.1270715950),
)
# Control the complete wrist pose: a free roll can put the proximal fingers inside the table.
H2_READY_HAND_ROTATIONS = (
    (-0.9239897494, 0.0197845272, -0.3665561809, 0.1071824696),
    (0.9231279041, 0.0512161522, 0.3652593359, 0.1086158183),
)
H2_GRASP_HAND_ROTATIONS = (
    (-0.807865338839, -0.006732944032, -0.579201247457, 0.108785002508),
    (0.812353708056, 0.024445186614, 0.570667120832, 0.117570927880),
)
H2_PULL_HAND_ROTATIONS = (
    (-0.9524211656, 0.0290508089, -0.2846303110, 0.1050502728),
    (0.9514495724, 0.0604877516, 0.2834135157, 0.1037387205),
)
# Align with the settled hem, then return to the flat pull attitude after withdrawal.
H2_HAND_ROTATIONS = (*H2_READY_HAND_ROTATIONS, *(H2_GRASP_HAND_ROTATIONS * 5), *H2_READY_HAND_ROTATIONS)
# The index joints are identical in the asset's open and closed poses; only the thumbs close.
STATE_CLOSE = wp.constant(5)


@wp.func
def _smoothstep(value: float) -> float:
    value = wp.clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


@wp.func
def _minimum_jerk(value: float) -> float:
    value = wp.clamp(value, 0.0, 1.0)
    return value * value * value * (10.0 + value * (-15.0 + 6.0 * value))


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
    phase_durations: wp.array(dtype=float),
    phase_keyframes: wp.array(dtype=wp.vec2i),
    finger_open_values: wp.array(dtype=float),
    finger_closed_values: wp.array(dtype=float),
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

    closure = float(1.0)
    alpha = float(0.0)
    distance = H2_PULL_DISTANCE
    start_keyframe = keyframes.shape[0] - 2
    end_keyframe = start_keyframe
    if current_state < STATE_PULL:
        alpha = _smoothstep(elapsed / phase_durations[current_state])
        start_keyframe = phase_keyframes[current_state][0]
        end_keyframe = phase_keyframes[current_state][1]
        if current_state < STATE_CLOSE:
            closure = 0.0
        elif current_state == STATE_CLOSE:
            closure = alpha
    elif current_state == STATE_PULL:
        # Zero endpoint velocity and acceleration prevent a sudden load on the
        # pinch. The quintic's peak derivative is 1.875, preserving --speeds.
        duration = PULL_TIME_FACTOR * H2_PULL_DISTANCE / pull_speed[env_id]
        distance = H2_PULL_DISTANCE * _minimum_jerk(elapsed / duration)
        pull_distance[env_id] = distance

    for side in range(2):
        position = keyframes[start_keyframe + side]
        rotation = hand_rotations[start_keyframe + side]
        if current_state < STATE_PULL:
            position = wp.lerp(position, keyframes[end_keyframe + side], alpha)
            rotation = _interpolate_rotation(rotation, hand_rotations[end_keyframe + side], alpha)
        else:
            position += wp.vec3(-distance, 0.0, 0.0)
            if current_state == STATE_PULL:
                rotation = _interpolate_rotation(rotation, pull_hand_rotations[side], distance / H2_PULL_DISTANCE)
            else:
                rotation = pull_hand_rotations[side]
        for axis in range(3):
            commands[env_id, 7 * side + axis] = position[axis]
        for axis in range(4):
            commands[env_id, 7 * side + 3 + axis] = rotation[axis]
    for finger in range(finger_closed_values.shape[0]):
        commands[env_id, POSE_COMPONENTS + finger] = wp.lerp(
            finger_open_values[finger], finger_closed_values[finger], closure
        )

    if current_state == STATE_PULL and distance >= H2_PULL_DISTANCE:
        current_state = STATE_HOLD
    state[env_id] = current_state
    state_time[env_id] = elapsed


class _GraspStateMachine:
    """Advance calibrated hand poses and finger targets using only Warp buffers."""

    def __init__(self, dt: float, speeds: tuple[float, ...], device: str) -> None:
        self.num_envs = len(speeds)
        self.device = device
        self.dt = dt
        self.pull_speed = wp.array(speeds, dtype=float, device=device)
        open_values = tuple(H2_OPEN_JOINT_POS.values())
        closed_values = tuple(H2_PINCH_JOINT_POS[name] for name in H2_OPEN_JOINT_POS)
        self._keyframes = wp.array(H2_HAND_KEYFRAMES, dtype=wp.vec3, device=device)
        self._hand_rotations = wp.array(H2_HAND_ROTATIONS, dtype=wp.vec4, device=device)
        self._pull_hand_rotations = wp.array(H2_PULL_HAND_ROTATIONS, dtype=wp.vec4, device=device)
        self._phase_durations = wp.array(H2_PHASE_DURATIONS, dtype=float, device=device)
        self._phase_keyframes = wp.array(H2_PHASE_KEYFRAMES, dtype=wp.vec2i, device=device)
        self._finger_open_values = wp.array(open_values, dtype=float, device=device)
        self._finger_closed_values = wp.array(closed_values, dtype=float, device=device)
        self._state = wp.zeros(self.num_envs, dtype=int, device=device)
        self._state_time = wp.zeros(self.num_envs, dtype=float, device=device)
        self._pull_distance = wp.zeros(self.num_envs, dtype=float, device=device)
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
                self._phase_durations,
                self._phase_keyframes,
                self._finger_open_values,
                self._finger_closed_values,
                self._state,
                self._state_time,
                self._pull_distance,
                self._commands,
            ],
            device=self.device,
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
    fingers: wp.array(dtype=int),
    velocity_limits: wp.array2d(dtype=float),
    dt: float,
    targets: wp.array2d(dtype=float),
    velocities: wp.array2d(dtype=float),
):
    lane, joint = wp.tid()
    value = targets[lane, joint]
    if indices[joint] >= 0:
        value = solved[lane, indices[joint]]
    if fingers[joint] >= 0:
        value = poses[lane, POSE_COMPONENTS + fingers[joint]]
    velocity = float(0.0)
    if indices[joint] >= 0:
        # Match moving arm targets without exciting the quasi-massless fingers.
        limit = wp.min(velocity_limits[lane, joint], 20.0)
        velocity = wp.clamp((value - targets[lane, joint]) / dt, -limit, limit)
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

    def __init__(self, robot: Articulation, usd_path: str, dt: float, speeds: tuple[float, ...]) -> None:
        from pxr import Gf, Usd, UsdPhysics  # noqa: PLC0415

        self.robot = robot
        self.device = robot.device
        self.num_lanes = len(speeds)
        self.num_joints = robot.num_joints
        if robot.num_instances != self.num_lanes:
            raise ValueError("The example requires one identical fixed-base robot per environment.")
        self.hand_body_names = ("left_hand_C_MC", "right_hand_C_MC")
        self.hand_offsets = H2_HAND_OFFSETS
        arm_expr = [r"(left|right)_(shoulder|elbow|wrist)_.*"]
        finger_names = list(H2_OPEN_JOINT_POS)
        _, arm_names = robot.find_joints(arm_expr)
        self._arm_joint_names = arm_names

        # Lock non-IK coordinates in an anonymous session layer only. The
        # simulation retains every articulated finger; IK solves just the arms.
        # This also keeps LM's tiled solve within the GPU shared-memory budget.
        stage = Usd.Stage.Open(usd_path)
        stage.SetEditTarget(stage.GetSessionLayer())
        fixed_positions = dict(zip(robot.joint_names, robot.data.default_joint_pos.warp.numpy()[0], strict=True))
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
            floating=False,
            enable_self_collisions=False,
            load_visual_shapes=False,
            load_static_visual_shapes=False,
        )
        model = builder.finalize(device=self.device)
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
        self._velocity_limits = wp.clone(robot.data.joint_vel_limits.warp)
        self._targets = wp.clone(robot.data.default_joint_pos.warp)
        self._velocity_targets = wp.zeros_like(self._targets)
        wp.launch(
            _seed_coordinates,
            dim=(self.num_lanes, self.num_joints),
            inputs=[self._targets, self._indices, self._seed],
            device=self.device,
        )
        finger_ids, _ = robot.find_joints(finger_names, preserve_order=True)
        finger_indices = {joint: index for index, joint in enumerate(finger_ids)}
        self._fingers = wp.array(
            [finger_indices.get(i, -1) for i in range(self.num_joints)], dtype=int, device=self.device
        )
        # USD body transforms still describe the authored zero posture.
        # Evaluate the configured READY seed before constructing posture targets.
        initial_state = model.state()
        wp.copy(model.joint_q, self._seed, count=model.joint_coord_count)
        newton.eval_fk(model, model.joint_q, model.joint_qd, initial_state)
        body_poses = initial_state.body_q.numpy()
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
                position_weight=500.0,
                rotation_weight=50.0,
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
                NewtonIKJointLimitObjectiveCfg(weight=100.0),
            ]
        )
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
            NewtonIKSolverCfg(iterations=32, lambda_initial=0.1),
            model=model,
            num_envs=self.num_lanes,
            device=self.device,
            objectives=objectives,
            link_resolver=lambda name: links[name],
        )
        self._poses = [self._solver.objectives_by_name[name] for name in ("left_hand", "right_hand")]
        # The fixed torso target only needs one upload; hand and elbow targets advance per frame.
        torso_objective = self._solver.objectives_by_name["torso"]
        wp.launch(
            _set_pose_targets,
            dim=self.num_lanes,
            inputs=[
                torso_pose,
                0,
                self._root,
                torso_objective.position_objective.target_positions,
                torso_objective.rotation_objective.target_rotations,
            ],
            device=self.device,
        )
        self._elbow_positions = []
        for side in ("left", "right"):
            positions = self._solver.objectives_by_name[f"{side}_elbow"].position_objective.target_positions
            positions.assign(np.repeat(body_poses[links[f"{side}_elbow_link"], :3][None], self.num_lanes, axis=0))
            self._elbow_positions.append(positions)
        self._machine = _GraspStateMachine(dt, speeds, self.device)

    def initialize(
        self,
        model: newton.Model,
        state: newton.State,
        control: newton.Control,
        env_paths: list[str],
        substeps: int,
    ) -> None:
        """Bind scene observations and submit initial targets before graph capture.

        Register :meth:`compute` once per frame and :meth:`interpolate_arm_targets`
        after the native actuator adapter on each inner physics tick.

        Args:
            model: Native scene model containing one H2 per environment.
            state: Initial native joint state, used as the first ramp's start.
            control: Native control whose position targets are updated in place.
            env_paths: Environment prim paths, ordered like the controller lanes.
            substeps: Number of inner physics ticks per controller update.
        """
        if len(env_paths) != self.num_lanes or substeps < 1:
            raise ValueError("Expected one environment per lane and a positive substep count.")
        # Resolve all native labels once. These buffers are shared with the grasp monitor.
        body_names = ("pelvis", "left_elbow_link", "right_elbow_link", *self.hand_body_names)
        bodies = _resolve_robot_indices(model.body_label, env_paths, body_names)
        self._scene_state = state
        self._scene_roots = wp.array(bodies[:, 0], dtype=int, device=self.device)
        self._scene_elbows = [wp.array(bodies[:, side], dtype=int, device=self.device) for side in (1, 2)]
        self._scene_hands = wp.array(bodies[:, 3:], dtype=int, device=self.device)
        joints = _resolve_robot_indices(model.joint_label, env_paths, self._arm_joint_names)
        q_starts = model.joint_q_start.numpy()
        target_starts = model.joint_target_q_start.numpy()
        if not ((np.diff(q_starts)[joints] == 1).all() and (np.diff(target_starts)[joints] == 1).all()):
            raise ValueError("H2 arm joints must have scalar state and target coordinates.")
        q_coordinates = q_starts[joints]
        self._arm_target_coordinates = wp.array(target_starts[joints], dtype=int, device=self.device)
        initial_positions = wp.empty(q_coordinates.shape, dtype=float, device=self.device)
        wp.launch(
            _read_arm_positions,
            dim=q_coordinates.shape,
            inputs=[state.joint_q, wp.array(q_coordinates, dtype=int, device=self.device), initial_positions],
            device=self.device,
        )
        self._arm_start = initial_positions
        self._arm_end = wp.clone(initial_positions)
        self._arm_counters = wp.zeros(self.num_lanes, dtype=int, device=self.device)
        self._arm_substeps = substeps
        self._arm_control = control
        self.compute()

    def interpolate_arm_targets(self) -> None:
        """Apply the next linear arm-position sample after the native actuator adapter."""
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

    @staticmethod
    def duration(speeds: tuple[float, ...], dt: float) -> float:
        """Include per-phase frame rounding, the slowest pull, and a one-second hold."""
        return (
            sum(H2_PHASE_DURATIONS)
            + len(H2_PHASE_DURATIONS) * dt
            + PULL_TIME_FACTOR * H2_PULL_DISTANCE / min(speeds)
            + 1.0
        )

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
        # Soft posture targets discourage torso/elbow crossings while leaving X/Z free.
        for elbows, positions, guard in zip(self._scene_elbows, self._elbow_positions, (0.180, -0.180), strict=True):
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
        solved = self._solver.solve(self._seed)
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
                self._fingers,
                self._velocity_limits,
                self._machine.dt,
                self._targets,
                self._velocity_targets,
            ],
            device=self.device,
        )
        self.robot.actuators.target_command.set_position_index(value=self._targets)
        self.robot.actuators.target_command.set_velocity_index(value=self._velocity_targets)
        self.robot.write_data_to_sim()
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

    def __init__(self, model: newton.Model, state: newton.State, controller: TableclothController) -> None:
        device = controller.device
        count = controller.num_lanes
        self._state = state
        self._states = controller._machine._state
        self._hands = controller._scene_hands
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


def _resolve_robot_indices(labels: list[str], env_paths: list[str], names: list[str] | tuple[str, ...]) -> np.ndarray:
    """Resolve each requested native body or joint exactly once per H2 instance."""
    indices = np.empty((len(env_paths), len(names)), dtype=np.int32)
    for lane, env_path in enumerate(env_paths):
        selected = {}
        for index, label in enumerate(labels):
            name = label.rsplit("/", 1)[-1]
            if label.startswith(env_path + "/Robot/") and name in names:
                if name in selected:
                    raise ValueError(f"Duplicate H2 label {name!r} in {env_path!r}.")
                selected[name] = index
        if set(selected) != set(names):
            raise ValueError(f"Missing H2 labels in {env_path!r}: {set(names) - set(selected)}")
        indices[lane] = [selected[name] for name in names]
    return indices
