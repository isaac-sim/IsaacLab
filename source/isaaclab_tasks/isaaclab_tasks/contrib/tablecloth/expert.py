# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""GPU-resident scripted expert for the contributed H1 tablecloth task.

The calibrated grasp and pull trajectory generate actions for the task. Scene
construction, physics, stepping, resets, rewards, and terminations remain owned
by the environment.
"""

from __future__ import annotations

import torch
import warp as wp

PULL_DISTANCE = 0.40
ARM_ACTION_DIM = 21

FINGERS_OPEN = (0.0, 0.0, 0.0, 0.0, 0.0)
FINGERS_CURLED = (0.0, 0.0, 0.0, 0.0, 0.80)
FINGERS_INSERTED = (0.0, 0.0, 0.75, 0.75, 0.80)
FINGERS_PREPINCHED = (0.0, 0.0, 0.737080, 0.713855, 0.80)
FINGERS_PINCHED = (1.0, 1.0, 0.737080, 0.713855, 0.80)

PHASE_DURATIONS = (0.50, 0.80, 0.60, 0.50, 0.60, 0.60, 0.10, 0.10)
PHASE_KEYFRAMES = ((0, 0), (0, 2), (2, 4), (4, 6), (6, 8), (8, 8), (8, 10), (10, 10))
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

# Positions are expressed in the robot root frame, matching the absolute Newton IK action contract.
HAND_KEYFRAMES = (
    (0.27, 0.24, 0.140),
    (0.27, -0.24, 0.140),
    (0.43, 0.38, 0.060),
    (0.43, -0.38, 0.060),
    (0.43, 0.38, -0.050),
    (0.43, -0.38, -0.048),
    (0.545, 0.38, -0.050),
    (0.545, -0.38, -0.048),
    (0.545, 0.38, -0.020),
    (0.545, -0.38, -0.020),
    (0.545, 0.38, -0.015),
    (0.545, -0.38, -0.015),
)
HAND_ROTATIONS = (
    (-0.09022585, 0.46115433, 0.03007528, 0.88220828),
    (0.09023000, 0.46114998, -0.03008000, 0.88220997),
)
# Feasible corner-grasp attitudes for H1's five-DOF arms, calibrated with forward kinematics.
GRASP_HAND_ROTATIONS = (
    (-0.07698878, 0.37041666, 0.08826291, 0.92145205),
    (0.07860907, 0.37042228, -0.08392787, 0.92171799),
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


@wp.func
def _smoothstep(value: float) -> float:
    value = wp.clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


@wp.func
def _pull_offset(distance: float) -> wp.vec3:
    # This arc preserves the grasp attitude within the arms' reachable pose manifold.
    radius = 0.3072
    center = 0.2248
    height = wp.sqrt(radius * radius - center * center) - wp.sqrt(
        radius * radius - (distance - center) * (distance - center)
    )
    return wp.vec3(-distance, 0.0, height)


@wp.kernel
def _infer_state_machine(
    dt: float,
    pull_speed: float,
    keyframes: wp.array(dtype=wp.vec3),
    hand_rotations: wp.array(dtype=wp.vec4),
    torso_pose: wp.array2d(dtype=float),
    phase_durations: wp.array(dtype=float),
    phase_start_keyframes: wp.array(dtype=wp.int32),
    phase_end_keyframes: wp.array(dtype=wp.int32),
    phase_start_fingers: wp.array(dtype=float),
    phase_end_fingers: wp.array(dtype=float),
    finger_closed_values: wp.array(dtype=float),
    finger_groups: wp.array(dtype=wp.int32),
    finger_fractions: wp.array2d(dtype=float),
    state: wp.array(dtype=wp.int32),
    state_time: wp.array(dtype=float),
    pull_distance: wp.array(dtype=float),
    actions: wp.array2d(dtype=float),
):
    env_id = wp.tid()
    current_state = state[env_id]
    elapsed = state_time[env_id] + dt

    if current_state < STATE_PULL and elapsed >= phase_durations[current_state]:
        current_state += 1
        elapsed = 0.0

    left = keyframes[0]
    right = keyframes[1]
    if current_state < STATE_PULL:
        alpha = _smoothstep(elapsed / phase_durations[current_state])
        start_keyframe = phase_start_keyframes[current_state]
        end_keyframe = phase_end_keyframes[current_state]
        left = wp.lerp(keyframes[start_keyframe], keyframes[end_keyframe], alpha)
        right = wp.lerp(keyframes[start_keyframe + 1], keyframes[end_keyframe + 1], alpha)
        for group in range(5):
            index = current_state * 5 + group
            finger_fractions[env_id, group] = wp.lerp(phase_start_fingers[index], phase_end_fingers[index], alpha)
    elif current_state == STATE_PULL:
        # Accelerate and brake to rest without a velocity discontinuity at the final hold.
        duration = 1.5 * PULL_DISTANCE / pull_speed
        distance = PULL_DISTANCE * _smoothstep(elapsed / duration)
        pull_distance[env_id] = distance
        offset = _pull_offset(distance)
        left = keyframes[10] + offset
        right = keyframes[11] + offset
        for group in range(5):
            finger_fractions[env_id, group] = phase_end_fingers[(STATE_PULL - 1) * 5 + group]
        if distance >= PULL_DISTANCE:
            current_state = STATE_HOLD
    else:
        distance = pull_distance[env_id]
        offset = _pull_offset(distance)
        left = keyframes[10] + offset
        right = keyframes[11] + offset
        for group in range(5):
            finger_fractions[env_id, group] = phase_end_fingers[(STATE_PULL - 1) * 5 + group]

    # Finish changing attitude before closing the thumb, not while transporting the cloth.
    rotation_alpha = 0.0
    if current_state >= 5:
        rotation_alpha = 1.0
    elif current_state == 4:
        rotation_alpha = _smoothstep(elapsed / phase_durations[4])
    left_rotation = wp.normalize(wp.lerp(hand_rotations[0], hand_rotations[2], rotation_alpha))
    right_rotation = wp.normalize(wp.lerp(hand_rotations[1], hand_rotations[3], rotation_alpha))

    for axis in range(3):
        actions[env_id, axis] = left[axis]
        actions[env_id, 7 + axis] = right[axis]
    for axis in range(4):
        actions[env_id, 3 + axis] = left_rotation[axis]
        actions[env_id, 10 + axis] = right_rotation[axis]
    for axis in range(7):
        actions[env_id, 14 + axis] = torso_pose[env_id, axis]

    for finger in range(finger_closed_values.shape[0]):
        fraction = finger_fractions[env_id, finger_groups[finger]]
        actions[env_id, ARM_ACTION_DIM + finger] = fraction * finger_closed_values[finger]

    state[env_id] = current_state
    state_time[env_id] = elapsed


class H1TableclothStateMachine:
    """Generate batched absolute hand poses and finger targets on the simulation device."""

    def __init__(self, dt: float, torso_pose: torch.Tensor, action_dim: int, pull_speed: float) -> None:
        """Initialize the expert on the torso-pose tensor's device.

        Args:
            dt: Environment step duration [s].
            torso_pose: Robot-root-frame torso position [m] and quaternion, shape [N, 7].
            action_dim: Number of task action components.
            pull_speed: Peak withdrawal speed [m/s].

        Raises:
            ValueError: If the task action dimension does not match the expert.
        """
        self.num_envs = torso_pose.shape[0]
        self.device = torso_pose.device
        self._warp_device = wp.device_from_torch(self.device)
        self.dt = dt
        self.pull_speed = pull_speed
        expected_action_dim = ARM_ACTION_DIM + len(FINGER_CLOSED_VALUES)
        if action_dim != expected_action_dim:
            raise ValueError(f"Expected a {expected_action_dim}-D H1 action, received {action_dim}")

        phases = list(zip(PHASE_DURATIONS, PHASE_KEYFRAMES, PHASE_FINGERS, strict=True))
        self._keyframes = wp.array(HAND_KEYFRAMES, dtype=wp.vec3, device=self._warp_device)
        self._hand_rotations = wp.array(
            (*HAND_ROTATIONS, *GRASP_HAND_ROTATIONS), dtype=wp.vec4, device=self._warp_device
        )
        self._torso_pose = wp.from_torch(torso_pose.contiguous(), dtype=wp.float32)
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
        self._finger_closed_values = wp.array(FINGER_CLOSED_VALUES, dtype=float, device=self._warp_device)
        self._finger_groups = wp.array(FINGER_GROUPS, dtype=wp.int32, device=self._warp_device)
        self._finger_fractions = wp.zeros((self.num_envs, 5), dtype=float, device=self._warp_device)
        self._state = wp.zeros(self.num_envs, dtype=wp.int32, device=self._warp_device)
        self._state_time = wp.zeros(self.num_envs, dtype=float, device=self._warp_device)
        self._pull_distance = wp.zeros(self.num_envs, dtype=float, device=self._warp_device)
        self.actions = torch.zeros((self.num_envs, action_dim), device=self.device)
        self._actions = wp.from_torch(self.actions, dtype=wp.float32)

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset the expert for environments that have started a new episode.

        Args:
            env_ids: Indices of the environments to reset.
        """
        wp.to_torch(self._state)[env_ids] = 0
        wp.to_torch(self._state_time)[env_ids] = 0.0
        wp.to_torch(self._pull_distance)[env_ids] = 0.0
        wp.to_torch(self._finger_fractions)[env_ids] = 0.0

    def compute(self) -> torch.Tensor:
        """Advance the expert and return its batched action tensor.

        Returns:
            Expert-owned actions, shape [N, 45], containing absolute robot-root-frame
            poses and finger joint targets [rad]. The next call overwrites this tensor.
        """
        wp.launch(
            _infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.dt,
                self.pull_speed,
                self._keyframes,
                self._hand_rotations,
                self._torso_pose,
                self._phase_durations,
                self._phase_start_keyframes,
                self._phase_end_keyframes,
                self._phase_start_fingers,
                self._phase_end_fingers,
                self._finger_closed_values,
                self._finger_groups,
                self._finger_fractions,
                self._state,
                self._state_time,
                self._pull_distance,
                self._actions,
            ],
            device=self._warp_device,
        )
        return self.actions
