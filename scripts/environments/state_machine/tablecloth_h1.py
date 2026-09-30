# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a scripted H1 expert on the manager-based tablecloth task.

The task owns scene construction, physics, actions, observations, resets,
rewards, and terminations. This script only generates the expert's bimanual
task-space commands with a GPU-resident Warp state machine.

.. code-block:: bash

    uv run --extra importers python scripts/environments/state_machine/tablecloth_h1.py
    uv run --extra importers python scripts/environments/state_machine/tablecloth_h1.py \
        --visualizer none --max_steps 312

    uv run --extra importers --extra isaacsim --extra video python scripts/environments/state_machine/tablecloth_h1.py \
        --visualizer kit --video

"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Run the scripted H1 tablecloth expert.")
parser.add_argument("--task", type=str, default="IsaacContrib-Tablecloth-H1", help="Task to run.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
parser.add_argument("--pull_speed", type=float, default=2.0, help="Peak X withdrawal speed [m/s].")
parser.add_argument("--video", action="store_true", help="Record the rollout to videos/tablecloth_h1/.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()

import gymnasium as gym
import torch
import warp as wp

from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.utils.math import subtract_frame_transforms

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.tablecloth.mdp._metrics import cloth_pull_distance
from isaaclab_tasks.utils import parse_env_cfg

FPS = 60
VIDEO_STEPS = 312
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

    def __init__(self, dt: float, torso_pose: torch.Tensor, action_dim: int, pull_speed: float):
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
        """Reset the expert for environments that have started a new episode."""
        wp.to_torch(self._state)[env_ids] = 0
        wp.to_torch(self._state_time)[env_ids] = 0.0
        wp.to_torch(self._pull_distance)[env_ids] = 0.0
        wp.to_torch(self._finger_fractions)[env_ids] = 0.0

    def compute(self) -> torch.Tensor:
        """Advance the expert and return its batched action tensor."""
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


def _torso_pose_in_robot_frame(env) -> torch.Tensor:
    robot = env.scene["robot"]
    torso_ids, _ = robot.find_bodies("torso_link")
    torso_pos, torso_quat = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        robot.data.body_pos_w.torch[:, torso_ids[0]],
        robot.data.body_quat_w.torch[:, torso_ids[0]],
    )
    return torch.cat((torso_pos, torso_quat), dim=-1)


def main() -> None:
    """Launch the task and run its scripted expert."""
    if not math.isfinite(args_cli.pull_speed) or args_cli.pull_speed <= 0.0:
        raise ValueError("--pull_speed must be finite and positive")
    if args_cli.video and "none" in (args_cli.visualizer or []):
        raise ValueError("--video requires a capture-capable visualizer; omit --visualizer none")
    if args_cli.video and "kit" in (args_cli.visualizer or []):
        args_cli.enable_cameras = True

    max_steps = VIDEO_STEPS if args_cli.video and args_cli.max_steps < 0 else args_cli.max_steps
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    success_term = env_cfg.terminations.success
    # The expert rollout should remain visible after task completion rather than auto-resetting.
    env_cfg.terminations.success = None
    env_cfg.terminations.tableware_fallen = None
    env_cfg.rewards.success = None
    if args_cli.video:
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                source="visualizer",
                output_dir="videos/tablecloth_h1",
                output_filename_prefix="tablecloth_h1",
                fps=FPS,
                video_length=max_steps,
            )
        ]

    with launch_simulation(cfg=env_cfg, launcher_args=args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        try:
            env.reset()
            state_machine = H1TableclothStateMachine(
                env.unwrapped.step_dt,
                _torso_pose_in_robot_frame(env.unwrapped),
                env.unwrapped.action_manager.total_action_dim,
                args_cli.pull_speed,
            )
            print("[INFO]: Setup complete. H1 tablecloth expert is ready.", flush=True)
            step = 0
            while env.unwrapped.sim.is_running() and (max_steps < 0 or step < max_steps):
                with torch.inference_mode():
                    _, _, terminated, truncated, _ = env.step(state_machine.compute())
                    dones = terminated | truncated
                    if dones.any():
                        state_machine.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))
                step += 1
            success = success_term.func(env.unwrapped, **success_term.params)
            displacements = {
                name: torch.linalg.vector_norm(
                    env.unwrapped.scene[name].data.root_pos_w.torch[:, :3]
                    - env.unwrapped.scene[name].data.default_root_pose.torch[:, :3]
                    - env.unwrapped.scene.env_origins,
                    dim=1,
                )
                for name in success_term.params["asset_names"]
            }
            maximum_displacement = torch.stack(list(displacements.values())).amax(dim=0)
            displacement_report = {name: value.tolist() for name, value in displacements.items()}
            print(
                f"[INFO]: Tablecloth trick success: {success.tolist()}, cloth travel [m]: "
                f"{cloth_pull_distance(env.unwrapped).tolist()}, "
                f"maximum tableware displacement [m]: {maximum_displacement.tolist()}, "
                f"by object: {displacement_report}",
                flush=True,
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
