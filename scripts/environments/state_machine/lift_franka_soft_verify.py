# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded verification harness for the Isaac-Lift-Soft-Franka-v0 example.

This is a self-terminating variant of ``lift_franka_soft.py``: it builds the same
deformable-lifting environment and runs the same warp state machine, but stops
after a fixed number of steps and prints heartbeats so the run can be verified
in an automated / headless context.

.. code-block:: bash

    isaaclab.bat -p scripts/environments/state_machine/lift_franka_soft_verify.py --headless
"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Bounded verification: pick and lift a deformable with a Franka arm.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--max_steps", type=int, default=600, help="Number of environment steps before exiting.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# disable metrics assembler due to scene graph instancing
from isaacsim.core.experimental.utils.app import enable_extension

enable_extension("omni.usd.metrics.assembler.ui", enabled=False)

"""Rest everything else."""

from collections.abc import Sequence

import gymnasium as gym
import torch
import warp as wp

from isaaclab.assets.deformable_object.deformable_object_data import DeformableObjectData

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift_franka_soft.franka_soft_env_cfg import FrankaSoftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# initialize warp
wp.init()


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)
    OPEN_GRIPPER = wp.constant(5)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.OPEN_GRIPPER
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.OPEN_GRIPPER:
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.OPEN_GRIPPER:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.OPEN_GRIPPER
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(1.0)
    APPROACH_OBJECT = wp.constant(1.0)
    GRASP_OBJECT = wp.constant(1.0)
    LIFT_OBJECT = wp.constant(1.5)
    OPEN_GRIPPER = wp.constant(0.0)


class PickAndLiftSm:
    """A simple state machine in a robot's task space to pick and lift an object."""

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.03):
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        # convert to torch
        return torch.cat([self.des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
    # parse configuration
    env_cfg: FrankaSoftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Soft-Franka-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.viewer.eye = (2.1, 1.0, 1.3)
    env = gym.make("Isaac-Lift-Soft-Franka-v0", cfg=env_cfg, render_mode=None)

    print(f"[VERIFY] Environment created: num_envs={env.unwrapped.num_envs}, device={env.unwrapped.device}", flush=True)
    print(f"[VERIFY] action_space={env.unwrapped.action_space.shape}", flush=True)

    # reset environment at start
    env.reset()
    print("[VERIFY] env.reset() completed.", flush=True)

    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    # desired rotation after grasping
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 0] = 1.0

    object_grasp_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    object_grasp_orientation[:, 0] = 1.0
    object_local_grasp_position = torch.tensor([0.0, 0.0, 0.0], device=env.unwrapped.device)

    # create state machine
    pick_sm = PickAndLiftSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)

    step_count = 0
    max_state_reached = 0
    initial_obj_z = None
    last_obj_z = None
    while simulation_app.is_running():
        with torch.inference_mode():
            # step environment
            dones = env.step(actions)[-2]

            # observations
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = (
                ee_frame_sensor.data.target_pos_w.torch[..., 0, :].clone() - env.unwrapped.scene.env_origins
            )
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w.torch[..., 0, :].clone()

            object_data: DeformableObjectData = env.unwrapped.scene["deformable"].data
            object_position = object_data.root_pos_w.torch - env.unwrapped.scene.env_origins
            object_position += object_local_grasp_position

            desired_position = env.unwrapped.command_manager.get_command("deformable_pose")[..., :3]

            # advance state machine
            actions = pick_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                torch.cat([object_position, object_grasp_orientation], dim=-1),
                torch.cat([desired_position, desired_orientation], dim=-1),
            )

            # track progress
            cur_state = int(pick_sm.sm_state.max().item())
            max_state_reached = max(max_state_reached, cur_state)
            obj_z = float(object_data.root_pos_w.torch[0, 2].item())
            if initial_obj_z is None:
                initial_obj_z = obj_z
            last_obj_z = obj_z

            if dones.any():
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

            step_count += 1
            if step_count % 50 == 0:
                print(
                    f"[VERIFY] step={step_count}/{args_cli.max_steps} sm_state={pick_sm.sm_state.tolist()} "
                    f"obj_z={obj_z:.4f}",
                    flush=True,
                )
            if step_count >= args_cli.max_steps:
                break

    lift_delta = (last_obj_z - initial_obj_z) if (initial_obj_z is not None and last_obj_z is not None) else 0.0
    print(
        f"[VERIFY] SUCCESS: completed {step_count} steps with no error. "
        f"max_sm_state={max_state_reached} (5=OPEN_GRIPPER), "
        f"object z: {initial_obj_z:.4f} -> {last_obj_z:.4f} (delta={lift_delta:+.4f} m)",
        flush=True,
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
