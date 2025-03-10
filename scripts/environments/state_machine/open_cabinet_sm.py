# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a cabinet opening state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    ./isaaclab.sh -p scripts/environments/state_machine/open_cabinet_sm.py --num_envs 32

"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for cabinet environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
from collections.abc import Sequence

import warp as wp

from isaaclab.sensors import FrameTransformer

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import CabinetEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# initialize warp
wp.init()


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class OpenDrawerSmState:
    """States for the cabinet drawer opening state machine."""

    REST = wp.constant(0)
    APPROACH_INFRONT_HANDLE = wp.constant(1)
    APPROACH_HANDLE = wp.constant(2)
    GRASP_HANDLE = wp.constant(3)
    OPEN_DRAWER = wp.constant(4)
    RELEASE_HANDLE = wp.constant(5)


class OpenDrawerSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.5)
    APPROACH_INFRONT_HANDLE = wp.constant(1.25)
    APPROACH_HANDLE = wp.constant(1.0)
    GRASP_HANDLE = wp.constant(1.0)
    OPEN_DRAWER = wp.constant(3.0)
    RELEASE_HANDLE = wp.constant(0.2)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    handle_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    handle_approach_offset: wp.array(dtype=wp.transform),
    handle_grasp_offset: wp.array(dtype=wp.transform),
    drawer_opening_rate: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == OpenDrawerSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= OpenDrawerSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = OpenDrawerSmState.APPROACH_INFRONT_HANDLE
            sm_wait_time[tid] = 0.0
    elif state == OpenDrawerSmState.APPROACH_INFRONT_HANDLE:
        des_ee_pose[tid] = wp.transform_multiply(handle_approach_offset[tid], handle_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= OpenDrawerSmWaitTime.APPROACH_INFRONT_HANDLE:
                # move to next state and reset wait time
                sm_state[tid] = OpenDrawerSmState.APPROACH_HANDLE
                sm_wait_time[tid] = 0.0
    elif state == OpenDrawerSmState.APPROACH_HANDLE:
        des_ee_pose[tid] = handle_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= OpenDrawerSmWaitTime.APPROACH_HANDLE:
                # move to next state and reset wait time
                sm_state[tid] = OpenDrawerSmState.GRASP_HANDLE
                sm_wait_time[tid] = 0.0
    elif state == OpenDrawerSmState.GRASP_HANDLE:
        des_ee_pose[tid] = wp.transform_multiply(handle_grasp_offset[tid], handle_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= OpenDrawerSmWaitTime.GRASP_HANDLE:
            # move to next state and reset wait time
            sm_state[tid] = OpenDrawerSmState.OPEN_DRAWER
            sm_wait_time[tid] = 0.0
    elif state == OpenDrawerSmState.OPEN_DRAWER:
        des_ee_pose[tid] = wp.transform_multiply(drawer_opening_rate[tid], handle_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= OpenDrawerSmWaitTime.OPEN_DRAWER:
            # move to next state and reset wait time
            sm_state[tid] = OpenDrawerSmState.RELEASE_HANDLE
            sm_wait_time[tid] = 0.0
    elif state == OpenDrawerSmState.RELEASE_HANDLE:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= OpenDrawerSmWaitTime.RELEASE_HANDLE:
            # move to next state and reset wait time
            sm_state[tid] = OpenDrawerSmState.RELEASE_HANDLE
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class OpenDrawerSm:
    """A simple state machine in a robot's task space to open a drawer in the cabinet.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_HANDLE: The robot moves towards the handle of the drawer.
    3. GRASP_HANDLE: The robot grasps the handle of the drawer.
    4. OPEN_DRAWER: The robot opens the drawer.
    5. RELEASE_HANDLE: The robot releases the handle of the drawer. This is the final state.
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
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

        # approach infront of the handle
        self.handle_approach_offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.handle_approach_offset[:, 0] = -0.1
        self.handle_approach_offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # handle grasp offset
        self.handle_grasp_offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.handle_grasp_offset[:, 0] = 0.025
        self.handle_grasp_offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # drawer opening rate
        self.drawer_opening_rate = torch.zeros((self.num_envs, 7), device=self.device)
        self.drawer_opening_rate[:, 0] = -0.015
        self.drawer_opening_rate[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.handle_approach_offset_wp = wp.from_torch(self.handle_approach_offset, wp.transform)
        self.handle_grasp_offset_wp = wp.from_torch(self.handle_grasp_offset, wp.transform)
        self.drawer_opening_rate_wp = wp.from_torch(self.drawer_opening_rate, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        # reset state machine
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, handle_pose: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        handle_pose = handle_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        handle_pose_wp = wp.from_torch(handle_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                handle_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.handle_approach_offset_wp,
                self.handle_grasp_offset_wp,
                self.drawer_opening_rate_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
    # parse configuration
    env_cfg: CabinetEnvCfg = parse_env_cfg(
        "Isaac-Open-Drawer-Franka-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make("Isaac-Open-Drawer-Franka-IK-Abs-v0", cfg=env_cfg)
    # reset environment at start
    env.reset()

    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    # create state machine
    open_sm = OpenDrawerSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = env.step(actions)[-2]

            # observations
            # -- end-effector frame
            ee_frame_tf: FrameTransformer = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = ee_frame_tf.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_rest_orientation = ee_frame_tf.data.target_quat_w[..., 0, :].clone()
            # -- handle frame
            cabinet_frame_tf: FrameTransformer = env.unwrapped.scene["cabinet_frame"]
            cabinet_position = cabinet_frame_tf.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            cabinet_orientation = cabinet_frame_tf.data.target_quat_w[..., 0, :].clone()

            # advance state machine
            actions = open_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                torch.cat([cabinet_position, cabinet_orientation], dim=-1),
            )

            # reset state machine
            if dones.any():
                open_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
