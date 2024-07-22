# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a state machine to screw a nut onto a bolt.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    ./isaaclab.sh -p source/standalone/environments/state_machine/screw_nut_sm.py --num_envs 32

"""

"""Launch Omniverse Toolkit first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Screw a nut onto a bolt using a state machine.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
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


import gymnasium as gym
import torch
from collections.abc import Sequence

import warp as wp

from omni.isaac.lab.assets.rigid_object.rigid_object_data import RigidObjectData

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.manager_based.manipulation.screw.screw_env_cfg import ScrewEnvCfg
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

# initialize warp
wp.init()


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class ScrewNutState:
    """States for the screw nut state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_NUT = wp.constant(1)
    APPROACH_NUT = wp.constant(2)
    GRASP_NUT = wp.constant(3)
    APPROACH_ABOVE_BOLT = wp.constant(4)
    APPROACH_BOLT = wp.constant(5)
    SCREW_NUT = wp.constant(6)
    RELEASE_NUT = wp.constant(7)


class ScrewNutSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_NUT = wp.constant(2.5)
    APPROACH_NUT = wp.constant(1.6)
    GRASP_NUT = wp.constant(1.3)
    APPROACH_ABOVE_BOLT = wp.constant(2.0)
    APPROACH_BOLT = wp.constant(2.6)
    SCREW_NUT = wp.constant(3.0)
    RELEASE_NUT = wp.constant(0.3)


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    nut_pose: wp.array(dtype=wp.transform),
    bolt_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    above_nut_offset: wp.array(dtype=wp.transform),
    grasp_nut_offset: wp.array(dtype=wp.transform),
    above_bolt_offset: wp.array(dtype=wp.transform),
    on_bolt_offset: wp.array(dtype=wp.transform),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == ScrewNutState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= ScrewNutSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = ScrewNutState.APPROACH_ABOVE_NUT
            sm_wait_time[tid] = 0.0
    elif state == ScrewNutState.APPROACH_ABOVE_NUT:
        des_ee_pose[tid] = wp.transform_multiply(above_nut_offset[tid], nut_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= ScrewNutSmWaitTime.APPROACH_NUT:
            # move to next state and reset wait time
            sm_state[tid] = ScrewNutState.APPROACH_NUT
            sm_wait_time[tid] = 0.0
    elif state == ScrewNutState.APPROACH_NUT:
        des_ee_pose[tid] = wp.transform_multiply(grasp_nut_offset[tid], nut_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= ScrewNutSmWaitTime.APPROACH_NUT:
            # move to next state and reset wait time
            sm_state[tid] = ScrewNutState.GRASP_NUT
            sm_wait_time[tid] = 0.0
    elif state == ScrewNutState.GRASP_NUT:
        des_ee_pose[tid] = wp.transform_multiply(grasp_nut_offset[tid], nut_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= ScrewNutSmWaitTime.GRASP_NUT:
            # move to next state and reset wait time
            sm_state[tid] = ScrewNutState.APPROACH_ABOVE_BOLT
            sm_wait_time[tid] = 0.0
    elif state == ScrewNutState.APPROACH_ABOVE_BOLT:
        des_ee_pose[tid] = wp.transform_multiply(above_bolt_offset[tid], bolt_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= ScrewNutSmWaitTime.APPROACH_ABOVE_BOLT:
            # move to next state and reset wait time
            sm_state[tid] = ScrewNutState.APPROACH_BOLT
            sm_wait_time[tid] = 0.0
    elif state == ScrewNutState.APPROACH_BOLT:
        des_ee_pose[tid] = wp.transform_multiply(on_bolt_offset[tid], bolt_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= ScrewNutSmWaitTime.APPROACH_BOLT:
            # move to next state and reset wait time
            sm_state[tid] = ScrewNutState.SCREW_NUT
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class ScrewNutSm:
    """A simple state machine in a robot's task space to screw a nut onto a bolt.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_NUT: The robot moves above the nut.
    3. APPROACH_NUT: The robot moves to the nut.
    4. GRASP_NUT: The robot grasps the nut.
    5. APPROACH_ABOVE_BOLT: The robot moves above the bolt.
    6. APPROACH_BOLT: The robot moves to the bolt.
    7. SCREW_NUT: The robot screws the nut onto the bolt.
    8. RELEASE_NUT: The robot releases the nut.
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu"):
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
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # constants for nut and bolts
        nut_height = 0.016
        bolt_height = 0.1
        screw_speed = 60.0 / 180.0 * torch.pi
        screw_limit_angle = 60.0 / 180.0 * torch.pi
        # constant offsets for the state machine
        # -- approach above nut
        self.above_nut_offset = torch.zeros((self.num_envs, 7), device=self.device)
        # self.above_nut_offset[:, 2] = 0.08 + nut_height
        self.above_nut_offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)
        # -- approach nut
        self.grasp_nut_offset = torch.zeros_like(self.above_nut_offset)
        # self.grasp_nut_offset[:, 2] = 0.12 + nut_height
        self.grasp_nut_offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)
        # -- approach above bolt
        self.above_bolt_offset = torch.zeros_like(self.above_nut_offset)
        self.above_bolt_offset[:, 2] = 0.02 + bolt_height
        self.above_bolt_offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)
        # -- approach bolt
        self.on_bolt_offset = torch.zeros_like(self.above_nut_offset)
        self.on_bolt_offset[:, 2] = 0.8 * bolt_height
        self.on_bolt_offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        # inputs
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        # outputs
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        # constants
        self.above_nut_offset_wp = wp.from_torch(self.above_nut_offset, wp.transform)
        self.grasp_nut_offset_wp = wp.from_torch(self.grasp_nut_offset, wp.transform)
        self.above_bolt_offset_wp = wp.from_torch(self.above_bolt_offset, wp.transform)
        self.on_bolt_offset_wp = wp.from_torch(self.on_bolt_offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, nut_pose: torch.Tensor, bolt_pose: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        nut_pose = nut_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        bolt_pose = bolt_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        nut_pose_wp = wp.from_torch(nut_pose.contiguous(), wp.transform)
        bolt_pose_wp = wp.from_torch(bolt_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                # inputs
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                nut_pose_wp,
                bolt_pose_wp,
                # outputs
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                # constants
                self.above_nut_offset_wp,
                self.grasp_nut_offset_wp,
                self.above_bolt_offset_wp,
                self.on_bolt_offset_wp,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
    # parse configuration
    env_cfg: ScrewEnvCfg = parse_env_cfg(
        "Isaac-Screw-Nut-Franka-IK-Abs-v0",
        use_gpu=not args_cli.cpu,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make("Isaac-Screw-Nut-Franka-IK-Abs-v0", cfg=env_cfg)
    # reset environment at start
    env.reset()

    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    # create state machine
    screw_sm = ScrewNutSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = env.step(actions)[-2]

            # observations
            # -- end-effector frame
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            # -- nut frame
            nut_pose: RigidObjectData = env.unwrapped.scene["nut"].data.root_state_w[:, :7].clone()
            nut_pose[:, :3] -= env.unwrapped.scene.env_origins
            # -- bolt frame
            bolt_pose: RigidObjectData = env.unwrapped.scene["bolt"].data.root_state_w[:, :7].clone()
            bolt_pose[:, :3] -= env.unwrapped.scene.env_origins

            # advance state machine
            actions = screw_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1), nut_pose, bolt_pose
            )
            actions[:, 3] = 0.0
            actions[:, 4] = 1.0
            actions[:, 5] = 0.0
            actions[:, 6] = 0.0

            # reset state machine
            if dones.any():
                screw_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
