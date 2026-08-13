# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to demonstrate lifting a deformable object with a robotic arm.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    # Kitless run with the Newton OpenGL viewer (default).
    uv run python scripts/environments/state_machine/lift_franka_soft.py

    # Headless.
    uv run python scripts/environments/state_machine/lift_franka_soft.py --viz none

"""

import argparse
import sys
from collections.abc import Sequence

import gymnasium as gym
import torch
import warp as wp

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets.deformable_object.deformable_object_data import DeformableObjectData
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift a deformable with a robotic arm.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of environment steps to run.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Soft-Franka", help="The task to run.")
add_launcher_args(parser)
# the task runs on Newton, so default to the kitless viewer
parser.set_defaults(visualizer=["newton"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

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
        # des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.OPEN_GRIPPER:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.OPEN_GRIPPER
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching.

    Wait times are generous because the low-PD Franka takes a while to settle on each IK target.
    """

    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(1.0)
    APPROACH_OBJECT = wp.constant(1.5)
    GRASP_OBJECT = wp.constant(1.5)
    LIFT_OBJECT = wp.constant(1.5)
    OPEN_GRIPPER = wp.constant(0.0)


class PickAndLiftSm:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_OBJECT: The robot moves above the object.
    3. APPROACH_OBJECT: The robot moves to the object.
    4. GRASP_OBJECT: The robot grasps the object.
    5. LIFT_OBJECT: The robot lifts the object to the desired pose. This is the final state.
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.03):
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
    # parse configuration via Hydra, so presets can be selected on the CLI (e.g. presets=isaacsim_physx)
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    # Scripted demo: keep only the time-out, extended to 10 s so the slow low-PD Franka can finish a
    # pick-and-lift, and drop the failure terminations so a transient bound or velocity spike does
    # not cut a run short.
    env_cfg.episode_length_s = 10.0
    for term_name in list(vars(env_cfg.terminations)):
        if term_name != "time_out":
            setattr(env_cfg.terminations, term_name, None)
    # the state machine emits absolute end-effector poses, so pick the task's own IK action preset
    # (e.g. the cloth closes the gripper fully); the env otherwise defaults to relative joint
    # targets, which RL trains on.
    env_cfg.actions = type(env_cfg)().actions.ik
    env_cfg.viewer.eye = (1.3, 0.6, 0.5)
    env_cfg.viewer.lookat = (0.5, 0.0, 0.05)
    env_cfg.sim.default_visualizer_cfg = VisualizerCfg(eye=env_cfg.viewer.eye, lookat=env_cfg.viewer.lookat)

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        is_cable = "cable" in env.unwrapped.scene.keys()

        # reset environment at start
        env.reset()

        # create action buffers (position + quaternion)
        actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
        actions[:, 3] = 1.0
        # desired rotation after grasping
        desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
        desired_orientation[:, 0] = 1.0

        # Top-down approach: identity quaternion (wxyz, w=1) aligns panda_hand with the Franka root,
        # giving the canonical top-down grasp pose. The bar lies along world-X, so the gripper
        # closes across its short side without any wrist twist.
        object_grasp_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
        object_grasp_orientation[:, 0] = 1.0
        # Grasp 1 cm below the deformable's centre of mass, so the fingers close around its lower half.
        # The cloth drapes over a support cube, so its COM sits well below the graspable fold; reach
        # 8 cm higher to close on the raised cloth instead of the table.
        grasp_z = -0.01 + (0.08 if "Cloth" in args_cli.task else 0.0)
        object_local_grasp_position = torch.tensor([0.0, 0.0, grasp_z], device=env.unwrapped.device)

        # create state machine
        pick_sm = PickAndLiftSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)

        for _ in range(args_cli.num_steps):
            # run everything in inference mode
            with torch.inference_mode():
                # step environment
                _, _, terminated, time_outs, _ = env.step(actions)
                dones = terminated | time_outs

                # reset state machine
                if dones.any():
                    pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

                # observations
                # -- end-effector frame
                ee_frame_sensor = env.unwrapped.scene["ee_frame"]
                tcp_rest_position = (
                    ee_frame_sensor.data.target_pos_w.torch[..., 0, :].clone() - env.unwrapped.scene.env_origins
                )
                tcp_rest_orientation = ee_frame_sensor.data.target_quat_w.torch[..., 0, :].clone()
                # -- object frame
                if is_cable:
                    segment_index = env_cfg.commands.cable_pose.segment_index
                    object_position = (
                        env.unwrapped.scene["cable"].data.segment_pose_w.torch[:, segment_index, :3]
                        - env.unwrapped.scene.env_origins
                    )
                    command_name = "cable_pose"
                else:
                    object_data: DeformableObjectData = env.unwrapped.scene["deformable"].data
                    object_position = object_data.root_pos_w.torch - env.unwrapped.scene.env_origins
                    object_position += object_local_grasp_position
                    command_name = "deformable_pose"

                # -- target object frame
                desired_position = env.unwrapped.command_manager.get_command(command_name)[..., :3]

                # advance state machine
                actions = pick_sm.compute(
                    torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                    torch.cat([object_position, object_grasp_orientation], dim=-1),
                    torch.cat([desired_position, desired_orientation], dim=-1),
                )

        # close the environment
        env.close()


if __name__ == "__main__":
    main()
