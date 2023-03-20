"""
Script to run an environment with a pick and lift state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

Usage:
    python play_lift.py --num_envs 128
"""

"""Launch Omniverse Toolkit first."""

import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
args_cli = parser.parse_args()

# launch the simulator
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)

"""Rest everything else."""

import gym
import torch
from enum import Enum
from typing import Sequence, Union

import warp as wp

from omni.isaac.orbit.utils.timer import Timer

import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils.parse_cfg import parse_env_cfg

# initialize warp
wp.init()


class GripperState(Enum):
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState(Enum):
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)
    DROP_OBJECT = wp.constant(5)


class PickSmWaitTime(Enum):
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(1.0)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.3)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(2.0)
    DROP_OBJECT = wp.constant(0.2)


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=wp.float32),
    sm_state: wp.array(dtype=wp.int32),
    sm_wait_time: wp.array(dtype=wp.float32),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=wp.float32),
    offset: wp.array(dtype=wp.transform),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickSmState.REST.value:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN.value
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST.value:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT.value
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT.value:
        des_ee_pose[tid] = object_pose[tid]
        # TODO: This is causing issues.
        # des_ee_pose[tid] = wp.transform_multiply(des_ee_pose[tid], offset[tid])
        gripper_state[tid] = GripperState.OPEN.value
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT.value:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_OBJECT.value
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT.value:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN.value
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT.value:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.GRASP_OBJECT.value
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT.value:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE.value
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT.value:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT.value
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT.value:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE.value
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT.value:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.DROP_OBJECT.value
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.DROP_OBJECT.value:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.OPEN.value
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.DROP_OBJECT.value:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.REST.value
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


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
    5. LIFT_OBJECT: The robot lifts the object.
    6. DROP_OBJECT: The robot drops the object.
    """

    def __init__(self, dt: float, num_envs: int, device: Union[torch.device, str] = "cpu"):
        """Initialize the state machine.

        Args:
            dt (float): The environment time step.
            num_envs (int): The number of environments to simulate.
            device (Union[torch.device, str], optional): The device to run the state machine on.
        """
        # save parameters
        self.dt = dt
        self.num_envs = num_envs
        self.device = device
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, dtype=torch.float32, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0, dtype=torch.float32, device=self.device)
        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device)
        self.offset[:, 2] = 0.1
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
            env_ids = ...
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
            ],
        )
        wp.synchronize()
        # convert to torch
        return torch.cat([self.des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
    # parse configuration
    env_cfg = parse_env_cfg("Isaac-Lift-Franka-v0", use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    # -- control configuration
    env_cfg.control.control_type = "inverse_kinematics"
    env_cfg.control.inverse_kinematics.command_type = "position_abs"
    # -- randomization configuration
    env_cfg.randomization.object_initial_pose.position_cat = "uniform"
    env_cfg.randomization.object_desired_pose.position_cat = "uniform"
    # -- termination configuration
    env_cfg.terminations.episode_timeout = False
    # -- robot configuration
    env_cfg.robot.robot_type = "franka"
    # create environment
    env = gym.make("Isaac-Lift-Franka-v0", cfg=env_cfg, headless=args_cli.headless)

    # create action buffers
    actions = torch.zeros((env.num_envs, env.action_space.shape[0]), device=env.device)
    # create state machine
    pick_sm = PickAndLiftSm(env.dt, env.num_envs, env.device)

    # reset environment
    env.reset()

    # run state machine
    for _ in range(10000):
        # step environment
        dones = env.step(actions)[-2]

        # observations
        ee_pose = env.robot.data.ee_state_w[:, :7].clone()
        object_pose = env.object.data.root_state_w[:, :7].clone()
        des_object_pose = env.object_des_pose_w.clone()
        # transform from world to base frames
        ee_pose[:, :3] -= env.robot.data.root_pos_w
        object_pose[:, :3] -= env.robot.data.root_pos_w
        des_object_pose[:, :3] -= env.robot.data.root_pos_w
        # advance state machine
        with Timer("state machine"):
            sm_actions = pick_sm.compute(ee_pose, object_pose, des_object_pose)

        # set actions for IK with positions
        actions[:, :3] = sm_actions[:, :3]
        actions[:, -1] = sm_actions[:, -1]
        # reset state machine
        if dones.any():
            pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))


if __name__ == "__main__":
    # run main function
    main()
    # close simulation
    simulation_app.close()
