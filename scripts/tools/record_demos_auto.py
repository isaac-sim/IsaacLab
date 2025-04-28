# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to record demonstrations with Isaac Lab environments using human teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Device for interacting with environment. (default: keyboard)
    --dataset_file            File path to export recorded demos. (default: "./datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 0)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful. (default: 10)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.teleop_device.lower() == "handtracking":
    vars(args_cli)["experience"] = f'{os.environ["ISAACLAB_PATH"]}/apps/isaaclab.python.xr.openxr.kit'

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import time
import torch

import omni.log

from isaaclab.devices import Se3HandTracking, Se3Keyboard, Se3SpaceMouse
from isaaclab.envs import ViewerCfg,ManagerBasedRLEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def pre_process_actions(arm_action: torch.Tensor, open_gripper: bool) -> torch.Tensor:
    """Pre-process actions for the environment.
    gripper_command:  True  开   False  关
    """
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return arm_action
    else:
        # resolve gripper command
        gripper_vel = torch.zeros((arm_action.shape[0], 1), dtype=torch.float, device=arm_action.device)
        gripper_vel[:] = 1 if open_gripper else -1
        # compute actions
        return torch.concat([arm_action, gripper_vel], dim=1)


def get_waypoints(env:ManagerBasedRLEnv):
    """从场景中找到其中设定的路径点位置"""
    waypoint_states = env.obs_buf["policy"]["waypoint_states"]
    raw_waypoint_poses = waypoint_states[:, :7]
    hand_waypoint_poses = waypoint_states[:, 7:-1]
    waypoint_gripper_actions = waypoint_states[:, -1:]
    return raw_waypoint_poses,hand_waypoint_poses, waypoint_gripper_actions 


def gen_actions(env:ManagerBasedRLEnv):
    """将路点转换为末端执行器(ee)对应要求的任务空间的动作"""
    
    # 以观测的形式获取场景中定义的路点的位置以及夹爪动作命令
    raw_waypoint_poses,hand_waypoint_poses, gripper_actions = get_waypoints(env)
    
    # 随便写的一些动作，仅仅是为了占位，满足任务空间动作的形式要求 
    # ee_goal_wrench_set_tilted_task = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                                               device=env.device).repeat(raw_waypoint_poses.shape[0], 1)
    
    # 随便写的一些动作，仅仅是为了占位，满足任务空间动作的形式要求 
    kp_set_task = torch.tensor([420.0, 420.0, 420.0, 420.0, 420.0, 420.0],
                               device=env.device).repeat(raw_waypoint_poses.shape[0], 1)
    
    actions = torch.cat([hand_waypoint_poses, kp_set_task], dim=-1)
    # gripper 动作命令，  0： 关闭   1： 打开  -1： 不动
    gripper_commands = gripper_actions[:, 0]
    return raw_waypoint_poses, actions, gripper_commands



def execute_action(env:ManagerBasedRLEnv, arm_action: torch.Tensor, 
                      gripper_command: torch.Tensor, success_term=None, 
                      rate_limiter=None,marker:VisualizationMarkers=None,last_gripper_command:bool=None):
    """执行单次路点动作，包含ee动作和夹爪动作"""
    # 
    should_reset_recording_instance = False
    success_step_count = 0
    # convert to torch
    arm_action = torch.tensor(arm_action.clone().detach(),
                              dtype=torch.float, device=env.device).repeat(env.num_envs, 1)

    if gripper_command == -1:
        # 如果不动，则维持上一个夹爪动作
        bool_gripper_command = last_gripper_command
    elif gripper_command == 1: # 要求打开
        bool_gripper_command = True
    else: # 要求关闭
        bool_gripper_command = False

    # 夹爪动作置为false, 在执行arm动作时不执行夹爪动作
    ee_action = pre_process_actions(arm_action, open_gripper=last_gripper_command)

    gripper_action = pre_process_actions(arm_action, open_gripper=bool_gripper_command)
    
    # 先执行ee动作,夹爪保持不变
    #while True: # 当夹爪还没有到达目标位置时，不停循环执行动作
    for _ in range(50):

        # perform action on environment
        env.step(ee_action)
        # 计算当前末端执行器的位姿
        current_ee_pos = env.scene
        # 显示当前ee手指中心的位置
        marker.visualize(env.obs_buf["policy"]["ee_pos"], env.obs_buf["policy"]["ee_quat"])
        
        # 判断是否成功
        if success_term is not None:
            if bool(success_term.func(env, **success_term.params)[0]):
                success_step_count += 1
                # 检查当前连续成功的步数（success_step_count）是否达到预设阈值
                if success_step_count >= args_cli.num_success_steps: 
                    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                    env.recorder_manager.set_success_to_episodes(
                        [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes([0])
                    should_reset_recording_instance = True
            else:
                success_step_count = 0


        # TODO: 这里需要检查当前末端执行器的位姿是否到达了目标位置,我这里简写了一下是固定的循环次数
        if env.sim.is_stopped()  or should_reset_recording_instance:
            break
        # 
        if rate_limiter:
            rate_limiter.sleep(env)


    # 再执行gripper动作
    current_gripper_state = env.obs_buf["policy"]["gripper_state"]
    if bool_gripper_command != current_gripper_state and not should_reset_recording_instance:
        for _ in range(30):
            # perform gripper action on environment
            env.step(gripper_action)
        
            # 判断是否成功
            if success_term is not None:
                if bool(success_term.func(env, **success_term.params)[0]):
                    success_step_count += 1
                    if success_step_count >= args_cli.num_success_steps:
                        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                        env.recorder_manager.set_success_to_episodes(
                            [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                        )
                        env.recorder_manager.export_episodes([0])
                        should_reset_recording_instance = True
                else:
                    success_step_count = 0
            # 检查是否打断
            if env.sim.is_stopped() or should_reset_recording_instance:
                break
            
            if rate_limiter:
                rate_limiter.sleep(env)

    # 更新上一个指令
    last_gripper_command = bool_gripper_command

    return should_reset_recording_instance,last_gripper_command


def main():
    """Collect demonstrations from the environment using teleop interfaces."""

    # if handtracking is selected, rate limiting is achieved via OpenXR
    if args_cli.teleop_device.lower() == "handtracking":
        rate_limiter = None
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task

    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        omni.log.warn(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None

    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped


    should_reset_recording_instance = False

    # def reset_recording_instance():
    #     nonlocal should_reset_recording_instance
    #     should_reset_recording_instance = True

    # # create controller
    # if args_cli.teleop_device.lower() == "keyboard":
    #     teleop_interface = Se3Keyboard(pos_sensitivity=0.2, rot_sensitivity=0.5)
    # elif args_cli.teleop_device.lower() == "spacemouse":
    #     teleop_interface = Se3SpaceMouse(pos_sensitivity=0.2, rot_sensitivity=0.5)
    # elif args_cli.teleop_device.lower() == "handtracking":
    #     from isaacsim.xr.openxr import OpenXRSpec

    #     teleop_interface = Se3HandTracking(OpenXRSpec.XrHandEXT.XR_HAND_RIGHT_EXT, False, True)
    #     teleop_interface.add_callback("RESET", reset_recording_instance)
    #     viewer = ViewerCfg(eye=(-0.25, -0.3, 0.5), lookat=(0.6, 0, 0), asset_name="viewer")
    #     ViewportCameraController(env, viewer)
    # else:
    #     raise ValueError(
    #         f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse', 'handtracking'."
    #     )

    # teleop_interface.add_callback("R", reset_recording_instance)
    # print(teleop_interface)

    # reset before starting
    env.reset()
    # teleop_interface.reset()

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
    
    
    current_recorded_demo_count = 0
    
    # 一直自动生成，直到到达指定的成功demo数量
    while current_recorded_demo_count < args_cli.num_demos:

        # 获取当前回合场景中的路点以及对应的夹爪动作
        raw_waypoint_poses, actions, gripper_commands = gen_actions(env)
        # 默认初始的gripper 是 打开的动作
        last_gripper_command = True

        for waypoint_idx in range(actions.shape[0]):
            # update marker positions
            # 显示原始的路点位置姿态
            goal_marker.visualize(raw_waypoint_poses[waypoint_idx][None,0:3], raw_waypoint_poses[waypoint_idx][None,3:7])
            # 执行动作
            should_reset_recording_instance,last_gripper_command = execute_action(env, actions[waypoint_idx], 
                                                            gripper_commands[waypoint_idx], 
                                                            success_term=success_term, 
                                                            rate_limiter=rate_limiter,
                                                            marker=ee_marker,
                                                            last_gripper_command=last_gripper_command)
            if should_reset_recording_instance:
                # 退出当前回合
                break

        # 执行完一个回合的所有路点之后，打印出当前完成的demo数量
        if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
            current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
            print(f"Recorded {current_recorded_demo_count} successful demonstrations.")

        env.recorder_manager.reset()
        env.reset()

    # 完成所有的demo之后退出环境
    print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
    env.close() 
    
    

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
