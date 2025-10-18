# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for velocity-based locomotion environments."""

# We leave this file empty since we don't want to expose any configs in this package directly.
# We still need this file to import the "config" module in the parent package.

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # parameters
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeGo2TeacherParkourEnvCfg",
        # Teacher Environment
        ## scene, height_scanner, contact_forces, robot, sky_light, terrain, ParkourDCMotorCfg, 
        ## observations:  
        #   3(root_ang_vel_b) + 2(imu = roll + pitch) +
        #   3(placeholder, delta_yaw, delta_next_yaw) + 
        #   5(x,y,yaw command,env_idx_tensor, invert_env_idx_tensor) + 
        #   36(joint-pos, joint-vel, history_joint) + 4(contact_sensor) = 53 + 
        #   132(measured_heights) + 9(base_lin_vel + 0 * base_lin_vel + 0 * base_lin_vel) + 
        #   29 (body_mass 1 + body_com 3 + friction coefficient 1 + joint_stiffness 12 + joint_damping 12)
        ## actions:  12(joint position) 
        ## commands: lin_vel_x, heading 
        ## rewards: reward_collision, reward_feet_edge, reward_torques, reward_dof_error, reward_hip_pos, reward_ang_vel_xy, reward_action_rate, reward_dof_acc, reward_lin_vel_z, reward_orientation, reward_feet_stumble, reward_tracking_goal_vel, reward_tracking_yaw, reward_delta_torques
        # terminations: total_terminates
        # parkours: ParkourEventsCfg -> base_parkour = parkours.ParkourEventsCfg(ParkourTermCfg); reach_goal_delay, next_goal_threshold
        # events: Events are things that happen to the robot or environment during simulation to change conditions, randomize, or test robustness.
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeGo2TeacherParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Eval-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeGo2TeacherParkourEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Student-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_student_cfg:UnitreeGo2StudentParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:UnitreeGo2ParkourStudentPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_student_cfg:UnitreeGo2StudentParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:UnitreeGo2ParkourStudentPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Student-Unitree-Go2-Eval-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_student_cfg:UnitreeGo2StudentParkourEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:UnitreeGo2ParkourStudentPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)
