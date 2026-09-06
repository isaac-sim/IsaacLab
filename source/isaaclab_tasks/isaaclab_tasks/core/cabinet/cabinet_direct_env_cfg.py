# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.core.cabinet.cabinet_env_cfg import (
    CABINET_CFG,
    LIGHT_CFG,
    PLANE_CFG,
    CabinetPhysicsCfg,
    EventCfg,
)


@configclass
class CabinetDirectSceneCfg(InteractiveSceneCfg):
    """Scene configuration shared by direct-workflow cabinet tasks."""

    robot: ArticulationCfg = MISSING
    cabinet: ArticulationCfg = CABINET_CFG
    plane: AssetBaseCfg = PLANE_CFG
    light: AssetBaseCfg = LIGHT_CFG


@configclass
class CabinetDirectEnvCfg(DirectRLEnvCfg):
    """Base configuration for the direct-workflow cabinet task."""

    # environment and simulation
    episode_length_s = 8.0
    decimation: int = 1
    action_space = 8
    observation_space = 31
    state_space = 0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=1,
        physics=CabinetPhysicsCfg(),
        default_visualizer_cfg=VisualizerCfg(eye=(-2.0, 2.0, 2.0), lookat=(0.8, 0.0, 0.5)),
    )
    scene: CabinetDirectSceneCfg = CabinetDirectSceneCfg(num_envs=4096, env_spacing=2.0)
    events: EventCfg = EventCfg()

    # robot joints and frames -- set by a robot-specific subclass
    arm_joint_names: str | list[str] = MISSING
    finger_joint_names: str | list[str] = MISSING
    ee_body_name: str = MISSING
    left_finger_body_name: str = MISSING
    right_finger_body_name: str = MISSING
    ee_pos_offset: tuple[float, float, float] = MISSING
    """End-effector frame position offset [m]."""
    finger_pos_offset: tuple[float, float, float] = MISSING
    """Fingertip frame position offset [m]."""

    # action processing -- set by a robot-specific subclass
    arm_action_scale: float = 1.0
    """Arm joint position scale [m or rad, depending on joint type]."""
    gripper_open_command: float = MISSING
    """Open gripper joint position [m or rad, depending on joint type]."""
    gripper_close_command: float = MISSING
    """Closed gripper joint position [m or rad, depending on joint type]."""

    # cabinet joint and frame
    drawer_joint_name: str = "drawer_top_joint"
    drawer_handle_body_name: str = "drawer_handle_top"
    drawer_handle_pos_offset: tuple[float, float, float] = (0.305, 0.0, 0.01)
    """Drawer-handle frame position offset [m]."""
    drawer_handle_rot_offset: tuple[float, float, float, float] = (0.5, -0.5, -0.5, 0.5)
    """Drawer-handle frame orientation as an ``(x, y, z, w)`` quaternion."""

    # reward parameters, matching :class:`~isaaclab_tasks.core.cabinet.cabinet_env_cfg.RewardsCfg`
    approach_ee_handle_threshold: float = 0.2
    """End-effector distance threshold [m] for the near-handle bonus."""
    approach_gripper_handle_offset: float = MISSING
    """Fingertip reach offset [m]."""
    grasp_handle_threshold: float = 0.03
    """Maximum end-effector distance [m] for the grasp reward."""
    success_drawer_pos_threshold: float = 0.30
    """Drawer joint position [m] above which an episode is successful."""

    approach_ee_handle_reward_scale: float = 2.0
    align_ee_handle_reward_scale: float = 0.5
    approach_gripper_handle_reward_scale: float = 5.0
    align_grasp_around_handle_reward_scale: float = 0.125
    grasp_handle_reward_scale: float = 0.5
    open_drawer_reward_scale: float = 7.5
    multi_stage_open_drawer_reward_scale: float = 1.0
    action_rate_reward_scale: float = -1e-2
    joint_vel_reward_scale: float = -1e-4
