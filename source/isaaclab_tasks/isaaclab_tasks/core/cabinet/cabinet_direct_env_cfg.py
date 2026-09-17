# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import REQUIRED

from isaaclab_tasks.core.cabinet.cabinet_env_cfg import (
    CABINET_CFG,
    LIGHT_CFG,
    PLANE_CFG,
    CabinetDecimationCfg,
    CabinetSimCfg,
    EventCfg,
)


@dataclass
class CabinetDirectSceneCfg(InteractiveSceneCfg):
    """Scene configuration shared by direct-workflow cabinet tasks."""

    robot: ArticulationCfg = REQUIRED
    cabinet: ArticulationCfg = field(default_factory=lambda: deepcopy(CABINET_CFG))
    plane: AssetBaseCfg = field(default_factory=lambda: deepcopy(PLANE_CFG))
    light: AssetBaseCfg = field(default_factory=lambda: deepcopy(LIGHT_CFG))


@dataclass
class CabinetDirectEnvCfg(DirectRLEnvCfg):
    """Base configuration for the direct-workflow cabinet task."""

    # environment and simulation
    episode_length_s: Any = 8.0
    decimation: int = field(default_factory=CabinetDecimationCfg)
    action_space: Any = 8
    observation_space: Any = 31
    state_space: Any = 0
    sim: CabinetSimCfg = field(default_factory=CabinetSimCfg)
    scene: CabinetDirectSceneCfg = field(default_factory=lambda: CabinetDirectSceneCfg(num_envs=4096, env_spacing=2.0))
    events: EventCfg = field(default_factory=EventCfg)

    # robot joints and frames -- set by a robot-specific subclass
    arm_joint_names: str | list[str] = REQUIRED
    finger_joint_names: str | list[str] = REQUIRED
    ee_body_name: str = REQUIRED
    left_finger_body_name: str = REQUIRED
    right_finger_body_name: str = REQUIRED
    ee_pos_offset: tuple[float, float, float] = REQUIRED
    """End-effector frame position offset [m]."""
    finger_pos_offset: tuple[float, float, float] = REQUIRED
    """Fingertip frame position offset [m]."""

    # action processing -- set by a robot-specific subclass
    arm_action_scale: float = 1.0
    """Arm joint position scale [m or rad, depending on joint type]."""
    gripper_open_command: float = REQUIRED
    """Open gripper joint position [m or rad, depending on joint type]."""
    gripper_close_command: float = REQUIRED
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
    approach_gripper_handle_offset: float = REQUIRED
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
