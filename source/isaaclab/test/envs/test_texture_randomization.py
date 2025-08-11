# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script tests the functionality of texture randomization applied to the cartpole scene.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch

import omni.usd
import pytest

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleSceneCfg


@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=5.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # on prestartup apply a new set of textures
    # note from @mayank: Changed from 'reset' to 'prestartup' to make test pass.
    #   The error happens otherwise on Kit thread which is not the main thread.
    cart_texture_randomizer = EventTerm(
        func=mdp.randomize_visual_texture_material,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["cart"]),
            "texture_paths": [
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry/Cherry_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber/Timber_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber_Cladding/Timber_Cladding_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
            ],
            "event_name": "cart_texture_randomizer",
            "texture_rotation": (math.pi / 2, math.pi / 2),
        },
    )

    # on reset apply a new set of textures
    pole_texture_randomizer = EventTerm(
        func=mdp.randomize_visual_texture_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
            "texture_paths": [
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry/Cherry_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber/Timber_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber_Cladding/Timber_Cladding_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
            ],
            "event_name": "pole_texture_randomizer",
            "texture_rotation": (math.pi / 2, math.pi / 2),
        },
    )

    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )


@configclass
class EventCfgFallback:
    """Configuration for events that tests the fallback mechanism."""

    # Test fallback when /visuals pattern doesn't match
    test_fallback_texture_randomizer = EventTerm(
        func=mdp.randomize_visual_texture_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["slider"]),
            "texture_paths": [
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry/Cherry_BaseColor.png",
            ],
            "event_name": "test_fallback_texture_randomizer",
            "texture_rotation": (0.0, 0.0),
        },
    )

    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene = CartpoleSceneCfg(env_spacing=2.5)

    # Basic settings
    actions = ActionsCfg()
    observations = ObservationsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_texture_randomization(device):
    """Test texture randomization for cartpole environment."""
    # Create a new stage
    omni.usd.get_context().new_stage()

    try:
        # Set the arguments
        env_cfg = CartpoleEnvCfg()
        env_cfg.scene.num_envs = 16
        env_cfg.scene.replicate_physics = False
        env_cfg.sim.device = device

        # Setup base environment
        env = ManagerBasedEnv(cfg=env_cfg)

        try:
            # Simulate physics
            with torch.inference_mode():
                for count in range(50):
                    # Reset every few steps to check nothing breaks
                    if count % 10 == 0:
                        env.reset()
                    # Sample random actions
                    joint_efforts = torch.randn_like(env.action_manager.action)
                    # Step the environment
                    env.step(joint_efforts)
        finally:
            env.close()
    finally:
        # Clean up stage
        omni.usd.get_context().close_stage()


def test_texture_randomization_failure_replicate_physics():
    """Test texture randomization failure when replicate physics is set to True."""
    # Create a new stage
    omni.usd.get_context().new_stage()

    try:
        # Set the arguments
        cfg_failure = CartpoleEnvCfg()
        cfg_failure.scene.num_envs = 16
        cfg_failure.scene.replicate_physics = True

        # Test that creating the environment raises RuntimeError
        with pytest.raises(RuntimeError):
            env = ManagerBasedEnv(cfg_failure)
            env.close()
    finally:
        # Clean up stage
        omni.usd.get_context().close_stage()
