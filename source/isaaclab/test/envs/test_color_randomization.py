# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script tests the functionality of color randomization applied to the cartpole scene.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math

import pytest
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.version import get_isaac_sim_version

from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import CartpoleSceneCfg


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

    # Color randomization terms; the mode is overridden per parametrization in
    # :func:`test_color_randomization` (both "prestartup" and "reset" are exercised).
    cart_color_randomizer = EventTerm(
        func=mdp.randomize_visual_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["cart"]),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "event_name": "cart_color_randomizer",
        },
    )

    pole_color_randomizer = EventTerm(
        func=mdp.randomize_visual_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "event_name": "pole_color_randomizer",
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
@pytest.mark.parametrize("mode", ["prestartup", "reset"])
def test_color_randomization(mode, device):
    """Test color randomization for the cartpole environment in both event modes.

    The randomizer is exercised in both ``prestartup`` and ``reset`` modes. The ``reset`` case is a
    regression guard: reset-mode color randomization on the articulation used to invalidate the
    PhysX articulation view and crash environment construction with an
    ``AttributeError: 'NoneType' object has no attribute 'link_count'``.
    """
    # skip test if stage in memory is not supported
    if get_isaac_sim_version().major < 5:
        pytest.skip("Color randomization test hangs in this version of Isaac Sim")

    # Create a new stage
    sim_utils.create_new_stage()

    try:
        # Set the arguments
        env_cfg = CartpoleEnvCfg()
        env_cfg.scene.num_envs = 16
        env_cfg.scene.replicate_physics = False
        env_cfg.sim.device = device
        # exercise the parametrized event mode for both color terms
        env_cfg.events.cart_color_randomizer.mode = mode
        env_cfg.events.pole_color_randomizer.mode = mode

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
        sim_utils.close_stage()
