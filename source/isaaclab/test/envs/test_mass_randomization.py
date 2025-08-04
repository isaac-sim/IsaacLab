# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the environment concept that combines a scene with an action,
observation and event manager for a quadruped robot.

A locomotion policy is loaded and used to control the robot. This shows how to use the
environment with a policy.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher, run_tests

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the concept of an Environment.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
import unittest

import omni.usd

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip


##
# Scene definition
##
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    # add terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=ROUGH_TERRAINS_CFG,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # add robot
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=900.0,
            texture_file=f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr",
            visible_in_primary_ray=False,
        ),
    )


##
# MDP settings
##
def constant_commands(env: ManagerBasedEnv) -> torch.Tensor:
    """The generated command from the command generator."""
    return torch.tensor([[1, 0, 0]], device=env.device).repeat(env.num_envs, 1)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_pos_z = ObsTerm(
            func=mdp.base_pos_z,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene_to_default = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class EventRandomizedCfg:
    """Configuration for events."""

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    reset_scene_to_default = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


##
# Environment configuration
##


@configclass
class QuadrupedEnvCfg(ManagerBasedEnvCfg):
    randomize_mass: bool = False
    """Configuration for the locomotion velocity-tracking environment."""
    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=3, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg | None = None

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.events = EventRandomizedCfg() if self.randomize_mass else EventCfg()
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005


class TestMassRandomization(unittest.TestCase):
    def test_randomized_mass(self):
        omni.usd.get_context().new_stage()
        env = ManagerBasedEnv(cfg=QuadrupedEnvCfg(randomize_mass=True))
        obs, _ = env.reset()
        actions = torch.rand((3, 12), device=env.device)  # 12 joints in ANMAL-C
        # reset
        for i in range(10):
            obs, _ = env.step(actions)
        # reset counters
        masses = NewtonManager._model.body_mass.numpy()
        masses_mjwarp = NewtonManager._solver.mjw_model.body_mass.numpy()
        masses = masses.reshape((3, -1))
        masses_mjwarp = masses_mjwarp.reshape((3, -1))
        # print("[INFO]: Environment reset. Observations:\n", obs)
        # 1 index is for body base in mjwarp
        # 0 index is for body base in newton
        self.assertTrue((np.abs(masses[0, 0] - masses[1, 0]) > 0.0).all())
        self.assertTrue((np.abs(masses[1, 0] - masses[2, 0]) > 0.0).all())
        self.assertTrue((np.abs(masses_mjwarp[0, 1] - masses_mjwarp[1, 1]) > 0.0).all())
        self.assertTrue((np.abs(masses_mjwarp[1, 1] - masses_mjwarp[2, 1]) > 0.0).all())
        # environment should have different heights
        heights = obs["policy"].cpu().numpy()
        self.assertTrue(np.abs(heights[0] - heights[1]) > 0.0)
        self.assertTrue(np.abs(heights[1] - heights[2]) > 0.0)
        env.close()

    def test_constant_mass(self):
        omni.usd.get_context().new_stage()
        env = ManagerBasedEnv(cfg=QuadrupedEnvCfg(randomize_mass=False))
        obs, _ = env.reset()
        actions = torch.rand((3, 12), device=env.device)  # 12 joints in ANMAL-C
        # reset
        for i in range(10):
            obs, _ = env.step(actions)
        masses = NewtonManager._model.body_mass.numpy()
        masses_mjwarp = NewtonManager._solver.mjw_model.body_mass.numpy()
        masses = masses.reshape((3, -1))
        masses_mjwarp = masses_mjwarp.reshape((3, -1))
        # print("[INFO]: Environment reset. Observations:\n", obs)
        self.assertTrue((np.abs(masses[0, :] - masses[1, :]) == 0.0).all())
        self.assertTrue((np.abs(masses[1, :] - masses[2, :]) == 0.0).all())
        self.assertTrue((np.abs(masses_mjwarp[0, :] - masses_mjwarp[1, :]) == 0.0).all())
        self.assertTrue((np.abs(masses_mjwarp[1, :] - masses_mjwarp[2, :]) == 0.0).all())
        # environment should have different heights
        # heights = obs["policy"].cpu().numpy()
        # self.assertTrue(np.abs(heights[0]- heights[1]) < 1e-2)
        # self.assertTrue(np.abs(heights[1]- heights[2]) < 1e-2)
        env.close()


if __name__ == "__main__":
    run_tests()
