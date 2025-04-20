# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_tasks.direct.nao.nao import NAO_CFG  # Change import to NAO configuration

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv


@configclass
class NaoEnvCfg(DirectRLEnvCfg):  # Rename class to reflect NAO robot

    # def __post_init__(self):
    #     self.sim.physx.gpu_max_rigid_patch_count = 4096 * 4096

    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 24  # Update for NAO's DOFs (typically 25 joints)
    observation_space = 84  # Update based on NAO sensor data dimensions
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene - adjusted for NAO's smaller size
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = NAO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # NAO joint gears - adjusted for NAO's motors
    # The values below are estimates and should be calibrated based on actual NAO specifications
    joint_gears: list = [
        # Head joints
        1.0,  # HeadYaw
        1.0,  # HeadPitch
        
        # Left arm joints
        1.0,  # LShoulderPitch
        1.0,  # LShoulderRoll
        1.0,  # LElbowYaw
        1.0,  # LElbowRoll
        1.0,  # LWristYaw
        
        # Left hand joints (if included)
        # 5.0,   # LHand
        
        # Right arm joints
        1.0,  # RShoulderPitch
        1.0,  # RShoulderRoll
        1.0,  # RElbowYaw
        1.0,  # RElbowRoll
        1.0,  # RWristYaw
        
        # Right hand joints (if included)
        # 5.0,   # RHand
        
        # Left leg joints
        1.0,  # LHipYawPitch
        1.0,  # LHipRoll
        1.0, # LHipPitch
        1.0,  # LKneePitch
        1.0,  # LAnklePitch
        1.0,  # LAnkleRoll
        
        # Right leg joints
        1.0,  # RHipYawPitch
        1.0,  # RHipRoll
        1.0, # RHipPitch
        1.0,  # RKneePitch
        1.0,  # RAnklePitch
        1.0,  # RAnkleRoll
    ]

    # Reward weights adjusted for NAO
    heading_weight: float = 1.0
    up_weight: float = 0.1  # Increased slightly as balance is more critical for NAO

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0  # Increased to encourage stable postures
    dof_vel_scale: float = 0.1

    death_cost: float = -1.0
    termination_height: float = 0.25  # Reduced for NAO's smaller height (about 58cm tall)

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01  # Reduced for NAO's lighter weight


class NaoEnv(LocomotionEnv):  # Rename class
    cfg: NaoEnvCfg

    def __init__(self, cfg: NaoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
