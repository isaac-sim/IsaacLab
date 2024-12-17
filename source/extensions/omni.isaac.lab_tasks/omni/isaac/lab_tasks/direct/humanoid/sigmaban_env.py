# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab_assets import HUMANOID_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.direct.locomotion.locomotion_env import LocomotionEnv
from omni.isaac.lab_assets import SIGMABAN_CFG


@configclass
class SigmabanEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 20
    observation_space = 72
    state_space = 0

    # simulation
    physx: PhysxCfg = PhysxCfg(gpu_max_rigid_patch_count=4096 * 4096)
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physx=physx)

    # Ground
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = SIGMABAN_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint_gears: list = [
        # 200.0,  # head_yaw
        # 200.0,  # left_shoulder_pitch
        # 200.0,  # right_shoulder_pitch
        # 200.0,  # left_hip_yaw
        # 200.0,  # right_hip_yaw
        # 200.0,  # head_pitch
        # 200.0,  # left_shoulder_roll
        # 200.0,  # right_shoulder_roll
        # 250.0,  # left_hip_roll
        # 250.0,  # right_hip_roll
        # 200.0,  # left_elbow
        # 200.0,  # right_elbow
        # 250.0,  # left_hip_pitch
        # 250.0,  # right_hip_pitch
        # 250.0,  # left_knee
        # 250.0,  # right_knee
        # 250.0,  # left_ankle_pitch
        # 250.0,  # right_ankle_pitch
        # 250.0,  # left_ankle_roll
        # 250.0,  # right_ankle_roll
        5.0,  # head_yaw
        5.0,  # left_shoulder_pitch
        5.0,  # right_shoulder_pitch
        5.0,  # left_hip_yaw
        5.0,  # right_hip_yaw
        5.0,  # head_pitch
        5.0,  # left_shoulder_roll
        5.0,  # right_shoulder_roll
        5.0,  # left_hip_roll
        5.0,  # right_hip_roll
        5.0,  # left_elbow
        5.0,  # right_elbow
        5.0,  # left_hip_pitch
        5.0,  # right_hip_pitch
        5.0,  # left_knee
        5.0,  # right_knee
        5.0,  # left_ankle_pitch
        5.0,  # right_ankle_pitch
        5.0,  # left_ankle_roll
        5.0,  # right_ankle_roll
    ]

    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.1

    death_cost: float = -1.0
    termination_height: float = 0.25

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01


class SigmabanEnv(LocomotionEnv):
    cfg: SigmabanEnvCfg

    def __init__(self, cfg: SigmabanEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
