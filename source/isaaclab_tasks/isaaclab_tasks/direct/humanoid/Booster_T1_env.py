# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_assets import BOOSTER_T1_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv


@configclass
class BoosterT1EnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 23
    observation_space = 79
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = BOOSTER_T1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # TODO Correct these values
    joint_gears: list = [
        50.0,   # AAHead_yaw (head - moderate)
        50.0,   # Head_pitch (head - moderate)
    
        40.0,   # Left_Shoulder_Pitch (arms - lower effort)
        40.0,   # Right_Shoulder_Pitch
    
        40.0,   # Left_Shoulder_Roll
        40.0,   # Right_Shoulder_Roll
    
        40.0,   # Left_Elbow_Pitch
        40.0,   # Right_Elbow_Pitch
    
        40.0,   # Left_Elbow_Yaw
        40.0,   # Right_Elbow_Yaw
    
        30.0,   # Waist (legs group, moderate effort)
    
        45.0,   # Left_Hip_Pitch (legs - high effort)
        45.0,   # Right_Hip_Pitch
    
        35.0,   # Left_Hip_Roll
        35.0,   # Right_Hip_Roll
    
        35.0,   # Left_Hip_Yaw
        35.0,   # Right_Hip_Yaw
    
        60.0,   # Left_Knee_Pitch (highest effort)
        60.0,   # Right_Knee_Pitch
    
        25.0,   # Left_Ankle_Pitch (feet - moderate)
        25.0,   # Right_Ankle_Pitch
    
        15.0,   # Left_Ankle_Roll (feet - lower effort)
        15.0,   # Right_Ankle_Roll
    ]


    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.1

    death_cost: float = -1.0
    termination_height: float = 0.8

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01


class BoosterT1Env(LocomotionEnv):
    cfg: BoosterT1EnvCfg

    def __init__(self, cfg: BoosterT1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
