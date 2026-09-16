# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import config_field, replace_config

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.humanoid import HUMANOID_CFG


@dataclass
class HumanoidPhysicsCfg(PresetCfg):
    isaacsim_physx: PhysxCfg = config_field(PhysxCfg(bounce_threshold_velocity=0.2))
    ovphysx: OvPhysxCfg = config_field(OvPhysxCfg())
    physx: PhysxAutoCfg = config_field(PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx))
    newton_mjwarp: NewtonCfg = config_field(
        NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=80,
                nconmax=25,
                cone="pyramidal",
                update_data_interval=2,
                integrator="implicitfast",
                impratio=1,
            ),
            num_substeps=2,
            debug_mode=False,
        )
    )
    default: Any = config_field(newton_mjwarp)


@dataclass
class HumanoidEnvCfg(DirectRLEnvCfg):
    """Configuration for the direct-workflow Humanoid walking environment."""

    # env
    episode_length_s: Any = config_field(16.0)
    decimation: Any = config_field(2)
    action_scale: Any = config_field(1.0)
    action_space: Any = config_field(21)
    observation_space: Any = config_field(87)
    state_space: Any = config_field(0)

    # simulation
    sim: SimulationCfg = config_field(
        SimulationCfg(dt=1 / 120, render_interval=decimation, physics=HumanoidPhysicsCfg())
    )
    terrain: Any = config_field(
        TerrainImporterCfg(
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
    )

    # scene
    scene: InteractiveSceneCfg = config_field(
        InteractiveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True, clone_in_fabric=True)
    )

    # robot
    robot: ArticulationCfg = config_field(replace_config(HUMANOID_CFG, prim_path="{ENV_REGEX_NS}/Robot"))

    # effort scale per joint, keyed by joint name expression
    joint_gears: dict[str, float] = config_field(
        {
            ".*_waist.*": 67.5,
            ".*_upper_arm.*": 67.5,
            "pelvis": 67.5,
            ".*_lower_arm": 45.0,
            ".*_thigh:0": 45.0,
            ".*_thigh:1": 135.0,
            ".*_thigh:2": 45.0,
            ".*_shin": 90.0,
            ".*_foot.*": 22.5,
        }
    )

    # sensors
    joint_wrench: JointWrenchSensorCfg = config_field(JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot"))
    feet_body_names: list[str] = config_field(["left_foot", "right_foot"])

    # walk target, relative to the environment origin
    target_pos: tuple[float, float, float] = config_field((1000.0, 0.0, 0.0))

    # reset
    initial_joint_pos_range: tuple[float, float] = config_field((-0.2, 0.2))  # [rad]
    initial_joint_vel_range: tuple[float, float] = config_field((-0.1, 0.1))  # [rad/s]

    heading_weight: float = config_field(0.5)
    up_weight: float = config_field(0.1)

    energy_cost_scale: float = config_field(0.005)
    actions_cost_scale: float = config_field(0.01)
    alive_reward_scale: float = config_field(2.0)
    joint_pos_limits_cost_scale: float = config_field(0.25)
    joint_pos_limits_threshold: float = config_field(0.98)

    death_cost: float = config_field(-1.0)
    termination_height: float = config_field(0.8)

    # observation scales
    dof_vel_scale: float = config_field(0.1)
    angular_velocity_scale: float = config_field(0.25)
    contact_force_scale: float = config_field(0.01)
