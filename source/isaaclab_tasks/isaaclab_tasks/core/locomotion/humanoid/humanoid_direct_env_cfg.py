# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_newton.physics import FeatherPGSSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg, preset

from isaaclab_assets import HUMANOID_CFG


@configclass
class HumanoidPhysicsCfg(PresetCfg):
    default: PhysxCfg = PhysxCfg()
    physx: PhysxCfg = PhysxCfg()
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            use_mujoco_contacts=False,
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
    feather_pgs: NewtonCfg = NewtonCfg(
        solver_cfg=FeatherPGSSolverCfg(
            angular_damping=5.0,
            enable_joint_limits=True,
            pgs_iterations=24,
            pgs_beta=0.002,
            pgs_cfm=1.0e-4,
            pgs_omega=0.5,
            dense_max_constraints=64,
            mf_max_constraints=512,
        ),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=False,
    )
    ovphysx: OvPhysxCfg = OvPhysxCfg()


@configclass
class FeatherPGSEventCfg:
    robot_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (2.0, 2.0),
            "operation": "abs",
            "recompute_inertia": False,
        },
    )
    robot_body_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "inertia_distribution_params": (2.0, 2.0),
            "operation": "add",
            "diagonal_only": True,
        },
    )


@configclass
class HumanoidDirectEventCfg(PresetCfg):
    default = None
    feather_pgs = FeatherPGSEventCfg()


@configclass
class HumanoidEnvCfg(DirectRLEnvCfg):
    """Configuration for the direct-workflow Humanoid walking environment."""

    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 21
    observation_space = 75
    state_space = 0
    events: HumanoidDirectEventCfg = HumanoidDirectEventCfg()

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=HumanoidPhysicsCfg())
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot.actuators["body"].armature = preset(default=robot.actuators["body"].armature, feather_pgs=0.2)

    physx_joint_gears: list[float] = [
        67.5000,  # lower_waist
        67.5000,  # lower_waist
        67.5000,  # right_upper_arm
        67.5000,  # right_upper_arm
        67.5000,  # left_upper_arm
        67.5000,  # left_upper_arm
        67.5000,  # pelvis
        45.0000,  # right_lower_arm
        45.0000,  # left_lower_arm
        45.0000,  # right_thigh: x
        135.0000,  # right_thigh: y
        45.0000,  # right_thigh: z
        45.0000,  # left_thigh: x
        135.0000,  # left_thigh: y
        45.0000,  # left_thigh: z
        90.0000,  # right_knee
        90.0000,  # left_knee
        22.5,  # right_foot
        22.5,  # right_foot
        22.5,  # left_foot
        22.5,  # left_foot
    ]
    newton_joint_gears: list[float] = [
        67.5000,  # left_upper_arm
        67.5000,  # left_upper_arm
        45.0000,  # left_lower_arm
        67.5000,  # lower_waist
        67.5000,  # lower_waist
        67.5000,  # pelvis
        45.0000,  # left_thigh: x
        135.0000,  # left_thigh: y
        45.0000,  # left_thigh: z
        90.0000,  # left_knee
        22.5,  # left_foot
        22.5,  # left_foot
        45.0000,  # right_thigh: x
        135.0000,  # right_thigh: y
        45.0000,  # right_thigh: z
        90.0000,  # right_knee
        22.5,  # right_foot
        22.5,  # right_foot
        67.5000,  # right_upper_arm
        67.5000,  # right_upper_arm
        45.0000,  # right_lower_arm
    ]
    joint_gears = {
        "physx": physx_joint_gears,
        "newton": newton_joint_gears,
    }

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
