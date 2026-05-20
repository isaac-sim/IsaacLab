# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka surface deformable lifting environment."""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.deformable.newton_manager_cfg import CoupledMJWarpVBDSolverCfg, NewtonModelCfg, VBDSolverCfg

from isaaclab_tasks.utils import PresetCfg

from . import mdp
from .franka_soft_env_cfg import DeformableNewtonCfg, FrankaSoftEnvCfg, _FrankaSoftSceneCfg
from .franka_soft_env_cfg import EventCfg as FrankaSoftEventCfg

##
# Scene definition
##

ROBOT_SHAPE_MATERIAL_MU = 100.0
"""Franka collision-shape friction coefficient [dimensionless] used for Newton cloth contact."""

ROBOT_SHAPE_MATERIAL_BODY_NAMES = ".*"
"""Franka body-name regex receiving :data:`ROBOT_SHAPE_MATERIAL_MU`."""


@configclass
class PhysicsCfg(PresetCfg):
    # Newton physics: MJWarp rigid + VBD soft, one-way coupled
    # (matches newton/examples/softbody/example_softbody_franka.py)
    newton_mjwarp_vdb: DeformableNewtonCfg = DeformableNewtonCfg(
        solver_cfg=CoupledMJWarpVBDSolverCfg(
            rigid_solver_cfg=MJWarpSolverCfg(
                njmax=40,
                nconmax=20,
                ls_iterations=20,
                cone="pyramidal",
                impratio=1,
                ls_parallel=False,
                integrator="implicitfast",
                ccd_iterations=100,
            ),
            soft_solver_cfg=VBDSolverCfg(
                iterations=10,
                integrate_with_external_rigid_solver=True,
                particle_enable_self_contact=False,
                particle_collision_detection_interval=-1,
            ),
            coupling_mode="two_way",
        ),
        model_cfg=NewtonModelCfg(
            soft_contact_ke=1e3,
            soft_contact_kd=1e-5,
            soft_contact_mu=0.5,
            shape_material_ke=1e3,
            shape_material_kd=1e-5,
            shape_material_mu=1e-4,
        ),
        num_substeps=10,
        use_cuda_graph=True,
    )

    default = newton_mjwarp_vdb


@configclass
class DeformableCfg(PresetCfg):
    """Preset config for the deformable object, matching the Newton example."""

    newton_mjwarp_vdb: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.2)),
        spawn=sim_utils.MeshRectangleCfg(
            size=(0.2, 0.2),
            resolution=(30, 30),
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=50.0,
                particle_radius=0.005,
                tri_ke=5e2,
                tri_ka=5e2,
                tri_kd=1e-3,
                edge_ke=2.0,
                edge_kd=1e-3,
            ),
        ),
    )

    default = newton_mjwarp_vdb


@configclass
class FrankaClothSceneCfg(_FrankaSoftSceneCfg):
    """Scene for the Franka surface deformable environment."""

    deformable: DeformableCfg = DeformableCfg()

    # static collidable cubes the cloth drops onto (sits on the table top at z = 0).
    # Modeled as a static asset (no rigid body / no DOFs) so adding it does not
    # extend the Newton model's joint state.
    cube: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0.0, 0.04)),
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.01, 0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.25)),
        ),
    )


@configclass
class ActionsCfg:
    """7-dim arm joint position + 1-dim binary gripper."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], scale=0.1, use_default_offset=True
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.05},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class RewardsCfg:
    """Lift-to-target reward for a deformable object."""

    reaching_deformable = RewTerm(
        func=mdp.deformable_ee_distance,
        params={"std": 0.1, "asset_cfg": SceneEntityCfg("deformable")},
        weight=5.0,
    )
    lifting_deformable = RewTerm(
        func=mdp.deformable_lifted,
        params={"minimal_height": 0.04, "asset_cfg": SceneEntityCfg("deformable")},
        weight=5.0,
    )
    deformable_goal_tracking = RewTerm(
        func=mdp.deformable_com_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.075,
            "command_name": "deformable_pose",
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=16.0,
    )
    deformable_goal_tracking_fine_grained = RewTerm(
        func=mdp.deformable_com_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.075,
            "command_name": "deformable_pose",
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    gripper_close = RewTerm(
        func=mdp.gripper_close_action,
        params={"action_name": "gripper_action"},
        weight=-1.0,
    )
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-2)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-4)


@configclass
class EventCfg(FrankaSoftEventCfg):
    """Reset and startup events for the Franka cloth environment."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_SHAPE_MATERIAL_BODY_NAMES),
            "static_friction_range": (ROBOT_SHAPE_MATERIAL_MU, ROBOT_SHAPE_MATERIAL_MU),
            "dynamic_friction_range": (ROBOT_SHAPE_MATERIAL_MU, ROBOT_SHAPE_MATERIAL_MU),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )


##
# Environment configuration
##


@configclass
class FrankaClothEnvCfg(FrankaSoftEnvCfg):
    """Manager-based RL environment: Franka Panda lifting a surface deformable."""

    # Scene settings
    scene: FrankaClothSceneCfg = FrankaClothSceneCfg(num_envs=128, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        # general settings
        self.decimation = 1
        self.episode_length_s = 5.0

        # simulation settings
        self.sim.dt = 1 / 60.0
        self.sim.render_interval = self.decimation

        # viewer settings
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (1.25, -1.5, 0.6)
        self.viewer.resolution = (1920, 1080)
        self.sim.physics = PhysicsCfg()

        # increase franka gripper stiffness
        self.scene.robot.actuators["panda_hand"].effort_limit_sim = 500.0
        self.scene.robot.actuators["panda_hand"].stiffness = 2000.0
        self.scene.robot.actuators["panda_hand"].damping = 100.0
