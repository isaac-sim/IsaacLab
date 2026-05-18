# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda lifting a Newton cable via proxy-coupled MJWarp+VBD."""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg
from isaaclab_newton.sim.spawners.materials import NewtonCableMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from isaaclab_contrib.cable.cable_object_cfg import CableObjectCfg
from isaaclab_contrib.deformable.newton_manager_cfg import (
    VBDSolverCfg,
    CoupledNewtonCfg,
    NewtonModelCfg,
    ProxyCoupledMJWarpVBDSolverCfg,
)

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip

from . import mdp
from .franka_soft_env_cfg import FrankaSoftEnvCfg, _FrankaSoftSceneCfg


@configclass
class _FrankaCableSceneCfg(_FrankaSoftSceneCfg):
    """Scene for the Franka cable lifting environment.

    Inherits ``robot``, ``ee_frame``, ``table``, ``ground``, ``sky_light`` and the
    ``__post_init__`` actuator tuning from :class:`_FrankaSoftSceneCfg`; replaces the
    volumetric ``object`` asset with a Newton cable.
    """

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    object: CableObjectCfg = CableObjectCfg(
        prim_path="/World/envs/env_.*/Cable",
        init_state=CableObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.1)),
        spawn=sim_utils.CableCfg(
            positions=[(i * 0.02, 0.0, 0.0) for i in range(20)],
            width=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            physics_material=NewtonCableMaterialCfg(
                stretch_stiffness=1.0e6,
                stretch_damping=1.0e-1,
                bend_stiffness=5.0e-3,
                bend_damping=2.0e-3,
                density=100.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.rigid_props.disable_gravity = True
        self.robot.spawn.rigid_props = sim_utils.MujocoRigidBodyPropertiesCfg(gravcomp=1.0)
        
        # increase franka gripper stiffness
        self.robot.actuators["panda_hand"].effort_limit_sim = 1500.0
        self.robot.actuators["panda_hand"].stiffness = 1000.0
        self.robot.actuators["panda_hand"].damping = 100.0


@configclass
class CommandsCfg:
    """Cable goal pose sampled in the robot root frame."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.25, 0.25),
            pos_z=(0.1, 0.3),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        goal_pose_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.02,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2), opacity=0.01),
                ),
            },
        ),
        # Hide the EE frame
        current_pose_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/Command/body_pose",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=1e-6,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), opacity=0.0),
                ),
            },
        ),
    )


@configclass
class ActionsCfg:
    """7-dim absolute end-effector pose (xyz + quaternion) via differential IK + 1-dim binary gripper."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.05},
        ),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.05},
        close_command_expr={"panda_finger_.*": 0.0},
    )

@configclass
class ObservationsCfg:
    """Policy observations for the cable lifting task."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cable_sampled_points = ObsTerm(
            func=mdp.ObjectSampledPointsInRobotRootFrame,
            params={"asset_cfg": SceneEntityCfg("object"), "num_points": 20},
        )
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events for the cable lifting task."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )


@configclass
class RewardsCfg:
    """Lift-to-target reward for the cable."""

    reaching_cable = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "asset_cfg": SceneEntityCfg("object")},
        weight=5.0,
    )
    lifting_cable = RewTerm(
        func=mdp.object_lifted,
        params={"minimal_height": 0.04, "asset_cfg": SceneEntityCfg("object")},
        weight=5.0,
    )
    cable_goal_tracking = RewTerm(
        func=mdp.object_com_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.075,
            "command_name": "object_pose",
            "asset_cfg": SceneEntityCfg("object"),
        },
        weight=16.0,
    )
    cable_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_com_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.075,
            "command_name": "object_pose",
            "asset_cfg": SceneEntityCfg("object"),
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
class TerminationsCfg:
    """Time out and out-of-bounds terminations."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class FrankaCableEnvCfg(FrankaSoftEnvCfg):
    """Franka Panda lifting a Newton cable via proxy-coupled MJWarp+VBD."""

    scene: _FrankaCableSceneCfg = _FrankaCableSceneCfg(num_envs=128, env_spacing=2.5, replicate_physics=True)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # general settings
        self.decimation = 1
        self.episode_length_s = 2.0

        # simulation settings
        self.sim.dt = 1 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.81)


        # Proxy-coupled MJWarp + VBD: rigid arm in MJWarp, cable particles in VBD, and the gripper
        # fingers exposed as virtual proxies so VBD detects them as contacts on the cable.
        self.sim.physics = CoupledNewtonCfg(
            scene_cfg=self.scene,
            solver_cfg=ProxyCoupledMJWarpVBDSolverCfg(
                mjwarp_cfg=MJWarpSolverCfg(
                    cone="elliptic",
                    ls_parallel=True,
                    ls_iterations=20,
                    integrator="implicitfast",
                ),
                vbd_cfg=VBDSolverCfg(
                    iterations=20,
                    rigid_avbd_beta=1e2
                ),
                mjwarp_bodies=[SceneEntityCfg("robot")],
                vbd_bodies=[SceneEntityCfg("object")],
                proxy_bodies=[
                    SceneEntityCfg("robot", body_names=["panda_hand", "panda_(left|right)finger"]),
                ],
                proxy_collide_interval=5,
            ),
            model_cfg=NewtonModelCfg(
                shape_material_ke=1e4,
                shape_material_kd=1e-5,
                shape_material_mu=1.0,
            ),
            num_substeps=5,
        )
