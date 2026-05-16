# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda lifting a Newton cable via proxy-coupled MJWarp+VBD."""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg
from isaaclab_newton.sim.spawners.materials import NewtonCableMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from isaaclab_contrib.cable.cable_object_cfg import CableObjectCfg
from isaaclab_contrib.deformable.newton_manager_cfg import (
    CoupledNewtonCfg,
    NewtonModelCfg,
    ProxyCoupledMJWarpVBDSolverCfg,
)

from . import mdp
from .franka_soft_env_cfg import FrankaSoftEnvCfg, _FrankaSoftSceneCfg


@configclass
class _FrankaCableSceneCfg(_FrankaSoftSceneCfg):
    """Scene for the Franka cable lifting environment.

    Inherits ``robot``, ``ee_frame``, ``table``, ``ground``, ``sky_light`` and the
    ``__post_init__`` actuator tuning from :class:`_FrankaSoftSceneCfg`; replaces the
    volumetric ``deformable`` asset with a Newton cable.
    """

    deformable = None
    """Disable the volumetric deformable asset inherited from the soft scene
    (``InteractiveScene`` skips fields that are ``None``)."""

    cable: CableObjectCfg = CableObjectCfg(
        prim_path="/World/envs/env_.*/Cable",
        init_state=CableObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
        spawn=sim_utils.CableCfg(
            positions=[(i * 0.03, 0.0, 0.0) for i in range(10)],
            width=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            physics_material=NewtonCableMaterialCfg(
                stretch_stiffness=1.0e6,
                bend_stiffness=1.0e-3,
                stretch_damping=1.0e-3,
                bend_damping=1.0e-3,
                density=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )


@configclass
class CommandsCfg:
    """Cable goal pose sampled in the robot root frame."""

    cable_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.25, 0.25),
            pos_z=(0.25, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        goal_pose_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2), opacity=0.4),
                ),
            },
        ),
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
            params={"asset_cfg": SceneEntityCfg("cable"), "num_points": 20},
        )
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "cable_pose"})
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
        params={"std": 0.1, "asset_cfg": SceneEntityCfg("cable")},
        weight=5.0,
    )
    lifting_cable = RewTerm(
        func=mdp.object_lifted,
        params={"minimal_height": 0.04, "asset_cfg": SceneEntityCfg("cable")},
        weight=5.0,
    )
    cable_goal_tracking = RewTerm(
        func=mdp.object_com_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.075,
            "command_name": "cable_pose",
            "asset_cfg": SceneEntityCfg("cable"),
        },
        weight=16.0,
    )
    cable_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_com_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.075,
            "command_name": "cable_pose",
            "asset_cfg": SceneEntityCfg("cable"),
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

    cable_outside_table = DoneTerm(
        func=mdp.object_outside_table_bounds,
        params={
            "x_bounds": (0.0, 1.0),
            "y_bounds": (-0.5, 0.5),
            "asset_cfg": SceneEntityCfg("cable"),
        },
    )

    cable_dropped = DoneTerm(
        func=mdp.object_com_below_minimum,
        params={"minimum_height": -0.1, "asset_cfg": SceneEntityCfg("cable")},
    )

    ee_below_table = DoneTerm(
        func=mdp.ee_below_minimum,
        params={"minimum_height": 0.0, "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )


@configclass
class FrankaCableEnvCfg(FrankaSoftEnvCfg):
    """Franka Panda lifting a Newton cable via proxy-coupled MJWarp+VBD."""

    scene: _FrankaCableSceneCfg = _FrankaCableSceneCfg(num_envs=128, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Proxy-coupled MJWarp + VBD: rigid arm in MJWarp, cable particles in VBD, and the gripper
        # fingers exposed as virtual proxies so VBD detects them as contacts on the cable.
        self.sim.physics = CoupledNewtonCfg(
            scene_cfg=self.scene,
            solver_cfg=ProxyCoupledMJWarpVBDSolverCfg(
                mjwarp_cfg=MJWarpSolverCfg(
                    njmax=40,
                    nconmax=20,
                    ls_iterations=20,
                    integrator="implicitfast",
                    ccd_iterations=100,
                ),
                mjwarp_bodies=[SceneEntityCfg("robot")],
                vbd_bodies=[SceneEntityCfg("cable")],
                proxy_bodies=[
                    SceneEntityCfg("robot", body_names=["panda_hand", "panda_(left|right)finger"]),
                ],
            ),
            model_cfg=NewtonModelCfg(
                soft_contact_ke=1e4,
                soft_contact_kd=1e-5,
                soft_contact_mu=5.0,
                shape_material_ke=4e4,
                shape_material_kd=1e-5,
                shape_material_mu=5.0,
            ),
            num_substeps=10,
        )
