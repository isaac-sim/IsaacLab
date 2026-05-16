# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka cable lifting environment.

Mirrors :mod:`franka_soft_env_cfg` but replaces the volumetric deformable
object with a single Newton cable and swaps the coupled solver for the
proxy-coupled variant (:class:`ProxyCoupledMJWarpVBDSolverCfg`). The RL task
is to lift the cable's midpoint to a randomised target position sampled in
the robot's root frame.
"""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg
from isaaclab_newton.sim.spawners.materials import NewtonCableMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_contrib.cable.cable_object_cfg import CableObjectCfg
from isaaclab_contrib.deformable.newton_manager_cfg import (
    CoupledNewtonCfg,
    NewtonModelCfg,
    ProxyCoupledMJWarpVBDSolverCfg,
    VBDSolverCfg,
)

from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort:skip


##
# Scene definition
##


@configclass
class _FrankaCableSceneCfg(InteractiveSceneCfg):
    """Scene for the Franka cable lifting environment."""

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # end-effector frame for reward shaping
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
        ],
    )

    # Newton cable: 10 control points along local +X (27 cm total length), 1 cm diameter.
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

    # static table matching the soft env: top surface at z = 0
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.0, 0.0, 0.707, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # ground plane
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    def __post_init__(self) -> None:
        # Disable gravity on the arm so the low-PD actuators don't fight gravity sag,
        # which is the dominant source of steady-state IK tracking error.
        self.robot.spawn.rigid_props.disable_gravity = True

        # Increase Franka gripper stiffness so it can grip the cable firmly.
        self.robot.actuators["panda_hand"].effort_limit_sim = 500.0
        self.robot.actuators["panda_hand"].stiffness = 1000.0
        self.robot.actuators["panda_hand"].damping = 100.0


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Commands for the cable goal pose (xyz + identity quat in robot root frame)."""

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
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.1, 0.9, 0.2),
                        opacity=0.4,
                    ),
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
            ik_params={"lambda_val": 0.6},
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
    """Policy observations: joint state, cable segments in robot frame, target, last action."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cable_sampled_points = ObsTerm(
            func=mdp.CableSampledPointsInRobotRootFrame,
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
    """Reset events: robot to default joint config. Cables have no nodal snap-back."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )


@configclass
class RewardsCfg:
    """Lift-to-target reward for the cable."""

    reaching_cable = RewTerm(
        func=mdp.cable_ee_distance,
        params={"std": 0.1, "asset_cfg": SceneEntityCfg("cable")},
        weight=5.0,
    )
    lifting_cable = RewTerm(
        func=mdp.cable_lifted,
        params={"minimal_height": 0.04, "asset_cfg": SceneEntityCfg("cable")},
        weight=5.0,
    )
    cable_goal_tracking = RewTerm(
        func=mdp.cable_com_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.075,
            "command_name": "cable_pose",
            "asset_cfg": SceneEntityCfg("cable"),
        },
        weight=16.0,
    )
    cable_goal_tracking_fine_grained = RewTerm(
        func=mdp.cable_com_goal_distance,
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
    """Time out + table bounds/drop termination."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cable_outside_table = DoneTerm(
        func=mdp.cable_outside_table_bounds,
        params={
            "x_bounds": (0.0, 1.0),
            "y_bounds": (-0.5, 0.5),
            "asset_cfg": SceneEntityCfg("cable"),
        },
    )

    cable_dropped = DoneTerm(
        func=mdp.cable_com_below_minimum,
        params={"minimum_height": -0.1, "asset_cfg": SceneEntityCfg("cable")},
    )

    ee_below_table = DoneTerm(
        func=mdp.ee_below_minimum,
        params={"minimum_height": 0.0, "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )


##
# Environment configuration
##


@configclass
class FrankaCableEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL environment: Franka Panda lifting a Newton cable via proxy-coupled MJWarp+VBD."""

    # Scene settings
    scene: _FrankaCableSceneCfg = _FrankaCableSceneCfg(num_envs=128, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        # general settings
        self.decimation = 1
        self.episode_length_s = 5.0

        # simulation settings
        self.sim.dt = 1 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, 0.0)
        self.sim.physics = CoupledNewtonCfg(
            scene_cfg=self.scene,
            solver_cfg=ProxyCoupledMJWarpVBDSolverCfg(
                mjwarp_cfg=MJWarpSolverCfg(
                    njmax=40,
                    nconmax=20,
                    ls_iterations=20,
                    cone="pyramidal",
                    impratio=1,
                    ls_parallel=False,
                    integrator="implicitfast",
                    ccd_iterations=100,
                ),
                vbd_cfg=VBDSolverCfg(
                    iterations=10,
                    integrate_with_external_rigid_solver=True,
                    particle_enable_self_contact=False,
                    particle_collision_detection_interval=-1,
                ),
                # Declare which scene entities belong to which sub-solver.
                # The manager grep-matches these prim-path templates against
                # `model.body_label` (the full USD prim path of each body) to
                # bucket bodies, joints, and shapes between the two entries.
                mjwarp_prim_paths=[self.scene.robot.prim_path],
                vbd_prim_paths=[self.scene.cable.prim_path],
                # Expose the Franka gripper bodies as proxies in the VBD view so the
                # cable feels gripper contacts and the gripper feels cable feedback.
                proxy_bodies=[
                    SceneEntityCfg("robot", body_names=["panda_hand", "panda_(left|right)finger"]),
                ],
                proxy_mode="lagged",
                proxy_iterations=1,
                proxy_collide_interval=1,
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
            use_cuda_graph=True,
        )

        # viewer settings
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (1.25, -1.5, 0.75)
        self.viewer.resolution = (1920, 1080)
