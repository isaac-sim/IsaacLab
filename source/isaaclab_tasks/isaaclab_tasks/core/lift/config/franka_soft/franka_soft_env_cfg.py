# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka deformable (soft beam) lifting environment."""

from __future__ import annotations

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_contrib.coupling import (
    CouplerEntryCfg,
    CouplerProxyCfg,
    CouplerProxyMappingCfg,
)

from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from ... import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.franka import FRANKA_PANDA_MENAGERIE_CFG  # isort:skip


##
# Helpers
##


# Shared volume material parameters. The Newton config below uses the equivalent Lame parameters.
YOUNGS_MODULUS = 2e5
POISSONS_RATIO = 0.3

# Table collider whose top surface sits at z = 0. Spawned invisible: the command term's success
# visualizer draws it instead, tinted by whether the goal is reached.
TABLE_SPAWN_CFG = sim_utils.CuboidCfg(
    size=(1.3, 0.9, 1.05),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visible=False,
)


FRANKA_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Camera",
    offset=CameraCfg.OffsetCfg(
        pos=(0.85, -0.55, 0.42),
        rot=(0.5080, 0.2114, 0.318, 0.7720),
        convention="opengl",
    ),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 3.0)),
    width=128,
    height=128,
    renderer_cfg=MultiBackendRendererCfg(),
)


@configclass
class DeformableCfg(PresetCfg):
    """Preset config for the deformable object, matching the Newton example."""

    newton_mjwarp_vbd_proxy: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.3, 0.04, 0.04),
            edge_refinement=3.0,
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.85)),
            physics_material=NewtonDeformableBodyMaterialCfg(
                density=1000.0,
                k_mu=YOUNGS_MODULUS / (2.0 * (1.0 + POISSONS_RATIO)),
                k_lambda=(YOUNGS_MODULUS * POISSONS_RATIO / ((1.0 + POISSONS_RATIO) * (1.0 - 2.0 * POISSONS_RATIO))),
                particle_radius=0.0025,
            ),
        ),
    )

    physx: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.3, 0.04, 0.04),
            edge_refinement=3.0,
            deformable_props=PhysxDeformableBodyPropertiesCfg(),
            collision_props=[PhysxCollisionCfg(rest_offset=0.0025, contact_offset=0.01)],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.85)),
            physics_material=PhysxDeformableBodyMaterialCfg(
                density=1000.0,
                youngs_modulus=YOUNGS_MODULUS,
                poissons_ratio=POISSONS_RATIO,
                static_friction=10.0,
                dynamic_friction=10.0,
            ),
        ),
    )
    isaacsim_physx = physx

    default = newton_mjwarp_vbd_proxy


@configclass
class PhysicsCfg(PresetCfg):
    newton_mjwarp_vbd_proxy: NewtonCfg = NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(
                        cone="elliptic",
                        ls_iterations=20,
                        integrator="implicitfast",
                    ),
                    bodies=[r"/World/envs/env_[^/]+/Robot"],
                ),
                CouplerEntryCfg(
                    name="soft",
                    solver_cfg=VBDSolverCfg(iterations=10, rigid_body_particle_contact_buffer_size=256),
                    all_particles=True,
                    include_static_shapes=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="soft",
                    bodies=[
                        r"/World/envs/env_[^/]+/Robot/Geometry/.*panda_hand",
                        r"/World/envs/env_[^/]+/Robot/Geometry/.*panda_(left|right)finger",
                    ],
                    collide_interval=1,
                    collision_pipeline=NewtonCollisionPipelineCfg(
                        enable_rigid_soft_full_surface_contact=True,
                    ),
                )
            ],
            iterations=1,
        ),
        soft_contact_cfg=NewtonSoftContactCfg(
            soft_contact_ke=8.0e3,
            soft_contact_kd=1.0e-2,
            soft_contact_mu=10.0,
        ),
        num_substeps=2,
    )

    isaacsim_physx: PhysxCfg = PhysxCfg()

    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)

    default = newton_mjwarp_vbd_proxy


##
# Scene definition
##


@configclass
class _FrankaSoftSceneCfg(InteractiveSceneCfg):
    """Scene for the Franka deformable environment."""

    robot: ArticulationCfg = FRANKA_PANDA_MENAGERIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # end-effector frame for reward shaping
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=(
                    "{ENV_REGEX_NS}/Robot/Geometry/panda_link0/panda_link1/panda_link2/panda_link3/"
                    "panda_link4/panda_link5/panda_link6/panda_link7/panda_hand"
                ),
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
        ],
    )

    deformable: DeformableCfg = DeformableCfg()

    # static table collider with its top surface at z = 0. Kept invisible: the success
    # visualizer renders the visible table, colored by whether the goal is reached
    # (see CommandsCfg).
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, -0.525]),
        spawn=TABLE_SPAWN_CFG,
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
        self.robot.actuators = {
            # inspired by libfranka's joint_impedance_control.cpp
            "panda_arm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-7]"],
                effort_limit_sim={"panda_joint[1-4]": 87.0, "panda_joint[5-7]": 12.0},
                velocity_limit_sim={"panda_joint[1-4]": 2.175, "panda_joint[5-7]": 2.61},
                stiffness={
                    "panda_joint[1-4]": 600.0,
                    "panda_joint5": 250.0,
                    "panda_joint6": 150.0,
                    "panda_joint7": 50.0,
                },
                damping={
                    "panda_joint[1-4]": 50.0,
                    "panda_joint5": 30.0,
                    "panda_joint6": 25.0,
                    "panda_joint7": 15.0,
                },
                armature={
                    "panda_joint[1-2]": 0.6057,
                    "panda_joint[3-4]": 0.4625,
                    "panda_joint[5-7]": 0.2055,
                },
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint1"],
                effort_limit_sim=70.0,
                velocity_limit=0.2,
                velocity_limit_sim=2.0,
                stiffness=350.0,
                damping=175.0,
                armature=0.1,
            ),
            "panda_finger2_passive": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint2"],
                effort_limit_sim=1.0,
                velocity_limit=0.2,
                velocity_limit_sim=2.0,
                stiffness=0.0,
                damping=0.0,
                armature=0.1,
            ),
        }


@configclass
class _FrankaSoftCameraSceneCfg(_FrankaSoftSceneCfg):
    """Franka soft scene with a base camera."""

    base_camera: CameraCfg = FRANKA_CAMERA_CFG


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Commands for the deformable goal pose (xyz + identity quat in robot root frame)."""

    deformable_pose = mdp.DeformableUniformPoseCommandCfg(
        asset_name="robot",
        object_name="deformable",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.DeformableUniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.25, 0.25),
            pos_z=(0.25, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        # the invisible table is drawn by these markers, tinted green once the goal is reached
        success_vis_asset_name="table",
        success_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/SuccessMarkers",
            markers={
                "failure": TABLE_SPAWN_CFG.replace(
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.5, 0.5)), visible=True
                ),
                "success": TABLE_SPAWN_CFG.replace(
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.8, 0.5)), visible=True
                ),
            },
        ),
    )


@configclass
class _JointActionsCfg:
    """7-dim relative joint-position arm targets + 1-dim limit-rescaled gripper."""

    arm_action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=["panda_joint.*"], scale=0.03)

    gripper_action = mdp.JointPositionToLimitsActionCfg(
        asset_name="robot", joint_names=["panda_finger_joint1"], rescale_to_limits=True
    )


@configclass
class _IkActionsCfg:
    """7-dim absolute end-effector pose (xyz + quaternion) via differential IK + 1-dim binary gripper."""

    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.6},
        ),
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint1"],
        open_command_expr={"panda_finger_joint1": 0.04},
        close_command_expr={"panda_finger_joint1": 0.01},
    )


@configclass
class ActionsCfg(PresetCfg):
    """Action-space presets: joint-space for RL, task-space IK for scripted end-effector control."""

    joint: _JointActionsCfg = _JointActionsCfg()

    ik: _IkActionsCfg = _IkActionsCfg()

    default = joint


@configclass
class ObservationsCfg:
    """Policy observations: relative joint state, sampled deformable points, target command, and last action."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        deformable_sampled_points = ObsTerm(
            func=mdp.DeformableSampledPointsInRobotRootFrame,
            params={"asset_cfg": SceneEntityCfg("deformable"), "num_points": 20},
        )
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "deformable_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaCameraObservationsCfg:
    """Observation groups for visual deformable lifting."""

    @configclass
    class PolicyCfg(ObsGroup):
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "deformable_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ProprioCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PerceptionCfg(ObsGroup):
        deformable_sampled_points = ObsTerm(
            func=mdp.DeformableSampledPointsInRobotRootFrame,
            params={"asset_cfg": SceneEntityCfg("deformable"), "num_points": 20},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class BaseImageCfg(ObsGroup):
        image = ObsTerm(
            func=env_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("base_camera"),
                "data_type": "rgb",
                "normalize": True,
                "permute": True,
            },
        )

    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioCfg = ProprioCfg()
    perception: PerceptionCfg = PerceptionCfg()
    base_image: BaseImageCfg = BaseImageCfg()


@configclass
class EventCfg:
    """Reset events: robot to default joint config, deformable with small position randomization."""

    reset_robot_arm_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint.*"),
        },
    )

    reset_robot_gripper_joints = EventTerm(
        func=mdp.reset_joints_shared_offset,
        mode="reset",
        params={
            "position_range": (-0.02, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*"),
        },
    )

    reset_deformable = EventTerm(
        func=mdp.reset_nodal_state_uniform,
        mode="reset",
        params={
            "position_range": {"x": (-0.15, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("deformable"),
        },
    )

    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, -9.81], [0.0, 0.0, -9.81]),
            "operation": "abs",
        },
    )


@configclass
class RewardsCfg:
    """Lift-to-target reward for a deformable object."""

    reaching_deformable = RewTerm(
        func=mdp.deformable_com_ee_distance,
        params={"std": 0.1, "asset_cfg": SceneEntityCfg("deformable")},
        weight=5.0,
    )

    lifting_deformable = RewTerm(
        func=mdp.deformable_lifting,
        params={"std": 0.1, "minimal_height": 0.02, "asset_cfg": SceneEntityCfg("deformable")},
        weight=5.0,
    )

    deformable_goal_tracking = RewTerm(
        func=mdp.DeformableComGoalDistance,
        params={
            "std": 0.3,
            "minimal_height": 0.0,
            "command_name": "deformable_pose",
            "success_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=2.0,
    )

    success_bonus = RewTerm(
        func=mdp.deformable_com_goal_reached,
        params={
            "minimal_height": 0.0,
            "command_name": "deformable_pose",
            "success_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=20.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)


@configclass
class CurriculumCfg:
    """Ramp the action-rate penalty once the policy has learned to lift (matches rigid recipe)."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 15000}
    )

    # Since we use 24 steps per env, 10000 steps correspond to 10000/24 = 416.67 learning iterations
    gravity = CurrTerm(
        func=mdp.gravity_range_linear,
        params={
            "event_name": "variable_gravity",
            "start_gravity_z": -0.0001,
            "end_gravity_z": -9.81,
            "start_step": 0,
            "end_step": 10000,
        },
    )


@configclass
class TerminationsCfg:
    """Time out + workspace bounds termination."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    deformable_out_of_bounds = DoneTerm(
        func=mdp.deformable_outside_bounds,
        params={
            "x_bounds": (0.0, 1.0),
            "y_bounds": (-0.5, 0.5),
            "z_bounds": (-0.02, 1.0),
            "asset_cfg": SceneEntityCfg("deformable"),
        },
    )

    ee_below_table = DoneTerm(
        func=mdp.ee_below_minimum,
        params={"minimum_height": 0.0, "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )

    joint_vel_out_of_limit = DoneTerm(
        func=mdp.joint_vel_out_of_sim_limit,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


##
# Environment configuration
##


@configclass
class FrankaSoftSceneCfg(PresetCfg):
    newton_mjwarp_vbd_proxy: _FrankaSoftSceneCfg = _FrankaSoftSceneCfg(
        num_envs=2048, env_spacing=2.0, replicate_physics=True
    )

    # PhysX does not support replicating physics for deformable objects
    physx: _FrankaSoftSceneCfg = _FrankaSoftSceneCfg(num_envs=2048, env_spacing=2.0, replicate_physics=False)
    isaacsim_physx = physx

    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaSoftCameraSceneCfg(PresetCfg):
    """Scene presets for visual Franka soft lifting."""

    newton_mjwarp_vbd_proxy: _FrankaSoftCameraSceneCfg = _FrankaSoftCameraSceneCfg(
        num_envs=128, env_spacing=2.0, replicate_physics=True
    )
    physx: _FrankaSoftCameraSceneCfg = _FrankaSoftCameraSceneCfg(num_envs=128, env_spacing=2.0, replicate_physics=False)
    isaacsim_physx = physx
    default = newton_mjwarp_vbd_proxy


@configclass
class _FrankaSoftVisualizerCfg(VisualizerCfg):
    window_width: int = 1920
    window_height: int = 1080


@configclass
class FrankaSoftEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL environment: Franka Panda lifting a soft beam to a target pose."""

    # Scene settings
    scene: FrankaSoftSceneCfg = FrankaSoftSceneCfg()
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # general settings
        self.decimation = 4
        self.episode_length_s = 5.0

        # simulation settings
        self.sim.dt = 1.0 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()

        self.viewer.eye = (0.75, 0.25, 0.65)
        self.viewer.lookat = (0.0, 0.75, 0.4)
        self.sim.default_visualizer_cfg = _FrankaSoftVisualizerCfg(
            eye=self.viewer.eye,
            lookat=self.viewer.lookat,
        )

    def play_mode(self):
        super().play_mode()
        if self.curriculum is not None:
            self.curriculum.gravity = None


@configclass
class FrankaSoftCameraEnvCfg(FrankaSoftEnvCfg):
    """Visual Franka volume-deformable lifting environment."""

    scene: FrankaSoftCameraSceneCfg = FrankaSoftCameraSceneCfg()
    observations: FrankaCameraObservationsCfg = FrankaCameraObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Warm up the RTX render product/annotator (Newton skips the PhysX assets_loading render loop).
        self.num_rerenders_on_reset = 2
