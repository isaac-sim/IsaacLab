# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

# from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
ISAACLAB_NUCLEUS_DIR = "source/isaaclab_assets/data"
objects_dir = f"{ISAACLAB_NUCLEUS_DIR}/Props/Dextrah/Objects"
sub_dirs = sorted(os.listdir(objects_dir))
dirs = [object_name for object_name in sub_dirs if os.path.isdir(os.path.join(objects_dir, object_name))]


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Dextrah Scene for multi-objects Lifting"""
    # robot
    robot: ArticulationCfg = MISSING

    # object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[sim_utils.UsdFileCfg(usd_path=os.path.join(objects_dir, name, f"{name}.usd"))for name in dirs],
            random_choice=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                kinematic_enabled=False,
                disable_gravity=False,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1000.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 0.30)),
    )

    # table
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Props/Dextrah/table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.55, 0.0, 0.235),
            rot=(1.0, 0.0, 0.0, 0.0)),
    )
    
    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.5, -0.5), pos_y=(0., 0.), pos_z=(0.75, 0.75), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        hand_tips_pos = ObsTerm(func=mdp.body_state_w, params={"asset_cfg": SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"])})
        object_pos = ObsTerm(func=mdp.object_pose_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        objects_ids = ObsTerm(func=mdp.object_idx)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class CriticCfg(PolicyCfg):
        measured_body_forces = ObsTerm(func=mdp.body_incoming_wrench, params={"asset_cfg": SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"])})
        measured_joint_torques = ObsTerm(func=mdp.projected_joint_force, params={"asset_cfg": SceneEntityCfg("robot")})
        object_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_vel = ObsTerm(func=mdp.root_ang_vel_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_scale = ObsTerm(func=mdp.all_ones)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )

    joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.5, 1.),
            "damping_distribution_params": (0.5, 1.),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0. , 0.),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1., 1.),
            "velocity_range": (0., 0.),
        },
    )
    
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0., 0.), "y": (-0., 0.), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (-0., 0.), "y": (-0., 0.), "z": (-0., 0.)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class ActionsCfg:

    pass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.15, "robot_cfg": SceneEntityCfg("robot",body_names=["palm_link", ".*_tip"])},
        weight=1.0
    )

    object_to_goal = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.1, "minimal_height": 0.28, "command_name": "object_pose"},
        weight=15.0
    )

    lift = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.31}, weight=5.0)

    finger_curl_reg = RewTerm(
        func=mdp.joint_deviation_l1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="(thumb|index|middle|ring).*")},
        weight=-0.01
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={"in_bound_range": {"x":(-1., 0.), "y": (-.8, .8), "z": (.2, 2.)}, "asset_cfg": SceneEntityCfg("object")}
    )

@configclass
class DextrahEnvCfg(ManagerBasedEnvCfg):

    # Scene settings
    viewer: ViewerCfg = ViewerCfg(eye=(-5.0, 1., 0.75), lookat=(0., 1., 0.3), origin_type='env')
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None
    is_finite_horizon = True

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 1 / 120  # 60Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15
