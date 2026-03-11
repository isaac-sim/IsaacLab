# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import CapsuleCfg, ConeCfg, CuboidCfg, RigidBodyMaterialCfg, SphereCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.utils import PresetCfg

from . import mdp
from .adr_curriculum import CurriculumCfg

TABLE_SPAWN_CFG = sim_utils.CuboidCfg(
    size=(0.8, 1.5, 0.04),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    # trick: we let visualizer's color to show the table with success coloring
    visible=False,
)


@configclass
class ObjectCfg(PresetCfg):
    shapes = sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=[
            CuboidCfg(size=(0.05, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CuboidCfg(size=(0.05, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CuboidCfg(size=(0.025, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CuboidCfg(size=(0.025, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CuboidCfg(size=(0.025, 0.025, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CuboidCfg(size=(0.01, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            SphereCfg(radius=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            SphereCfg(radius=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CapsuleCfg(radius=0.04, height=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CapsuleCfg(radius=0.04, height=0.01, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CapsuleCfg(radius=0.04, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CapsuleCfg(radius=0.025, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CapsuleCfg(radius=0.025, height=0.2, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CapsuleCfg(radius=0.01, height=0.2, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            ConeCfg(radius=0.05, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            ConeCfg(radius=0.025, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
        ],
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
    )
    cube = sim_utils.CuboidCfg(
        size=(0.05, 0.1, 0.1),
        physics_material=RigidBodyMaterialCfg(static_friction=0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
    )
    default = shapes


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Dexsuite Scene for multi-objects Lifting"""

    # robot
    robot: ArticulationCfg = MISSING

    # object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=ObjectCfg(),  # type: ignore
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 0.35)),
    )

    # table
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=TABLE_SPAWN_CFG,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.0, 0.235), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(3.0, 5.0),
        debug_vis=False,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(-0.7, -0.3),
            pos_y=(-0.25, 0.25),
            pos_z=(0.55, 0.95),
            roll=(-3.14, 3.14),
            pitch=(-3.14, 3.14),
            yaw=(0.0, 0.0),
        ),
        success_vis_asset_name="table",
        success_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/SuccessMarkers",
            markers={
                "failure": TABLE_SPAWN_CFG.replace(
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.15, 0.15)), visible=True
                ),
                "success": TABLE_SPAWN_CFG.replace(
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.25, 0.15)), visible=True
                ),
            },
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        object_quat_b = ObsTerm(func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        target_object_pose_b = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class ProprioObsCfg(ObsGroup):
        """Observations for proprioception group."""

        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0))
        hand_tips_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            # good behaving number for position in m, velocity in m/s, rad/s,
            # and quaternion are unlikely to exceed -2 to 2 range
            clip=(-2.0, 2.0),
            params={
                "body_asset_cfg": SceneEntityCfg("robot"),
                "base_asset_cfg": SceneEntityCfg("robot"),
            },
        )
        contact: ObsTerm = MISSING

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class PerceptionObsCfg(ObsGroup):
        """Observations for perception group."""

        object_point_cloud = ObsTerm(
            func=mdp.object_point_cloud_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-2.0, 2.0),  # clamp between -2 m to 2 m
            params={"num_points": 64, "flatten": True},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_dim = 0
            self.concatenate_terms = True
            self.flatten_history_dim = True
            self.history_length = 5

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    perception: PerceptionObsCfg = PerceptionObsCfg()


@configclass
class StartupEventCfg:
    """Startup-mode domain randomization (PhysX only — Newton does not support startup events)."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": [0.5, 1.0],
            "dynamic_friction_range": [0.5, 1.0],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": [0.5, 1.0],
            "dynamic_friction_range": [0.5, 1.0],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )

    joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": [0.5, 2.0],
            "damping_distribution_params": [0.5, 2.0],
            "operation": "scale",
        },
    )

    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": [0.0, 5.0],
            "operation": "scale",
        },
    )

    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": [0.2, 2.0],
            "operation": "scale",
        },
    )


@configclass
class EventCfg:
    """Reset-mode events (shared by all physics backends)."""

    # Gravity scheduling is a deliberate curriculum trick — starting with no
    # gravity (easy) and gradually introducing full gravity (hard) makes learning
    # smoother and removes the need for a separate "Lift" reward.
    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            "operation": "abs",
        },
    )

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.2, 0.2],
                "y": [-0.2, 0.2],
                "z": [0.0, 0.4],
                "roll": [-3.14, 3.14],
                "pitch": [-3.14, 3.14],
                "yaw": [-3.14, 3.14],
            },
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "yaw": [-0.0, 0.0]},
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": [-0.50, 0.50],
            "velocity_range": [0.0, 0.0],
        },
    )

    reset_robot_wrist_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="iiwa7_joint_7"),
            "position_range": [-3, 3],
            "velocity_range": [0.0, 0.0],
        },
    )


@configclass
class ActionsCfg:
    pass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.005)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.005)

    fingers_to_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.4}, weight=1.0)

    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.2,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    orientation_tracking = RewTerm(
        func=mdp.orientation_command_error_tanh,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 1.5,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    success = RewTerm(
        func=mdp.success_reward,
        weight=10,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_std": 0.1,
            "rot_std": 0.5,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    early_termination = RewTerm(func=mdp.is_terminated_term, weight=-1, params={"term_keys": "abnormal_robot"})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-1.5, 0.5), "y": (-2.0, 2.0), "z": (0.0, 2.0)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)


@configclass
class DexsuiteReorientEnvCfg(ManagerBasedEnvCfg):
    """Dexsuite reorientation task definition, also the base definition for derivative Lift task and evaluation task"""

    # Scene settings
    viewer: ViewerCfg = ViewerCfg(eye=(-2.25, 0.0, 0.75), lookat=(0.0, 0.0, 0.45), origin_type="env")
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = MISSING  # type: ignore
    curriculum: CurriculumCfg | None = CurriculumCfg()

    def validate_config(self):
        """Check for invalid preset combinations after resolution."""
        is_newton = not isinstance(self.sim.physics, PhysxCfg)
        is_multi_asset = isinstance(self.scene.object.spawn, sim_utils.MultiAssetSpawnerCfg)

        if is_newton and is_multi_asset:
            raise ValueError(
                "Newton physics does not support multi-asset spawning."
                " Use a single-geometry object preset (e.g. presets=cube) instead of 'shapes'."
            )

        warp_supported = {"rgb", "depth", "distance_to_image_plane"}
        for cam_attr in ("base_camera", "wrist_camera"):
            cam = getattr(self.scene, cam_attr, None)
            if cam is None:
                continue
            renderer_type = getattr(cam.renderer_cfg, "renderer_type", None)
            if renderer_type == "newton_warp":
                unsupported = set(cam.data_types) - warp_supported
                if unsupported:
                    raise ValueError(
                        f"Warp renderer only supports data types {sorted(warp_supported)}, "
                        f"but '{cam_attr}' is configured with unsupported types: {sorted(unsupported)}. "
                        "Choose a compatible preset, e.g. presets=newton_renderer,rgb128."
                    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2  # 50 Hz

        # *single-goal setup
        self.commands.object_pose.resampling_time_range = (2.0, 3.0)
        self.commands.object_pose.position_only = False
        self.episode_length_s = 6.0
        self.is_finite_horizon = False

        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_max_rigid_patch_count=4 * 5 * 2**15,
            gpu_found_lost_pairs_capacity=2**26,
            gpu_found_lost_aggregate_pairs_capacity=2**29,
            gpu_total_aggregate_pairs_capacity=2**25,
        )


class DexsuiteLiftEnvCfg(DexsuiteReorientEnvCfg):
    """Dexsuite lift task definition"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.orientation_tracking = None  # no orientation reward
        self.commands.object_pose.position_only = True
        if self.curriculum is not None:
            self.rewards.success.params["rot_std"] = None  # make success reward not consider orientation


class DexsuiteReorientEnvCfg_PLAY(DexsuiteReorientEnvCfg):
    """Dexsuite reorientation task evaluation environment definition"""

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.debug_vis = True
        self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params["max_difficulty"]


class DexsuiteLiftEnvCfg_PLAY(DexsuiteLiftEnvCfg):
    """Dexsuite lift task evaluation environment definition"""

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.debug_vis = True
        self.commands.object_pose.position_only = True
        self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params["max_difficulty"]
