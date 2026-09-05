# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg, preset

from . import mdp
from .mdp import stack_events

# Shared rigid-body properties for the three stacking cubes (robot-neutral).
_CUBE_PROPERTIES = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)

# Properties of the shipped ``Props/Blocks/*_block.usd`` assets, mirrored by the Newton
# cube spawns below. The block mesh spans 1.8802 units and is sized purely by an
# ``xformOp:scale`` of 0.025 authored on the mesh prim, giving a 4.7 cm edge; mass and
# friction are read from the asset's ``PhysicsMassAPI`` / ``PhysicsMaterialAPI``.
_CUBE_EDGE_LEN = 0.047
_CUBE_MASS = 0.02
_CUBE_STATIC_FRICTION = 0.8
_CUBE_DYNAMIC_FRICTION = 1.0
# Approximate the blue / red / green block materials with flat colors.
_CUBE_COLORS = {"cube_1": (0.05, 0.1, 0.75), "cube_2": (0.75, 0.05, 0.05), "cube_3": (0.05, 0.6, 0.1)}


def _usd_cube_spawn(usd_name: str, semantic_class: str) -> UsdFileCfg:
    """Build the shipped USD block spawn used by the PhysX backend."""
    return UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/{usd_name}",
        scale=(1.0, 1.0, 1.0),
        rigid_props=_CUBE_PROPERTIES,
        semantic_tags=[("class", semantic_class)],
    )


def _newton_cube_spawn(semantic_class: str) -> sim_utils.CuboidCfg:
    """Build an equivalent procedural cube spawn that renders correctly under Newton.

    The shipped block assets apply ``PhysicsRigidBodyAPI`` directly to a mesh prim whose
    size comes solely from ``xformOp:scale``. Newton body transforms carry only
    translation and rotation, so the world matrix written back for rendering resets that
    scale to 1 and the block draws at its full unscaled mesh size. Spawning a
    :class:`~isaaclab.sim.CuboidCfg` instead encodes the size in the geometry itself and
    leaves the rigid-body prim at unit scale, which sidesteps the issue.

    The cube matches the asset it replaces: same edge length, mass, and friction, and the
    block's own collider was already a bounding cube.

    Args:
        semantic_class: Semantic class tag applied to the spawned cube.

    Returns:
        The configured :class:`~isaaclab.sim.CuboidCfg`.
    """
    return sim_utils.CuboidCfg(
        size=(_CUBE_EDGE_LEN, _CUBE_EDGE_LEN, _CUBE_EDGE_LEN),
        rigid_props=_CUBE_PROPERTIES,
        mass_props=sim_utils.MassPropertiesCfg(mass=_CUBE_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=_CUBE_STATIC_FRICTION,
            dynamic_friction=_CUBE_DYNAMIC_FRICTION,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_CUBE_COLORS[semantic_class], metallic=0.0),
        semantic_tags=[("class", semantic_class)],
    )


def _cube_spawn(usd_name: str, semantic_class: str):
    """Select the cube spawn appropriate for the active physics backend."""
    return preset(
        default=_usd_cube_spawn(usd_name, semantic_class),
        newton_mjwarp=_newton_cube_spawn(semantic_class),
    )


def make_ee_frame_cfg(
    base_prim_path: str,
    target_specs: list[tuple[str, str, tuple[float, float, float]]],
    marker_scale: tuple[float, float, float] = (0.1, 0.1, 0.1),
) -> FrameTransformerCfg:
    """Build the stacking end-effector :class:`FrameTransformerCfg` with the shared marker setup.

    The marker boilerplate is identical across robots; only the source/target prim paths and
    offsets are robot-specific, so those are passed in.

    Args:
        base_prim_path: Source-frame prim path (the robot base link).
        target_specs: List of ``(prim_path, name, offset_pos)`` tuples, one per target frame.
        marker_scale: Visualization marker scale.

    Returns:
        The configured :class:`FrameTransformerCfg`.
    """
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = marker_scale
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    return FrameTransformerCfg(
        prim_path=base_prim_path,
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path=prim_path, name=name, offset=OffsetCfg(pos=pos))
            for (prim_path, name, pos) in target_specs
        ],
    )


def apply_default_semantics(scene: "ObjectTableSceneCfg") -> None:
    """Tag the table, ground, and robot with their semantic classes.

    Call this from a robot env cfg's ``__post_init__`` after the robot has been set (the robot is
    populated by the derived cfg, so it cannot be tagged in the base ``__post_init__``).
    """
    scene.table.spawn.semantic_tags = [("class", "table")]
    scene.plane.semantic_tags = [("class", "ground")]
    scene.robot.spawn.semantic_tags = [("class", "robot")]


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the stacking scene with a robot and objects.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0, 0, 0.707, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Three stacking cubes (blue / red / green). Derived robot cfgs override only the per-cube
    # ``init_state`` (transforms) to place them within the robot's reach.
    cube_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0203], rot=[0, 0, 0, 1]),
        spawn=_cube_spawn("blue_block.usd", "cube_1"),
    )
    cube_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.0203], rot=[0, 0, 0, 1]),
        spawn=_cube_spawn("red_block.usd", "cube_2"),
    )
    cube_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_3",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.1, 0.0203], rot=[0, 0, 0, 1]),
        spawn=_cube_spawn("green_block.usd", "cube_3"),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class StackEventCfg:
    """Robot-neutral reset events shared by the stacking tasks.

    Derived robot cfgs typically override the cube ``pose_range`` / ``min_separation`` to match the
    robot's reachable workspace. The joint-randomization term holds the gripper joints fixed,
    resolving them from the env's ``gripper_joint_names`` (see
    :func:`~isaaclab_tasks.contrib.stack.mdp.stack_events.randomize_joint_by_gaussian_offset`).
    """

    randomize_joint_state = EventTerm(
        func=stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_cube_positions = EventTerm(
        func=stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.6), "y": (-0.10, 0.10), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object = ObsTerm(func=mdp.object_obs)
        cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )
        stack_1 = ObsTerm(
            func=mdp.object_stacked,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("cube_2"),
                "lower_object_cfg": SceneEntityCfg("cube_1"),
            },
        )
        grasp_2 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_3"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_1_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_1")}
    )

    cube_2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_2")}
    )

    cube_3_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_3")}
    )

    success = DoneTerm(func=mdp.cubes_stacked)


@configclass
class PhysicsCfg(PresetCfg):
    """Physics backend presets for stack tasks."""

    isaacsim_physx = PhysxCfg(
        bounce_threshold_velocity=0.01,
        gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
        gpu_total_aggregate_pairs_capacity=2**21,
        friction_correlation_distance=0.00625,
        # Let object contacts stall position-driven grippers before they tunnel through a grasp.
        solve_articulation_contact_last=True,
    )
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=300,
            nconmax=200,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
            iterations=100,
            ls_iterations=15,
            ls_parallel=False,
            use_mujoco_contacts=False,
            ccd_iterations=35,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(),
        default_shape_cfg=NewtonShapeCfg(),
        num_substeps=2,
        debug_mode=False,
    )
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)
    default = isaacsim_physx


def raise_if_surface_gripper_on_newton(env_cfg) -> None:
    """Reject Newton physics for scenes that configure a surface gripper.

    Surface grippers are a PhysX/Isaac Sim feature (:class:`~isaaclab_physx.assets.SurfaceGripperCfg`)
    with no Newton equivalent. This raises an actionable error at config-validation time
    instead of failing later during scene creation with an opaque ``isaacsim.core`` import error.

    Args:
        env_cfg: The resolved environment config to inspect.
    """
    if getattr(env_cfg.scene, "surface_gripper", None) is None:
        return
    if isinstance(env_cfg.sim.physics, NewtonCfg):
        raise ValueError(
            "Surface grippers are only supported by the PhysX backend; the Newton backend has no "
            "surface-gripper implementation. Re-run this task with physics=isaacsim_physx (the default)."
        )


@configclass
class StackEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands = None
    rewards = None
    events = None
    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(-0.1, -0.5, -1.05),
        anchor_rot=(0, 0, -0.5, 0.866),
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2
        self.sim.physics = PhysicsCfg()
