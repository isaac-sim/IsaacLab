# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for pushing granular MPM media from a workbench into a side bin."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.sim.schemas import MujocoJointCfg, NewtonCollisionPropertiesCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab_assets.robots.universal_robots import UR10_CFG

from . import mdp

RIGID_ENTRY = "robot"
MPM_ENTRY = "media"
MPM_VOXEL_SIZE = 0.025
MPM_COLLIDER_MARGIN = 0.5 * MPM_VOXEL_SIZE
MPM_PARTICLES_PER_CELL = 2.0
# Limit jitter to 45% of sample spacing so particles remain in their original lattice cells.
MPM_RESET_JITTER_FRACTION = 0.45
MPM_VISUAL_COLOR = (0.83, 0.60, 0.22)
PUSH_ACTION_DIM = 6
HEIGHTMAP_VISUALIZATION_SHAPE = (8, 16)
HEIGHTMAP_VISUALIZER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Policy_Heightmap",
    markers={
        "height": sim_utils.SphereCfg(
            radius=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.85, 1.00)),
        )
    },
)

# Enforce a per-world lower-node minimum and a 32-node total upper minimum.
SPARSE_MPM_MIN_LOWER_NODES_PER_WORLD = 1 << 6
SPARSE_MPM_MIN_UPPER_NODES_PER_WORLD = 1
SPARSE_MPM_MIN_TOTAL_UPPER_NODE_COUNT = 1 << 5

# Shared source of truth for the visible MPM and hidden rigid solver geometry.
WORK_SURFACE_SIZE = (1.28, 0.91, 0.04)
WORK_SURFACE_POSITION = (0.3939, 0.0, -0.02)
MPM_GROUND_SIZE = (2.60, 2.60, 0.10)
MPM_GROUND_POSITION = (0.60, 0.0, -1.10)
BIN_FLOOR_SIZE = (0.36, 0.56, 0.04)
BIN_FLOOR_POSITION = (1.2139, 0.0, -0.20)
BIN_FRONT_SIZE = (0.05, 0.66, 0.20)
BIN_FRONT_POSITION = (1.0089, 0.0, -0.10)
BIN_BACK_SIZE = (0.05, 0.66, 0.32)
BIN_BACK_POSITION = (1.4189, 0.0, -0.04)
BIN_SIDE_SIZE = (0.41, 0.05, 0.32)
BIN_LEFT_POSITION = (1.2139, 0.305, -0.04)
BIN_RIGHT_POSITION = (1.2139, -0.305, -0.04)
BIN_INNER_X_BOUNDS = (
    BIN_FLOOR_POSITION[0] - 0.5 * BIN_FLOOR_SIZE[0],
    BIN_FLOOR_POSITION[0] + 0.5 * BIN_FLOOR_SIZE[0],
)
BIN_INNER_Y_BOUNDS = (
    BIN_FLOOR_POSITION[1] - 0.5 * BIN_FLOOR_SIZE[1],
    BIN_FLOOR_POSITION[1] + 0.5 * BIN_FLOOR_SIZE[1],
)
PADDLE_SIZE = (0.34, 0.24, 0.04)
PADDLE_OFFSET = (0.5 * PADDLE_SIZE[0] + 0.02, 0.0, 0.0)
PADDLE_MASS = 1.0
PADDLE_CONTACT_MARGIN = 0.4 * MPM_VOXEL_SIZE
# The blade starts one combined MPM margin above the support surface and clear of the pile.
PADDLE_RESET_CENTER = (
    0.42,
    0.0,
    0.5 * PADDLE_SIZE[0] + MPM_COLLIDER_MARGIN + PADDLE_CONTACT_MARGIN,
)
PILE_NOMINAL_CENTER = (0.67, 0.0, 0.0)

# Particle volume equals one lattice-cell volume; radius follows Newton's ``8 * radius**3`` convention.
PILE_LOCAL_SIZE = (0.274999, 0.349999, 0.099999)
PILE_LATTICE_RESOLUTION = tuple(
    math.ceil(MPM_PARTICLES_PER_CELL * extent / MPM_VOXEL_SIZE) for extent in PILE_LOCAL_SIZE
)
PILE_LATTICE_CELL_SIZE = tuple(
    extent / resolution for extent, resolution in zip(PILE_LOCAL_SIZE, PILE_LATTICE_RESOLUTION, strict=True)
)
MPM_PARTICLE_RADIUS = 0.5 * math.prod(PILE_LATTICE_CELL_SIZE) ** (1.0 / 3.0)
PILE_SUPPORTED_CENTER_Z = WORK_SURFACE_POSITION[2] + 0.5 * WORK_SURFACE_SIZE[2] + MPM_COLLIDER_MARGIN
PILE_LOCAL_LOWER = (
    -0.5 * PILE_LOCAL_SIZE[0],
    -0.5 * PILE_LOCAL_SIZE[1],
    PILE_SUPPORTED_CENTER_Z - 0.5 * PILE_LATTICE_CELL_SIZE[2],
)
PILE_LOCAL_UPPER = tuple(lower + extent for lower, extent in zip(PILE_LOCAL_LOWER, PILE_LOCAL_SIZE, strict=True))


def _packed_group_half_extent(
    particle_count: int,
    vertical_cell_count: int,
    footprint_aspect_ratio: float,
    yaw: float,
    jitter: float,
) -> tuple[float, float]:
    """Return the yaw-expanded XY half extent of a compact particle group."""
    footprint_count = max(1, math.ceil(particle_count / vertical_cell_count))
    x_cell_count = max(1, math.ceil(math.sqrt(footprint_count * footprint_aspect_ratio)))
    y_cell_count = max(1, math.ceil(footprint_count / x_cell_count))
    half_x = 0.5 * (x_cell_count - 1) * PILE_LATTICE_CELL_SIZE[0] + jitter
    half_y = 0.5 * (y_cell_count - 1) * PILE_LATTICE_CELL_SIZE[1] + jitter
    return (
        half_x * math.cos(yaw) + half_y * math.sin(yaw),
        half_y * math.cos(yaw) + half_x * math.sin(yaw),
    )


UR10_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def _spawn_fixed_paddle(
    prim_path: str,
    cfg: sim_utils.CuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a mass-bearing paddle welded to its parent rigid body."""
    # Keep both the USD imports and clone-wrapper resolution behind SimulationApp startup.
    # Importing either at config-parse time loads the kit-less ``usd-core`` bindings and
    # corrupts Kit's own USD binding registration.
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: PLC0415

    @sim_utils.clone
    def _spawn_one(
        prim_path: str,
        cfg: sim_utils.CuboidCfg,
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        **kwargs,
    ) -> Usd.Prim:
        paddle_prim = sim_utils.spawn_cuboid(
            prim_path,
            cfg,
            translation=translation,
            orientation=orientation,
            **kwargs,
        )
        stage = sim_utils.get_current_stage()
        parent_prim = paddle_prim.GetParent()
        if not parent_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Paddle parent '{parent_prim.GetPath()}' is not a rigid body.")

        # Keep the physical collider hidden while allowing a visible child mesh on the
        # same paddle body. Hiding the body root would also hide all of its children.
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{prim_path}/geometry/mesh")).MakeInvisible()

        local_position = translation if translation is not None else (0.0, 0.0, 0.0)
        local_rotation = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
        joint = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/FixedJoint")
        joint.CreateBody0Rel().SetTargets([parent_prim.GetPath()])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(prim_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_position))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(local_rotation[3], Gf.Vec3f(*local_rotation[:3])))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        return paddle_prim

    return _spawn_one(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)


# Collision-screened nominal pose for PADDLE_RESET_CENTER. The reset IK bank stays on this branch
# while varying only small upright paddle translations and world-Z yaw.
UR10_PUSH_HOME = (
    -2.6687801,
    -0.9745972,
    3.8264365,
    -1.2810427,
    1.5707960,
    -1.0979838,
)


def get_mpm_solver_cfg(cfg: UR10ParticlePushEnvCfg) -> MPMSolverCfg:
    """Return the unique implicit-MPM solver entry."""
    entries = [entry for entry in cfg.sim.physics.solver_cfg.entries if entry.name == MPM_ENTRY]
    if len(entries) != 1 or not isinstance(entries[0].solver_cfg, MPMSolverCfg):
        raise ValueError(f"Expected one {MPM_ENTRY!r} MPMSolverCfg entry, found {len(entries)}.")
    return entries[0].solver_cfg


def configure_sparse_mpm_capacities(cfg: UR10ParticlePushEnvCfg) -> None:
    """Scale rebuildable sparse-grid capacities with the final world count.

    Command-line ``--num_envs`` overrides are applied after config construction. The environment
    calls this function once more immediately before simulation creation, so reduced evaluation
    runs and large distributed jobs reserve proportional memory.
    """
    per_world = {
        "active cells": cfg.mpm_active_cell_count_per_world,
        "leaf nodes": cfg.mpm_leaf_node_count_per_world,
        "lower nodes": cfg.mpm_lower_node_count_per_world,
        "upper nodes": cfg.mpm_upper_node_count_per_world,
    }
    for name, capacity in per_world.items():
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"Sparse MPM {name} capacity per world must be a positive integer, got {capacity!r}.")
    if not (
        per_world["upper nodes"] <= per_world["lower nodes"] <= per_world["leaf nodes"] <= per_world["active cells"]
    ):
        raise ValueError(
            "Sparse MPM per-world capacity hierarchy must satisfy "
            "upper nodes <= lower nodes <= leaf nodes <= active cells."
        )
    if per_world["lower nodes"] < SPARSE_MPM_MIN_LOWER_NODES_PER_WORLD:
        raise ValueError(
            "UR10 Push sparse MPM lower-node capacity is below the runtime-topology-derived safety floor "
            f"of {SPARSE_MPM_MIN_LOWER_NODES_PER_WORLD} nodes per world."
        )
    if per_world["upper nodes"] < SPARSE_MPM_MIN_UPPER_NODES_PER_WORLD:
        raise ValueError(
            "UR10 Push sparse MPM upper-node capacity is below the runtime-topology-derived safety floor "
            f"of {SPARSE_MPM_MIN_UPPER_NODES_PER_WORLD} nodes per world."
        )

    world_count = max(1, int(cfg.scene.num_envs))
    solver_cfg = get_mpm_solver_cfg(cfg)
    solver_cfg.max_active_cell_count = per_world["active cells"] * world_count
    solver_cfg.max_leaf_node_count = per_world["leaf nodes"] * world_count
    solver_cfg.max_lower_node_count = per_world["lower nodes"] * world_count
    solver_cfg.max_upper_node_count = max(
        SPARSE_MPM_MIN_TOTAL_UPPER_NODE_COUNT,
        per_world["upper nodes"] * world_count,
    )


def _kinematic_box(
    prim_path: str,
    *,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float],
    visible: bool = True,
) -> RigidObjectCfg:
    """Build a declarative kinematic MPM collider."""
    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=UsdPhysicsRigidBodyCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
            ),
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=True,
                contact_margin=MPM_COLLIDER_MARGIN,
                # Implicit MPM consumes shape margin; gap is a rigid-contact parameter.
                contact_gap=0.0,
            ),
            physics_material=RigidBodyMaterialBaseCfg(
                static_friction=0.8,
                dynamic_friction=0.7,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.65,
            ),
            visible=visible,
        ),
    )


def _static_collision_box(
    prim_path: str,
    *,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
) -> AssetBaseCfg:
    """Build the hidden static mirror used exclusively by the rigid subsolver."""
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=True,
                contact_margin=0.004,
                contact_gap=0.002,
            ),
            physics_material=RigidBodyMaterialBaseCfg(
                static_friction=0.8,
                dynamic_friction=0.7,
            ),
            visible=False,
        ),
    )


@configclass
class UR10ParticlePushSceneCfg(InteractiveSceneCfg):
    """Official workcell plus aligned MPM and rigid work-surface/bin collision."""

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0),
            rot=(0.0, 0.0, 0.70710678, 0.70710678),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            # Newton imports the table's visual-only meshes when they belong to a body.
            rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
        ),
    )
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )

    robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.joint_pos = dict(zip(UR10_JOINT_NAMES, UR10_PUSH_HOME, strict=True))
    # Override arm drive gains; preserve the USD inertia, limits, and effort cap.
    robot.actuators["arm"].stiffness = 2400.0
    robot.actuators["arm"].damping = 70.0
    # Enable actuator gravity compensation in MuJoCo; do not add task-level effort commands.
    robot.spawn.joint_drive_props = [MujocoJointCfg(actuatorgravcomp=True)]

    # Weld the paddle to ee_link; separate collision and visual geometry for independent visibility.
    paddle = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/Paddle",
        init_state=AssetBaseCfg.InitialStateCfg(pos=PADDLE_OFFSET),
        spawn=sim_utils.CuboidCfg(
            func=_spawn_fixed_paddle,
            size=PADDLE_SIZE,
            rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassCfg(mass=PADDLE_MASS),
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=True,
                contact_margin=PADDLE_CONTACT_MARGIN,
                contact_gap=0.002,
            ),
            physics_material=RigidBodyMaterialBaseCfg(
                static_friction=0.8,
                dynamic_friction=0.7,
            ),
        ),
    )
    paddle_visual = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/Paddle/PaddleVisual",
        spawn=sim_utils.CuboidCfg(
            size=PADDLE_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.45, 0.82),
                metallic=0.25,
                roughness=0.35,
            ),
        ),
    )

    # The official table supplies rigid robot contact. This co-located simple slab belongs only to
    # the MPM entry, avoiding a mesh approximation and keeping particle collision explicit.
    mpm_work_surface = _kinematic_box(
        "{ENV_REGEX_NS}/MPMWorkSurface",
        size=WORK_SURFACE_SIZE,
        position=WORK_SURFACE_POSITION,
        color=(0.3, 0.3, 0.3),
        visible=False,
    )
    # Newton MPM owns a simple per-world mirror of the visible global lab floor. Without it, the
    # few grains permitted to spill leave the solver entirely and accelerate forever.
    mpm_ground = _kinematic_box(
        "{ENV_REGEX_NS}/MPMGround",
        size=MPM_GROUND_SIZE,
        position=MPM_GROUND_POSITION,
        color=(0.18, 0.18, 0.18),
        visible=False,
    )

    # Catch tote mounted against the +X edge of the workbench. Its front wall ends flush with the
    # tabletop and overlaps the floor, closing the otherwise hidden under-table escape slot without
    # adding a lip above the sweep surface.
    bin_floor = _kinematic_box(
        "{ENV_REGEX_NS}/MPMBinFloor",
        size=BIN_FLOOR_SIZE,
        position=BIN_FLOOR_POSITION,
        color=(0.12, 0.22, 0.34),
    )
    bin_front = _kinematic_box(
        "{ENV_REGEX_NS}/MPMBinFront",
        size=BIN_FRONT_SIZE,
        position=BIN_FRONT_POSITION,
        color=(0.12, 0.22, 0.34),
    )
    bin_back = _kinematic_box(
        "{ENV_REGEX_NS}/MPMBinBack",
        size=BIN_BACK_SIZE,
        position=BIN_BACK_POSITION,
        color=(0.12, 0.22, 0.34),
    )
    bin_left = _kinematic_box(
        "{ENV_REGEX_NS}/MPMBinLeft",
        size=BIN_SIDE_SIZE,
        position=BIN_LEFT_POSITION,
        color=(0.12, 0.22, 0.34),
    )
    bin_right = _kinematic_box(
        "{ENV_REGEX_NS}/MPMBinRight",
        size=BIN_SIDE_SIZE,
        position=BIN_RIGHT_POSITION,
        color=(0.12, 0.22, 0.34),
    )

    # These static mirrors are automatically owned by the rigid entry together with the official
    # table and ground. The MPM entry owns only the kinematic copies above, so no shape is shared.
    rigid_bin_floor = _static_collision_box(
        "{ENV_REGEX_NS}/RigidBinFloor",
        size=BIN_FLOOR_SIZE,
        position=BIN_FLOOR_POSITION,
    )
    rigid_bin_front = _static_collision_box(
        "{ENV_REGEX_NS}/RigidBinFront",
        size=BIN_FRONT_SIZE,
        position=BIN_FRONT_POSITION,
    )
    rigid_bin_back = _static_collision_box(
        "{ENV_REGEX_NS}/RigidBinBack",
        size=BIN_BACK_SIZE,
        position=BIN_BACK_POSITION,
    )
    rigid_bin_left = _static_collision_box(
        "{ENV_REGEX_NS}/RigidBinLeft",
        size=BIN_SIDE_SIZE,
        position=BIN_LEFT_POSITION,
    )
    rigid_bin_right = _static_collision_box(
        "{ENV_REGEX_NS}/RigidBinRight",
        size=BIN_SIDE_SIZE,
        position=BIN_RIGHT_POSITION,
    )

    media = MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Media",
        init_state=MPMObjectCfg.InitialStateCfg(pos=PILE_NOMINAL_CENTER),
        spawn=MPMGridCfg(
            # Emit cell centers so the lowest layer lies on the expanded support plane.
            lower=PILE_LOCAL_LOWER,
            upper=PILE_LOCAL_UPPER,
            voxel_size=MPM_VOXEL_SIZE,
            particles_per_cell=MPM_PARTICLES_PER_CELL,
            particle_placement="cell_center",
            # Use one jitter source. Per-reset cell-stratified jitter below provides a fresh
            # granular arrangement every episode instead of repeating one build-time sample.
            jitter=0.0,
            radius=MPM_PARTICLE_RADIUS,
            material=MPMParticleMaterialCfg(
                density=300.0,
                friction=0.7,
                yield_pressure=1.0e12,
            ),
            visual_color=MPM_VISUAL_COLOR,
        ),
    )


@configclass
class ActionsCfg:
    """One bounded relative-position command for the six UR10 arm joints."""

    # Snapshot one relative joint target per policy step.
    arm_action = mdp.ClampedRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(UR10_JOINT_NAMES),
        preserve_order=True,
        scale=dict(
            zip(
                UR10_JOINT_NAMES,
                (0.035, 0.035, 0.035, 0.052, 0.052, 0.052),
                strict=True,
            )
        ),
        use_zero_offset=True,
        joint_limit_margin=0.04,
    )


@configclass
class ObservationsCfg:
    """Actor image/vector groups and privileged critic state."""

    @configclass
    class PolicyCfg(ObsGroup):
        state = ObsTerm(func=mdp.policy_observation)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class HeightmapCfg(ObsGroup):
        image = ObsTerm(func=mdp.heightmap_observation)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        state = ObsTerm(func=mdp.critic_observation)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    heightmap: HeightmapCfg = HeightmapCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Signed potential progress, terminal outcomes, and small continuous-time costs."""

    # Delivery and transport are signed potential differences, so holding a partial solution does
    # not accumulate reward. Terminal impulses dominate the bounded shaping terms; the remaining
    # weights are continuous-time rates integrated by RewardManager.
    success = RewTerm(func=mdp.success_event, weight=2.0)
    bin_progress = RewTerm(func=mdp.BinProgressReward, weight=1.50)
    transport_progress = RewTerm(func=mdp.TransportProgressReward, weight=0.15)
    spill = RewTerm(func=mdp.spill_fraction, weight=-0.20)
    failure = RewTerm(func=mdp.failure_event, weight=-2.0)
    action_magnitude = RewTerm(func=mdp.action_magnitude, weight=-0.10)
    action_rate = RewTerm(func=mdp.action_rate, weight=-0.01)


@configclass
class TerminationsCfg:
    """Numerical safety, irrecoverable particle loss, success, and neutral timeout."""

    invalid_state = DoneTerm(func=mdp.invalid_state)
    escaped_workspace = DoneTerm(func=mdp.escaped_workspace)
    excessive_spill = DoneTerm(func=mdp.excessive_spill)
    success = DoneTerm(func=mdp.success)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventsCfg:
    """Domain randomization events."""

    randomize_robot_and_pile = EventTerm(func=mdp.randomize_push_scene, mode="reset")


@configclass
class CurriculumCfg:
    """Persistent coverage of the single-push reset distributions."""

    reset_randomization = CurrTerm(func=mdp.SinglePushCurriculum)


@configclass
class UR10ParticlePushEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based, relative-joint-control UR10 particle-pushing task."""

    decimation = 2
    # One approach and sweep comfortably fits within this horizon.
    episode_length_s = 12.0
    # Treat the deadline as a training truncation. The actor has no remaining-time observation,
    # so assigning a hidden finite-horizon terminal value would make the value function non-Markov.
    is_finite_horizon = False
    # 50 x 86 cells at 20 mm resolution; channels are current surface, delayed surface, and bin mask.
    heightmap_shape: tuple[int, int] = (50, 86)
    heightmap_history_steps: int = 4
    scene: UR10ParticlePushSceneCfg = UR10ParticlePushSceneCfg(
        num_envs=64,
        env_spacing=3.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    ee_body_name: str = "ee_link"
    paddle_offset: tuple[float, float, float] = PADDLE_OFFSET
    paddle_size: tuple[float, float, float] = PADDLE_SIZE
    paddle_reset_center: tuple[float, float, float] = PADDLE_RESET_CENTER
    pile_nominal_center: tuple[float, float, float] = PILE_NOMINAL_CENTER
    # Pack the fixed particle population into compact, volume-preserving shapes. Each profile is
    # ``(vertical cells, footprint X/Y aspect ratio)``. Their average width is close to the blade,
    # so one coordinated sweep is sufficient while shape variation remains observable.
    reset_source_shape_profiles: tuple[tuple[int, float], ...] = (
        (12, 1.00),
        (14, 0.75),
        (14, 1.25),
    )
    # The reset event samples a startup-only bank of collision-screened robot starts. Retain every
    # difficulty throughout training while exposing the policy to full randomization immediately.
    # Every level remains the same one-pile, one-sweep task and keeps all particles outside the bin.
    reset_pile_center_x: tuple[float, ...] = (0.78, 0.72, PILE_NOMINAL_CENTER[0])
    reset_randomization_scales: tuple[float, ...] = (0.35, 0.65, 1.0)
    reset_level_probabilities: tuple[float, ...] = (0.20, 0.40, 0.40)
    reset_seed: int = 17
    reset_pose_count: int = 192
    reset_cycle: bool = False
    # Randomize the robot in task space, then sample the pile relative to the paddle. This keeps
    # every reset immediately useful for learning while covering a broad region of the table.
    reset_paddle_longitudinal_offset_range: tuple[float, float] = (-0.08, 0.08)
    reset_paddle_max_lateral_offset: float = 0.16
    reset_paddle_max_yaw: float = 0.25
    reset_pile_paddle_distance_range: tuple[float, float] = (0.225, 0.285)
    reset_pile_paddle_lateral_offset_range: tuple[float, float] = (-0.06, 0.06)
    reset_particle_max_yaw: float = 0.20
    reset_particle_jitter: float = MPM_RESET_JITTER_FRACTION * min(PILE_LATTICE_CELL_SIZE)
    reset_ik_seeds: int = 64
    reset_ik_iterations: int = 96
    reset_ik_noise_std: float = 0.25
    reset_ik_max_cost: float = 1.0e-3
    # Keep the leading particle safely behind the expanded bin-front collision band.
    reset_bin_clearance: float = 0.01

    # Segmented overhead-depth adapter. These features can be reproduced from a calibrated real
    # overhead camera without exposing simulator-only particle identities to the actor.
    heightmap_x_bounds: tuple[float, float] = (-0.25, 1.46)
    heightmap_y_bounds: tuple[float, float] = (-0.50, 0.50)
    heightmap_z_min: float = -0.20
    heightmap_z_range: float = 0.40
    heightmap_depth_noise_std: float = 0.004
    heightmap_xy_noise_std: float = 0.003
    heightmap_dropout_probability: float = 0.01
    heightmap_visualizer_cfg: VisualizationMarkersCfg | None = None

    bin_inner_x_bounds: tuple[float, float] = BIN_INNER_X_BOUNDS
    bin_inner_y_bounds: tuple[float, float] = BIN_INNER_Y_BOUNDS
    # Delivery includes ordinary compliant contact down to the protected floor core so a
    # compressed bottom layer remains part of the successful payload.
    bin_inner_z_bounds: tuple[float, float] = (-0.186, 0.09)
    bin_physical_z_bounds: tuple[float, float] = (-0.186, 0.12)
    success_fraction: float = 0.60
    success_max_spill_fraction: float = 0.02
    failure_max_spill_fraction: float = 0.20
    # Settled-quality reference for privileged diagnostics and the physical validator. Terminal
    # success uses sustained bin occupancy so remaining source particles do not veto delivery.
    success_max_rms_particle_speed: float = 0.12
    success_dwell_time_s: float = 0.30
    particle_workspace_lower_bound: tuple[float, float, float] = (-0.30, -0.60, -0.35)
    particle_workspace_upper_bound: tuple[float, float, float] = (1.50, 0.60, 0.80)
    # Terminate when escaped particles exceed 2% to bound sparse-grid growth.
    max_escaped_particle_fraction: float = 0.02
    # Bound numerical velocity outliers before they can cross many sparse-grid regions.
    particle_max_velocity: float = 10.0
    # Reset extreme-but-finite rigid states before they can poison policy/value inputs. These are
    # numerical-instability bounds, deliberately much looser than the robot's normal motion.
    state_bound_joint_position_margin: float = 0.05
    state_bound_max_joint_velocity: float = 20.0
    state_bound_max_ee_linear_velocity: float = 10.0
    state_bound_max_ee_angular_velocity: float = 50.0
    # Reserve active sparse-grid cells per world, independent of particle count.
    mpm_active_cell_count_per_world: int = 3072
    mpm_leaf_node_count_per_world: int = 1 << 9
    mpm_lower_node_count_per_world: int = SPARSE_MPM_MIN_LOWER_NODES_PER_WORLD
    mpm_upper_node_count_per_world: int = SPARSE_MPM_MIN_UPPER_NODES_PER_WORLD
    # Scale only the virtual paddle inertia inside MPM to limit proxy acceleration under granular
    # loading; the rigid solver retains the authored 1 kg tool and receives the harvested wrench.
    proxy_mass_scale: float = 10.0

    def _validate_source_shape_profiles(self) -> None:
        """Validate source-pile profiles consumed by reset packing."""
        if not self.reset_source_shape_profiles or any(
            len(profile) != 2 for profile in self.reset_source_shape_profiles
        ):
            raise ValueError("Source shape profiles must contain vertical cells and an aspect ratio.")
        if any(
            isinstance(vertical_cells, bool)
            or not isinstance(vertical_cells, int)
            or vertical_cells < 1
            or not math.isfinite(aspect_ratio)
            or aspect_ratio <= 0.0
            for vertical_cells, aspect_ratio in self.reset_source_shape_profiles
        ):
            raise ValueError(
                "Source shape profiles require a positive integer height and finite positive aspect ratio."
            )

    def _validate_reset_config(self) -> None:
        """Validate values required to build the collision-screened reset bank."""
        self._validate_source_shape_profiles()
        particle_radius = self.scene.media.spawn.radius
        if particle_radius is None or not math.isfinite(particle_radius) or particle_radius <= 0.0:
            raise ValueError("UR10 Push requires an explicit positive particle radius.")
        level_count = len(self.reset_randomization_scales)
        if level_count < 1 or len(self.reset_pile_center_x) != level_count:
            raise ValueError("The reset curriculum must contain matching center and scale sequences.")
        if (
            len(self.reset_level_probabilities) != level_count
            or any(not math.isfinite(value) or value < 0.0 for value in self.reset_level_probabilities)
            or not math.isclose(sum(self.reset_level_probabilities), 1.0)
        ):
            raise ValueError("Reset-level probabilities must be finite, non-negative, and sum to one.")
        if (
            isinstance(self.reset_pose_count, bool)
            or not isinstance(self.reset_pose_count, int)
            or self.reset_pose_count < level_count
            or self.reset_pose_count % level_count != 0
        ):
            raise ValueError("reset_pose_count must be positive and divisible by the reset curriculum level count.")
        if (
            any(not math.isfinite(value) for value in self.reset_pile_center_x)
            or any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in self.reset_randomization_scales)
            or any(
                current >= following
                for current, following in zip(
                    self.reset_randomization_scales,
                    self.reset_randomization_scales[1:],
                )
            )
        ):
            raise ValueError("Reset scales must be finite, strictly increasing values in (0, 1].")

        centers = (*self.paddle_reset_center, *self.pile_nominal_center)
        if (
            len(self.paddle_reset_center) != 3
            or len(self.pile_nominal_center) != 3
            or not all(math.isfinite(value) for value in centers)
        ):
            raise ValueError("Paddle and pile reset centers must contain three finite coordinates.")
        reset_ranges = (
            self.reset_paddle_longitudinal_offset_range,
            self.reset_pile_paddle_distance_range,
            self.reset_pile_paddle_lateral_offset_range,
        )
        if any(
            len(bounds) != 2 or not all(math.isfinite(value) for value in bounds) or bounds[0] > bounds[1]
            for bounds in reset_ranges
        ):
            raise ValueError("Reset ranges must contain two finite values in increasing order.")
        if self.reset_pile_paddle_distance_range[0] <= 0.0:
            raise ValueError("The pile-to-paddle distance range must be positive.")
        randomization_magnitudes = (
            self.reset_particle_max_yaw,
            self.reset_particle_jitter,
            self.reset_paddle_max_lateral_offset,
            self.reset_paddle_max_yaw,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in randomization_magnitudes):
            raise ValueError("Reset randomization magnitudes must be finite and non-negative.")
        if max(self.reset_particle_max_yaw, self.reset_paddle_max_yaw) >= 0.5 * math.pi:
            raise ValueError("Reset yaw magnitudes must be smaller than pi/2.")
        if self.reset_particle_jitter >= 0.5 * min(PILE_LATTICE_CELL_SIZE):
            raise ValueError("Particle reset jitter must be smaller than half the particle spacing.")
        if not math.isfinite(self.reset_bin_clearance) or self.reset_bin_clearance < 0.0:
            raise ValueError("reset_bin_clearance must be finite and non-negative.")

        paddle_size = tuple(self.paddle_size)
        if (
            len(paddle_size) != 3
            or any(not math.isfinite(value) or value <= 0.0 for value in paddle_size)
            or tuple(self.scene.paddle.spawn.size) != paddle_size
            or tuple(self.scene.paddle_visual.spawn.size) != paddle_size
        ):
            raise ValueError("paddle_size must be finite, positive, and match both authored paddle geometries.")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (self.reset_ik_seeds, self.reset_ik_iterations)
        ):
            raise ValueError("Reset IK seed and iteration counts must be positive integers.")
        if any(not math.isfinite(value) or value <= 0.0 for value in (self.reset_ik_max_cost, self.reset_ik_noise_std)):
            raise ValueError("Reset IK cost and noise thresholds must be finite and positive.")
        self._validate_particle_reset_envelope()

    def _validate_particle_reset_envelope(self) -> None:
        """Keep every randomized single pile on the table and clear of the paddle and bin."""
        particle_count = math.prod(PILE_LATTICE_RESOLUTION)
        safe_front_x = BIN_FRONT_POSITION[0] - 0.5 * BIN_FRONT_SIZE[0] - MPM_COLLIDER_MARGIN - self.reset_bin_clearance
        safe_table_min_x = WORK_SURFACE_POSITION[0] - 0.5 * WORK_SURFACE_SIZE[0] + MPM_COLLIDER_MARGIN
        safe_table_half_width = 0.5 * WORK_SURFACE_SIZE[1] - MPM_COLLIDER_MARGIN
        nominal_distance = self.pile_nominal_center[0] - self.paddle_reset_center[0]
        required_paddle_gap = MPM_COLLIDER_MARGIN + PADDLE_CONTACT_MARGIN
        for level, (center_x, scale) in enumerate(
            zip(self.reset_pile_center_x, self.reset_randomization_scales, strict=True)
        ):
            half_extents = tuple(
                _packed_group_half_extent(
                    particle_count,
                    vertical_cell_count,
                    aspect_ratio,
                    scale * self.reset_particle_max_yaw,
                    self.reset_particle_jitter,
                )
                for vertical_cell_count, aspect_ratio in self.reset_source_shape_profiles
            )
            maximum_half_x = max(extent[0] for extent in half_extents)
            maximum_half_y = max(extent[1] for extent in half_extents)
            paddle_yaw = scale * self.reset_paddle_max_yaw
            paddle_half_x = 0.5 * (
                self.paddle_size[2] * math.cos(paddle_yaw) + self.paddle_size[1] * math.sin(paddle_yaw)
            )
            minimum_distance = nominal_distance + scale * (self.reset_pile_paddle_distance_range[0] - nominal_distance)
            minimum_paddle_gap = minimum_distance - maximum_half_x - paddle_half_x
            if minimum_paddle_gap < required_paddle_gap:
                raise ValueError(
                    f"Reset level {level} can overlap the paddle and pile "
                    f"({minimum_paddle_gap:.6f} < {required_paddle_gap:.6f})."
                )

            minimum_center_x = center_x + scale * (
                self.reset_paddle_longitudinal_offset_range[0]
                + self.reset_pile_paddle_distance_range[0]
                - nominal_distance
            )
            maximum_center_x = center_x + scale * (
                self.reset_paddle_longitudinal_offset_range[1]
                + self.reset_pile_paddle_distance_range[1]
                - nominal_distance
            )
            if minimum_center_x - maximum_half_x < safe_table_min_x:
                raise ValueError(f"Reset level {level} can spawn particles beyond the rear work-surface edge.")
            if maximum_center_x + maximum_half_x > safe_front_x:
                raise ValueError(f"Reset level {level} can spawn particles in the bin-front contact band.")
            maximum_center_y = scale * (
                self.reset_paddle_max_lateral_offset
                + max(abs(value) for value in self.reset_pile_paddle_lateral_offset_range)
            )
            if maximum_center_y + maximum_half_y > safe_table_half_width:
                raise ValueError(f"Reset level {level} can spawn particles beyond the lateral work-surface edge.")

    def _validate_heightmap_config(self) -> None:
        """Validate dimensions and bounds consumed by the image observation."""
        if len(self.heightmap_shape) != 2 or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 8 for size in self.heightmap_shape
        ):
            raise ValueError("heightmap_shape must contain two integer dimensions of at least eight cells.")
        if (
            isinstance(self.heightmap_history_steps, bool)
            or not isinstance(self.heightmap_history_steps, int)
            or self.heightmap_history_steps < 1
        ):
            raise ValueError("heightmap_history_steps must be a positive integer.")
        for name, bounds in (
            ("heightmap_x_bounds", self.heightmap_x_bounds),
            ("heightmap_y_bounds", self.heightmap_y_bounds),
            ("bin_inner_x_bounds", self.bin_inner_x_bounds),
            ("bin_inner_y_bounds", self.bin_inner_y_bounds),
            ("bin_inner_z_bounds", self.bin_inner_z_bounds),
            ("bin_physical_z_bounds", self.bin_physical_z_bounds),
        ):
            if len(bounds) != 2 or not all(math.isfinite(value) for value in bounds) or bounds[0] >= bounds[1]:
                raise ValueError(f"{name} must contain finite, strictly increasing bounds.")
        for heightmap_name, heightmap_bounds, bin_name, bin_bounds in (
            ("heightmap_x_bounds", self.heightmap_x_bounds, "bin_inner_x_bounds", self.bin_inner_x_bounds),
            ("heightmap_y_bounds", self.heightmap_y_bounds, "bin_inner_y_bounds", self.bin_inner_y_bounds),
        ):
            if heightmap_bounds[0] > bin_bounds[0] or heightmap_bounds[1] < bin_bounds[1]:
                raise ValueError(f"{heightmap_name} must contain the complete {bin_name}.")
        if (
            not math.isfinite(self.heightmap_z_min)
            or not math.isfinite(self.heightmap_z_range)
            or self.heightmap_z_range <= 0.0
        ):
            raise ValueError("The heightmap vertical range must be finite and positive.")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (self.heightmap_depth_noise_std, self.heightmap_xy_noise_std)
        ):
            raise ValueError("Heightmap noise magnitudes must be finite and non-negative.")
        if not math.isfinite(self.heightmap_dropout_probability) or not 0.0 <= self.heightmap_dropout_probability < 1.0:
            raise ValueError("heightmap_dropout_probability must lie in [0, 1).")
        work_surface_top = (
            self.scene.mpm_work_surface.init_state.pos[2] + 0.5 * self.scene.mpm_work_surface.spawn.size[2]
        )
        maximum_vertical_cells = max(profile[0] for profile in self.reset_source_shape_profiles)
        required_heightmap_top = (
            work_surface_top
            + MPM_COLLIDER_MARGIN
            + (maximum_vertical_cells - 1) * PILE_LATTICE_CELL_SIZE[2]
            + 2.0 * self.reset_particle_jitter
            + self.scene.media.spawn.radius
        )
        if self.heightmap_z_min + self.heightmap_z_range <= required_heightmap_top:
            raise ValueError("The heightmap vertical range must include the complete deployment pile.")

    def _validate_runtime_config(self) -> None:
        """Validate task thresholds, observation divisors, and workspace bounds."""
        if not math.isfinite(self.state_bound_joint_position_margin) or self.state_bound_joint_position_margin < 0.0:
            raise ValueError("state_bound_joint_position_margin must be finite and non-negative.")
        if not 0.0 < self.success_fraction <= 1.0 or not (
            0.0 <= self.success_max_spill_fraction <= self.failure_max_spill_fraction < 1.0
        ):
            raise ValueError("Success and failure fractions must be ordered within [0, 1].")
        if not 0.0 <= self.max_escaped_particle_fraction <= self.failure_max_spill_fraction:
            raise ValueError("The escaped-particle tolerance must lie within the irrecoverable-spill threshold.")
        runtime_bounds = (
            self.state_bound_max_joint_velocity,
            self.state_bound_max_ee_linear_velocity,
            self.state_bound_max_ee_angular_velocity,
            self.proxy_mass_scale,
            self.success_dwell_time_s,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in runtime_bounds):
            raise ValueError("State bounds, proxy mass scale, and success dwell time must be finite and positive.")
        observation_divisors = (
            self.particle_max_velocity,
            self.success_max_rms_particle_speed,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in observation_divisors):
            raise ValueError("Particle-speed scales must be finite and positive.")
        lower = self.particle_workspace_lower_bound
        upper = self.particle_workspace_upper_bound
        if len(lower) != 3 or len(upper) != 3 or any(not math.isfinite(value) for value in (*lower, *upper)):
            raise ValueError("Particle workspace bounds must each contain three finite values.")
        if any(low >= high for low, high in zip(lower, upper, strict=True)):
            raise ValueError("Particle workspace lower bounds must be smaller than upper bounds.")

    def validate_config(self) -> None:
        """Validate the final configuration after command-line overrides are applied."""
        self._validate_reset_config()
        self._validate_heightmap_config()
        self._validate_runtime_config()

    def __post_init__(self) -> None:
        from isaaclab_visualizers.kit import KitVisualizerCfg

        self.sim.default_visualizer_cfg = KitVisualizerCfg(
            eye=(1.7, 1.5, 1.0),
            lookat=(0.55, 0.0, 0.0),
            origin_type="env",
            origin_env_index=0,
        )
        self.sim.physics = NewtonCfg(
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(
                        name=RIGID_ENTRY,
                        solver_cfg=MJWarpSolverCfg(
                            use_mujoco_contacts=False,
                            integrator="implicitfast",
                            njmax=256,
                            nconmax=512,
                        ),
                        bodies=[r"/World/envs/env_.*/Robot", r"/World/envs/env_.*/Table"],
                        include_static_shapes=True,
                        # Refine rigid integration with three substeps per coupled interval.
                        substeps=3,
                    ),
                    CouplerEntryCfg(
                        name=MPM_ENTRY,
                        solver_cfg=MPMSolverCfg(
                            voxel_size=MPM_VOXEL_SIZE,
                            grid_type="sparse",
                            grid_padding=0,
                            strain_basis="P0",
                            transfer_scheme="apic",
                            # Preserve particle-backed constitutive history on the rebuildable sparse grid.
                            max_iterations=24,
                            tolerance=1.0e-4,
                            warmstart_mode="auto",
                            velocity_basis="Q1",
                            collider_basis="pic27",
                            collider_velocity_mode="forward",
                            solver="auto",
                            separate_worlds=True,
                            project_outside_colliders=False,
                        ),
                        bodies=[
                            r"/World/envs/env_.*/MPMWorkSurface",
                            r"/World/envs/env_.*/MPMGround",
                            r"/World/envs/env_.*/MPMBinFloor",
                            r"/World/envs/env_.*/MPMBinFront",
                            r"/World/envs/env_.*/MPMBinBack",
                            r"/World/envs/env_.*/MPMBinLeft",
                            r"/World/envs/env_.*/MPMBinRight",
                        ],
                        all_particles=True,
                        include_static_shapes=False,
                        include_child_joints=False,
                        # Keep entry-local substeps at one; collider poses refresh between outer coupled substeps.
                        substeps=1,
                        in_place=True,
                    ),
                ],
                proxies=[
                    CouplerProxyMappingCfg(
                        source=RIGID_ENTRY,
                        destination=MPM_ENTRY,
                        bodies=[r"/World/envs/env_.*/Robot/ee_link/Paddle"],
                        mode="lagged",
                        mass_scale=self.proxy_mass_scale,
                        collision_pipeline=None,
                    )
                ],
                iterations=1,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
            # Run one coupled solve per 120 Hz simulation step.
            num_substeps=1,
            # Bounded sparse topology defers capture until the initial reset has authored its graph shape.
            use_cuda_graph=True,
        )
        # Keep the UR10's implicit drives on the Newton backend used by the coupled step.
        self.sim.use_newton_actuators = True
        configure_sparse_mpm_capacities(self)

    def play_mode(self) -> None:
        """Cycle the randomized reset bank in a small playback scene."""
        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 4)
        self.heightmap_depth_noise_std = 0.0
        self.heightmap_xy_noise_std = 0.0
        self.heightmap_dropout_probability = 0.0
        self.reset_cycle = True
        self.reset_level_probabilities = (0.0,) * (len(self.reset_randomization_scales) - 1) + (1.0,)
        camera_cfg = self.sim.default_visualizer_cfg
        self.sim.default_visualizer_cfg = NewtonRTXVisualizerCfg(
            eye=camera_cfg.eye,
            lookat=camera_cfg.lookat,
            show_particles=True,
            particle_color=MPM_VISUAL_COLOR,
        )
        self.sim.visualizer_cfgs = [NewtonGLVisualizerCfg()]
        self.heightmap_visualizer_cfg = HEIGHTMAP_VISUALIZER_CFG
        configure_sparse_mpm_capacities(self)
