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
from isaaclab_visualizers.newton import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
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
# Newton's granular examples use a jitter range equal to one sample spacing, which
# displaces each point by up to half a cell. Retain a small stratification guard so
# neighboring samples cannot cross cell boundaries during randomized resets.
MPM_RESET_JITTER_FRACTION = 0.45
MPM_VISUAL_COLOR = (0.83, 0.60, 0.22)
PUSH_ACTION_DIM = 6
PUSH_POLICY_OBSERVATION_DIM = 31
PUSH_CRITIC_OBSERVATION_DIM = 11

# These conservative per-world floors retain spread headroom for the capacity-bounded
# rebuildable grid. Small evaluation runs retain 32 total upper nodes even when a linear
# per-world reservation would be smaller.
SPARSE_MPM_MIN_LOWER_NODES_PER_WORLD = 1 << 6
SPARSE_MPM_MIN_UPPER_NODES_PER_WORLD = 1
SPARSE_MPM_MIN_TOTAL_UPPER_NODE_COUNT = 1 << 5

# All task geometry is specified once and reused for the visible MPM colliders, the hidden rigid
# collision mirror, paddle safety checks, and tests. This prevents a visually plausible bin from
# silently disagreeing with either subsolver.
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
# A compact square blade cannot cover the nominal 0.40 m-wide pile in one pass, so deployment
# requires lateral correction while the blade still spans every randomized pile height.
PADDLE_SIZE = (0.24, 0.24, 0.04)
PADDLE_OFFSET = (0.5 * PADDLE_SIZE[0] + 0.02, 0.0, 0.0)
PADDLE_MASS = 1.0
PADDLE_CONTACT_MARGIN = 0.4 * MPM_VOXEL_SIZE
# The vertical blade starts with its lower edge exactly one table-plus-paddle MPM margin above the
# work surface. Newton MPM consumes shape margins but not rigid contact gaps, so the collision
# bands meet without overlap while the blade engages the complete supported pile.
# Reset roughly 70 mm behind the nominal trailing pile face. This removes an uninformative
# empty-space reach while leaving randomized clearance before first particle contact.
PADDLE_RESET_CENTER = (
    0.36,
    0.0,
    0.5 * PADDLE_SIZE[0] + MPM_COLLIDER_MARGIN + PADDLE_CONTACT_MARGIN,
)
PILE_NOMINAL_CENTER = (0.60, 0.0, 0.0)

# Two material samples per grid-cell axis retain useful sub-voxel quadrature without paying the
# 3.375x particle cost of three samples per axis. A broad, shallow 24 x 32 x 8 lattice holds
# 6,144 particles: essentially the former particle cost, but with a realistic sweep footprint.
# Every reset repacks this same material volume. The volume-equivalent radius keeps Newton's
# ``8 * radius**3`` volume consistent with each sample.
PILE_LOCAL_SIZE = (0.299999, 0.399999, 0.099999)
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

    # This is the same SeattleLab workbench placement used by Isaac Lab's manipulation tasks.
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
    # The stock gains visibly yield under a broad granular blade. Preserve the official USD,
    # inertia, joint limits, and effort cap while matching a stiff industrial position loop.
    robot.actuators["arm"].stiffness = 2400.0
    robot.actuators["arm"].damping = 70.0
    # A real UR controller compensates gravity while tracking joint-position targets. Author the
    # equivalent MuJoCo property declaratively instead of injecting task-level effort commands.
    robot.spawn.joint_drive_props = [MujocoJointCfg(actuatorgravcomp=True)]

    # A mass-bearing paddle is welded to the official UR10's terminal rigid body. Its local X
    # dimension is vertical at the reset pose and local Z is the pushing normal. Keep collision
    # and visual geometry separate so standalone viewers can hide the collider without hiding the
    # visible tool.
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
            # Cell-centered emission avoids the boundary overcount in Newton's historical public
            # examples. The lowest centers lie exactly on the expanded table support plane.
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
                # Use a realistic lightweight-pellet bulk density for safe UR10 manipulation.
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

    # These offsets let the stiff position loop reach the UR10 effort limit without commanding a
    # discontinuous absolute pose. The action term snapshots one target per policy step.
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
    paddle_reach = RewTerm(func=mdp.paddle_reach, weight=0.005)
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
    """Reset the robot and fixed granular payload after curriculum selection."""

    reset_scene = EventTerm(func=mdp.reset_push_scene, mode="reset")


@configclass
class CurriculumCfg:
    """Outcome-driven reverse curriculum over the reset families."""

    task_levels = CurrTerm(func=mdp.PushCurriculum)


@configclass
class UR10ParticlePushEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based, relative-joint-control UR10 particle-pushing task."""

    decimation = 2
    # An 18 s horizon leaves time for approach, multiple corrective sweeps, and the success dwell
    # while producing curriculum outcomes substantially more often than the former 30 s episodes.
    episode_length_s = 18.0
    # Treat the deadline as a training truncation. The actor has no remaining-time observation,
    # so assigning a hidden finite-horizon terminal value would make the value function non-Markov.
    is_finite_horizon = False
    # A 20 mm image grid resolves the 25 mm MPM grid over the 1.00 m y by 1.71 m x
    # camera workspace. Channels contain the current and four-policy-step-old surfaces plus a
    # fixed bin-region mask that is reproducible from camera calibration.
    heightmap_shape: tuple[int, int] = (50, 86)
    heightmap_history_steps: int = 4
    scene: UR10ParticlePushSceneCfg = UR10ParticlePushSceneCfg(
        # Runtime count should be selected from measured sparse-MPM throughput and memory headroom.
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
    # Pack the fixed particle population into visibly different, volume-preserving shapes. Each
    # profile is ``(vertical cells, footprint X/Y aspect ratio)``; sample spacing, material volume,
    # density, and PPC remain unchanged. Playback cycles profiles, while training samples them.
    reset_source_shape_profiles: tuple[tuple[int, float], ...] = (
        (12, 0.40),  # tall, laterally wide
        (10, 0.75),  # compact mound
        (8, 0.75),  # nominal broad pile
        (8, 1.50),  # deep, laterally narrow
        (10, 2.00),  # compact deep ridge
    )
    # Particle transforms are sampled continuously; a small startup-only pose bank supplies
    # canonical-branch, collision-screened robot starts.
    reset_seed: int = 17
    reset_pose_count: int = 200
    reset_cycle: bool = False
    # Optional deterministic curriculum-level sequence for inspection and evaluation. Training
    # leaves this disabled and advances levels only from completed-episode outcomes.
    reset_curriculum_level_cycle: tuple[int, ...] | None = None
    # Use asymmetric longitudinal bounds: deployment material can begin much farther from the bin,
    # while the positive bound keeps it clear of the front-wall coupling margin. Single piles can
    # use more table width than split piles, whose fixed side-to-side separation already provides
    # substantial lateral variation.
    reset_particle_longitudinal_offset_range: tuple[float, float] = (-0.18, 0.08)
    reset_particle_max_lateral_offset: float = 0.20
    reset_particle_split_max_lateral_offset: float = 0.12
    reset_particle_max_yaw: float = 0.30
    # Randomize each point through 90 percent of its lattice cell (up to 45 percent of the sample
    # spacing in either direction). This matches the scale used by Newton's granular examples while
    # keeping samples in their original strata. The support correction keeps the lowest point on
    # the table margin.
    reset_particle_jitter: float = MPM_RESET_JITTER_FRACTION * min(PILE_LATTICE_CELL_SIZE)
    reset_paddle_longitudinal_offset_range: tuple[float, float] = (-0.12, 0.06)
    reset_paddle_max_lateral_offset: float = 0.16
    reset_paddle_split_max_lateral_offset: float = 0.09
    reset_paddle_max_yaw: float = 0.35
    reset_particle_paddle_residual_half_range: tuple[float, float] = (0.06, 0.05)
    reset_ik_seeds: int = 64
    reset_ik_iterations: int = 96
    reset_ik_noise_std: float = 0.25
    reset_ik_max_cost: float = 1.0e-3

    # Reverse curriculum with one invariant 80% terminal objective. Early resets place a settled
    # fraction in the bin and expose only a small residual pile. Middle stages split the residual
    # into left/right piles and alternate the paddle focus, teaching recovery after the first
    # sweep. The last stage exactly restores the single broad deployment pile.
    curriculum_pile_center_x: tuple[float, ...] = (0.80, 0.77, 0.77, 0.74, 0.71, 0.68, 0.65, 0.63, 0.605, 0.60)
    curriculum_initial_bin_fraction: tuple[float, ...] = (0.72, 0.68, 0.64, 0.56, 0.44, 0.32, 0.20, 0.10, 0.0, 0.0)
    curriculum_source_pile_count: tuple[int, ...] = (1, 1, 2, 2, 2, 2, 2, 2, 2, 1)
    curriculum_source_lateral_offset: tuple[float, ...] = (0.0, 0.0, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.0)
    curriculum_randomization_scale: tuple[float, ...] = (0.40, 0.48, 0.56, 0.64, 0.72, 0.79, 0.85, 0.91, 0.96, 1.00)
    curriculum_successes_to_promote: int = 2
    curriculum_failures_to_demote: int = 3
    curriculum_level_override: int | None = None
    # Keep the leading particle safely behind the expanded bin-front collision band.
    reset_bin_clearance: float = 0.01

    # Segmented overhead-depth adapter. These features can be reproduced from a calibrated real
    # overhead camera without exposing simulator-only particle identities to the actor.
    heightmap_x_bounds: tuple[float, float] = (-0.25, 1.46)
    heightmap_y_bounds: tuple[float, float] = (-0.50, 0.50)
    # Preserve contrast across the physical bin floor, table pile, and ordinary airborne grains.
    heightmap_z_min: float = -0.20
    heightmap_z_range: float = 0.38
    heightmap_depth_noise_std: float = 0.004
    heightmap_xy_noise_std: float = 0.003
    heightmap_dropout_probability: float = 0.01

    bin_inner_x_bounds: tuple[float, float] = BIN_INNER_X_BOUNDS
    bin_inner_y_bounds: tuple[float, float] = BIN_INNER_Y_BOUNDS
    # Delivery includes ordinary compliant contact down to the protected floor core so a
    # compressed bottom layer remains part of the successful payload.
    bin_inner_z_bounds: tuple[float, float] = (-0.186, 0.09)
    bin_physical_z_bounds: tuple[float, float] = (-0.186, 0.12)
    success_fraction: float = 0.80
    success_max_spill_fraction: float = 0.02
    # Success remains strict, while only a clearly unrecoverable spill ends exploration.
    failure_max_spill_fraction: float = 0.20
    # Settled-quality reference for privileged diagnostics and the physical validator. Terminal
    # success uses sustained bin occupancy instead: global RMS includes the undelivered source
    # pile and would incorrectly veto partial-curriculum deliveries.
    success_max_rms_particle_speed: float = 0.12
    success_dwell_time_s: float = 0.30
    particle_workspace_lower_bound: tuple[float, float, float] = (-0.30, -0.60, -0.35)
    particle_workspace_upper_bound: tuple[float, float, float] = (1.50, 0.60, 0.80)
    # Match the physical 2% success spill tolerance. This is a sparse-topology safety bound rather
    # than an objective-reachability condition: retaining runaway particles can expand the
    # NanoVDB hierarchy indefinitely.
    max_escaped_particle_fraction: float = 0.02
    # Bound numerical velocity outliers before they can cross many sparse-grid regions.
    particle_max_velocity: float = 10.0
    # Reset extreme-but-finite rigid states before they can poison policy/value inputs. These are
    # numerical-instability bounds, deliberately much looser than the robot's normal motion.
    state_bound_joint_position_margin: float = 0.05
    state_bound_max_joint_velocity: float = 20.0
    state_bound_max_ee_linear_velocity: float = 10.0
    state_bound_max_ee_angular_velocity: float = 50.0
    paddle_reach_distance: float = 0.15
    # Saturate only after the blade has crossed two MPM voxels into the nominal pile envelope,
    # ensuring the exploration bridge includes first contact rather than stopping just before it.
    paddle_reach_contact_depth: float = 0.04
    paddle_lateral_alignment_distance: float = 0.25
    paddle_vertical_alignment_distance: float = 0.08
    # Active sparse-grid capacity follows occupied solver cells, not the raw 6,144-particle count.
    # The 3,072-cell reservation retains measured PIC27 spread headroom without using a dense grid.
    mpm_active_cell_count_per_world: int = 3072
    mpm_leaf_node_count_per_world: int = 1 << 9
    mpm_lower_node_count_per_world: int = SPARSE_MPM_MIN_LOWER_NODES_PER_WORLD
    mpm_upper_node_count_per_world: int = SPARSE_MPM_MIN_UPPER_NODES_PER_WORLD
    # Scale only the virtual paddle inertia inside MPM to limit proxy acceleration under granular
    # loading; the rigid solver retains the authored 1 kg tool and receives the harvested wrench.
    proxy_mass_scale: float = 10.0

    def _validate_source_shape_profiles(self) -> None:
        """Validate source-pile profiles before any derived bounds consume them."""
        if not self.reset_source_shape_profiles or any(
            len(profile) != 2 for profile in self.reset_source_shape_profiles
        ):
            raise ValueError("Source shape profiles must contain vertical cells and an aspect ratio.")
        if any(
            vertical_cells < 1 or not math.isfinite(aspect_ratio) or aspect_ratio <= 0.0
            for vertical_cells, aspect_ratio in self.reset_source_shape_profiles
        ):
            raise ValueError("Source shape profiles must contain positive finite values.")

    def _validate_reset_config(self) -> None:
        """Validate fixed-payload and canonical pose-bank resets."""
        particle_radius = self.scene.media.spawn.radius
        if particle_radius is None or not math.isfinite(particle_radius) or particle_radius <= 0.0:
            raise ValueError("UR10 Push requires an explicit positive particle radius.")
        if self.reset_pose_count < 1:
            raise ValueError("reset_pose_count must be positive.")

        centers = (*self.paddle_reset_center, *self.pile_nominal_center)
        if (
            len(self.paddle_reset_center) != 3
            or len(self.pile_nominal_center) != 3
            or not all(math.isfinite(value) for value in centers)
        ):
            raise ValueError("Paddle and pile reset centers must contain three finite coordinates.")
        longitudinal_ranges = (
            self.reset_particle_longitudinal_offset_range,
            self.reset_paddle_longitudinal_offset_range,
        )
        if any(
            len(bounds) != 2 or not all(math.isfinite(value) for value in bounds) or not bounds[0] <= 0.0 <= bounds[1]
            for bounds in longitudinal_ranges
        ):
            raise ValueError("Reset longitudinal ranges must be finite ordered intervals containing zero.")
        if len(self.reset_particle_paddle_residual_half_range) != 2:
            raise ValueError("reset_particle_paddle_residual_half_range must contain two values.")
        randomization_magnitudes = (
            self.reset_particle_max_lateral_offset,
            self.reset_particle_split_max_lateral_offset,
            self.reset_particle_max_yaw,
            self.reset_particle_jitter,
            self.reset_paddle_max_lateral_offset,
            self.reset_paddle_split_max_lateral_offset,
            self.reset_paddle_max_yaw,
            *self.reset_particle_paddle_residual_half_range,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in randomization_magnitudes):
            raise ValueError("Reset randomization magnitudes must be finite and non-negative.")
        if (
            self.reset_particle_longitudinal_offset_range[0] > self.reset_paddle_longitudinal_offset_range[0]
            or self.reset_particle_longitudinal_offset_range[1] < self.reset_paddle_longitudinal_offset_range[1]
            or self.reset_particle_max_lateral_offset < self.reset_paddle_max_lateral_offset
            or self.reset_particle_split_max_lateral_offset < self.reset_paddle_split_max_lateral_offset
        ):
            raise ValueError("Particle translation ranges must contain the paddle translation ranges.")
        if max(self.reset_particle_max_yaw, self.reset_paddle_max_yaw) >= 0.5 * math.pi:
            raise ValueError("Reset yaw magnitudes must be smaller than pi/2.")
        if self.reset_particle_jitter >= 0.5 * min(PILE_LATTICE_CELL_SIZE):
            raise ValueError("Particle reset jitter must be smaller than half the particle spacing.")

        self._validate_curriculum_config()
        paddle_size = tuple(self.paddle_size)
        if (
            tuple(self.scene.paddle.spawn.size) != paddle_size
            or tuple(self.scene.paddle_visual.spawn.size) != paddle_size
        ):
            raise ValueError("paddle_size must match both authored paddle geometries.")
        maximum_lateral_width = PILE_LATTICE_RESOLUTION[1] * PILE_LATTICE_CELL_SIZE[1]
        if paddle_size[1] >= maximum_lateral_width:
            raise ValueError("The paddle must remain narrower than the maximum pile.")
        if min(self.reset_ik_seeds, self.reset_ik_iterations) < 1:
            raise ValueError("Reset IK seed and iteration counts must be positive.")
        if any(not math.isfinite(value) or value <= 0.0 for value in (self.reset_ik_max_cost, self.reset_ik_noise_std)):
            raise ValueError("Reset IK cost and noise thresholds must be finite and positive.")
        if (
            not math.isfinite(self.actions.arm_action.joint_limit_margin)
            or self.actions.arm_action.joint_limit_margin < 0.0
        ):
            raise ValueError("The arm action joint-limit margin must be finite and non-negative.")

    def _validate_curriculum_config(self) -> None:
        """Validate the reverse curriculum and collision-safe reset envelope."""
        level_count = len(self.curriculum_pile_center_x)
        if level_count < 2 or any(
            len(values) != level_count
            for values in (
                self.curriculum_initial_bin_fraction,
                self.curriculum_source_pile_count,
                self.curriculum_source_lateral_offset,
                self.curriculum_randomization_scale,
            )
        ):
            raise ValueError("Every curriculum sequence must contain one value per level.")
        if self.reset_pose_count % level_count != 0:
            raise ValueError("reset_pose_count must be divisible by the number of curriculum levels.")
        if not all(math.isfinite(center_x) for center_x in self.curriculum_pile_center_x):
            raise ValueError("Curriculum pile centers must be finite.")
        if not all(0.0 <= value < self.success_fraction for value in self.curriculum_initial_bin_fraction):
            raise ValueError("Curriculum initial bin fractions must lie in [0, success_fraction).")
        if not all(0.0 < value <= 1.0 for value in self.curriculum_randomization_scale):
            raise ValueError("Curriculum randomization scales must lie in (0, 1].")
        rows_per_level = self.reset_pose_count // level_count
        for level, (pile_count, lateral_offset) in enumerate(
            zip(self.curriculum_source_pile_count, self.curriculum_source_lateral_offset, strict=True)
        ):
            if pile_count not in (1, 2):
                raise ValueError(f"Curriculum level {level} must contain one or two source piles.")
            if rows_per_level % pile_count != 0:
                raise ValueError(f"Reset-pose rows at curriculum level {level} must divide across its source piles.")
            if not lateral_offset >= 0.0 or (pile_count == 1) != math.isclose(lateral_offset, 0.0):
                raise ValueError(
                    "Single source piles require zero lateral offset; split piles require a positive offset."
                )
        if min(self.curriculum_successes_to_promote, self.curriculum_failures_to_demote) < 1:
            raise ValueError("Curriculum promotion and demotion streaks must be positive.")
        if self.curriculum_level_override is not None and not 0 <= self.curriculum_level_override < level_count:
            raise ValueError(f"curriculum_level_override must be None or lie in [0, {level_count - 1}].")
        if self.reset_curriculum_level_cycle is not None:
            if self.curriculum_level_override is not None:
                raise ValueError("reset_curriculum_level_cycle and curriculum_level_override are mutually exclusive.")
            if not self.reset_curriculum_level_cycle or any(
                not 0 <= level < level_count for level in self.reset_curriculum_level_cycle
            ):
                raise ValueError(f"reset_curriculum_level_cycle entries must lie in [0, {level_count - 1}].")
        if not self.reset_bin_clearance >= 0.0:
            raise ValueError("reset_bin_clearance must be non-negative.")
        self._validate_particle_reset_envelopes()

    def _validate_particle_reset_envelopes(self) -> None:
        """Keep every staged particle group inside its physical support geometry."""
        # Bound the compact source group generated at each level, including global translation,
        # yaw, and cell-local jitter. Source material must remain on the table and behind the
        # expanded front-wall collider; delivered material is checked against the bin interior.
        particle_count = math.prod(PILE_LATTICE_RESOLUTION)
        safe_front_x = BIN_FRONT_POSITION[0] - 0.5 * BIN_FRONT_SIZE[0] - MPM_COLLIDER_MARGIN - self.reset_bin_clearance
        safe_table_min_x = WORK_SURFACE_POSITION[0] - 0.5 * WORK_SURFACE_SIZE[0] + MPM_COLLIDER_MARGIN
        safe_table_half_width = 0.5 * WORK_SURFACE_SIZE[1] - MPM_COLLIDER_MARGIN
        for level, (center_x, delivered_fraction, pile_count, lateral_offset, scale) in enumerate(
            zip(
                self.curriculum_pile_center_x,
                self.curriculum_initial_bin_fraction,
                self.curriculum_source_pile_count,
                self.curriculum_source_lateral_offset,
                self.curriculum_randomization_scale,
                strict=True,
            )
        ):
            delivered_count = math.floor(particle_count * delivered_fraction)
            maximum_group_count = math.ceil((particle_count - delivered_count) / pile_count)
            yaw = self.reset_particle_max_yaw * scale
            shape_half_extents = tuple(
                _packed_group_half_extent(
                    maximum_group_count,
                    vertical_cell_count,
                    source_aspect_ratio,
                    yaw,
                    self.reset_particle_jitter,
                )
                for vertical_cell_count, source_aspect_ratio in self.reset_source_shape_profiles
            )
            maximum_half_x = max(extent[0] for extent in shape_half_extents)
            maximum_half_y = max(extent[1] for extent in shape_half_extents)
            trailing_x = center_x + self.reset_particle_longitudinal_offset_range[0] * scale - maximum_half_x
            if trailing_x < safe_table_min_x:
                raise ValueError(
                    f"Curriculum level {level} can spawn particles beyond the rear work-surface edge "
                    f"({trailing_x:.6f} < {safe_table_min_x:.6f})."
                )
            leading_x = center_x + self.reset_particle_longitudinal_offset_range[1] * scale + maximum_half_x
            if leading_x > safe_front_x:
                raise ValueError(
                    f"Curriculum level {level} can spawn particles in the bin-front contact band "
                    f"({leading_x:.6f} > {safe_front_x:.6f})."
                )
            lateral_range = (
                self.reset_particle_split_max_lateral_offset
                if pile_count == 2
                else self.reset_particle_max_lateral_offset
            )
            lateral_reach = lateral_offset + lateral_range * scale + maximum_half_y
            if lateral_reach > safe_table_half_width:
                raise ValueError(
                    f"Curriculum level {level} can spawn particles beyond the work surface "
                    f"({lateral_reach:.6f} > {safe_table_half_width:.6f})."
                )

        maximum_delivered_count = math.floor(particle_count * max(self.curriculum_initial_bin_fraction))
        bin_aspect_ratio = (self.bin_inner_x_bounds[1] - self.bin_inner_x_bounds[0]) / (
            self.bin_inner_y_bounds[1] - self.bin_inner_y_bounds[0]
        )
        delivered_half_x, delivered_half_y = _packed_group_half_extent(
            maximum_delivered_count,
            PILE_LATTICE_RESOLUTION[2],
            bin_aspect_ratio,
            0.0,
            self.reset_particle_jitter,
        )
        if delivered_half_x >= 0.5 * (
            self.bin_inner_x_bounds[1] - self.bin_inner_x_bounds[0]
        ) or delivered_half_y >= 0.5 * (self.bin_inner_y_bounds[1] - self.bin_inner_y_bounds[0]):
            raise ValueError("The largest staged delivered pile must fit inside the physical bin.")

    def _validate_heightmap_config(self) -> None:
        """Validate the deployable image observation and calibrated goal mask."""
        if len(self.heightmap_shape) != 2 or any(size < 8 for size in self.heightmap_shape):
            raise ValueError("heightmap_shape must contain two integer dimensions of at least eight cells.")
        if self.heightmap_history_steps < 1:
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
        height_cells, width_cells = self.heightmap_shape
        cell_size_x = (self.heightmap_x_bounds[1] - self.heightmap_x_bounds[0]) / width_cells
        cell_size_y = (self.heightmap_y_bounds[1] - self.heightmap_y_bounds[0]) / height_cells
        if max(cell_size_x, cell_size_y) > 1.05 * self.scene.media.spawn.voxel_size:
            raise ValueError("The heightmap must resolve the particle surface at approximately the MPM voxel scale.")
        if max(cell_size_x, cell_size_y) / min(cell_size_x, cell_size_y) > 1.05:
            raise ValueError("The heightmap cells must be approximately square.")
        for axis_name, map_bounds, bin_bounds, cell_count in (
            ("x", self.heightmap_x_bounds, self.bin_inner_x_bounds, width_cells),
            ("y", self.heightmap_y_bounds, self.bin_inner_y_bounds, height_cells),
        ):
            cell_size = (map_bounds[1] - map_bounds[0]) / cell_count
            if not any(
                bin_bounds[0] <= map_bounds[0] + (index + 0.5) * cell_size <= bin_bounds[1]
                for index in range(cell_count)
            ):
                raise ValueError(f"The bin must cover at least one heightmap cell center along {axis_name}.")
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

    def _validate_objective_config(self) -> None:
        """Validate cross-field objective ranges and observation divisors."""
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
            self.paddle_reach_distance,
            self.paddle_lateral_alignment_distance,
            self.paddle_vertical_alignment_distance,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in observation_divisors):
            raise ValueError("Particle-speed and paddle-alignment scales must be finite and positive.")
        nominal_pile_length = self.scene.media.spawn.upper[0] - self.scene.media.spawn.lower[0]
        if not 0.0 <= self.paddle_reach_contact_depth < 0.5 * nominal_pile_length:
            raise ValueError("Paddle reach contact depth must lie within the trailing half of the pile.")
        reward_values = (
            self.rewards.success.weight,
            self.rewards.bin_progress.weight,
            self.rewards.transport_progress.weight,
            self.rewards.paddle_reach.weight,
            self.rewards.spill.weight,
            self.rewards.failure.weight,
            self.rewards.action_magnitude.weight,
            self.rewards.action_rate.weight,
        )
        if not all(math.isfinite(value) for value in reward_values):
            raise ValueError("Every reward and penalty coefficient must be finite.")

    def __post_init__(self) -> None:
        from isaaclab_visualizers.kit import KitVisualizerCfg

        self._validate_source_shape_profiles()
        self._validate_heightmap_config()
        self._validate_reset_config()
        self._validate_objective_config()
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
                        # Match Pour's stable rigid refinement inside each coupled interval.
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
                            # Match Newton's granular rheology path. Particle-backed warm-starting
                            # preserves constitutive history on a rebuildable sparse grid. A loaded
                            # deterministic sweep validates 24 iterations without collider crossings or
                            # material delivery regression while avoiding an unnecessary solve tail.
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
                        # Match Pour: entry-local repeats reuse a stale proxy pose, so moving
                        # collider temporal resolution belongs in the outer coupled substeps.
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
            # One coupled solve per 120 Hz simulation step is sufficient for the coarse granular
            # media. Loaded slow and aggressive sweep regressions remain crossing-free at this
            # cadence, while a second outer solve nearly doubles collection time.
            num_substeps=1,
            # Fixed particle topology makes reset-time state writes CUDA-graph safe.
            use_cuda_graph=True,
        )
        # Keep the UR10's implicit drives on the Newton backend used by the coupled step.
        self.sim.use_newton_actuators = True
        configure_sparse_mpm_capacities(self)

    def play_mode(self) -> None:
        """Cycle the two hardest trained reset families in a small playback scene."""
        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 4)
        self.heightmap_depth_noise_std = 0.0
        self.heightmap_xy_noise_std = 0.0
        self.heightmap_dropout_probability = 0.0
        self.reset_cycle = True
        final_level = len(self.curriculum_pile_center_x) - 1
        self.curriculum_level_override = None
        # The penultimate level contains two separated source piles with an empty bin; the final
        # level contains one broad deployment pile. Both retain near-maximum/full reset
        # randomization and the same 80% terminal objective used during training.
        self.reset_curriculum_level_cycle = (final_level - 1, final_level)
        # Newton's generic viewer defaults to hidden particles. Enable them only for playback so
        # training remains renderer-free unless a visualizer is explicitly requested.
        self.sim.visualizer_cfgs = [
            NewtonVisualizerCfg(
                show_particles=True,
                particle_color=MPM_VISUAL_COLOR,
            )
        ]
        configure_sparse_mpm_capacities(self)
