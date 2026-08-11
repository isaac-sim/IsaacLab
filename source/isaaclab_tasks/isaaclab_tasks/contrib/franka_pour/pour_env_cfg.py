# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka grasp-a-cup-of-MPM-media-and-pour, on the stable Isaac-Lift-Cube-Franka foundation.

Scene assets are borrowed from the lift task (standard Franka + the SeattleLab table USD) for a
stable, familiar base. On top we add a coupled Newton solver with **proxy coupling**:

* an MJWarp ``arm`` entry owns the robot, the dynamic source cup, and the fixed receiver, and
* an implicit ``media`` entry owns the MPM particles.

The source cup is a real dynamic rigid body resting on the table: the Franka grasps it with its fingers
through Newton-generated friction contacts resolved by MJWarp, and a Newton proxy mapping exposes
both cups' ``COLLIDE_PARTICLES`` cavity meshes to the MPM solver as auto-pose-synced colliders.
This replaces the earlier welded-kinematic-cup design.

The source cup carries two co-located shapes on the same body: a solid grasp box (``COLLIDE_SHAPES``,
arm-entry-only) the fingers can actually grip, and a hollow cavity mesh (``COLLIDE_PARTICLES``) the
proxy bridges to MPM. Both learning variants use relative joint commands; the reset-dataset variant
uses a binary symmetric gripper.
"""

from __future__ import annotations

import math
from copy import deepcopy

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
)
from isaaclab_newton.sim.schemas import MujocoJointCfg
from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas import MassCfg, UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab_tasks.utils.adaptive_reset_sampler import AdaptiveResetSamplerCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from . import mdp
from .cube_bowl_mesh import cube_bowl_inner_bounds
from .cube_bowl_spawner_cfg import CubeBowlSpawnerCfg
from .cup_media import build_media_object_cfg, cup_cavity_lattice

RIGID_ENTRY = "arm"
MPM_ENTRY = "media"
FRANKA_POUR_ROBOT_USD_PATH = "omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/Robots/FrankaEmika/franka_panda.usda"
FRANKA_POUR_ARM_COLLISION_PROXIES = frozenset(
    {
        "link0_c",
        "link1_c",
        "link2_c",
        "link3_c",
        "link4_c",
        "link5_c0",
        "link5_c1",
        "link5_c2",
        "link6_c",
        "link7_c",
    }
)
SPILL_FLOOR_LABEL_PATTERN = r".*/SpillFloor$"
# Coupler body selectors use full Newton body-label regexes (not SceneEntityCfg).
ROBOT_BODY_LABEL_PATTERN = r"/World/envs/env_.*/Robot"
SOURCE_CUP_BODY_LABEL_PATTERN = r"/World/envs/env_.*/SourceCup"
TARGET_CUP_BODY_LABEL_PATTERN = r"/World/envs/env_.*/TargetCup"
GRASP_APPROACH_STAGE_NAMES = (
    "approach_1",
    "approach_2",
    "approach_3",
    "approach_4",
    "approach_5",
    "approach_6",
)
CURRICULUM_STAGE_NAMES = (
    "drain",
    "deep_tilt",
    "tilt",
    "pour",
    "near_carry",
    "mid_carry",
    "carry",
    "grasp",
    *GRASP_APPROACH_STAGE_NAMES,
    "full",
    "randomized",
)


def spawn_franka_with_arm_collisions(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the canonical Franka and activate only its dedicated arm collision proxies."""
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    robot_prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for root_prim in sim_utils.find_matching_prims(prim_path, stage=robot_prim.GetStage()):
        self_collision = root_prim.GetAttribute("newton:selfCollisionEnabled")
        if not self_collision or not self_collision.Set(True):
            raise RuntimeError(f"Franka asset at {root_prim.GetPath()} has no writable Newton self-collision flag.")
        proxy_roots = {
            prim.GetName(): prim.GetParent().GetPath()
            for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies())
            if prim.GetName() in FRANKA_POUR_ARM_COLLISION_PROXIES and prim.GetParent().GetName() == prim.GetName()
        }
        if proxy_roots.keys() != FRANKA_POUR_ARM_COLLISION_PROXIES:
            missing = sorted(FRANKA_POUR_ARM_COLLISION_PROXIES.difference(proxy_roots))
            raise RuntimeError(f"Franka asset at {root_prim.GetPath()} is missing collision proxies: {missing}.")
        for proxy_root in proxy_roots.values():
            root_prim.GetStage().OverridePrim(proxy_root).SetInstanceable(False)
        for proxy_root in proxy_roots.values():
            collision_prim = root_prim.GetStage().GetPrimAtPath(proxy_root.AppendChild(proxy_root.name))
            UsdPhysics.CollisionAPI(collision_prim).GetCollisionEnabledAttr().Set(True)
    return robot_prim


PANDA_ARM_JOINT_LIMITS = (
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
)


def _mpm_solver_cfg(cfg: FrankaPourEnvCfg) -> MPMSolverCfg:
    """Return the task's unique implicit-MPM solver config."""
    entries = [entry for entry in cfg.sim.physics.solver_cfg.entries if entry.name == MPM_ENTRY]
    if len(entries) != 1:
        raise ValueError(f"Expected exactly one {MPM_ENTRY!r} solver entry, found {len(entries)}.")
    return entries[0].solver_cfg


def _resolve_mpm_cell_cap(cfg: FrankaPourEnvCfg) -> int:
    """Resolve the total MPM active-cell capacity without mutating ``cfg``.

    Sparse training reserves an aligned hard upper bound per independent world so Newton can
    capture topology rebuilds. Fixed and dense grids retain their configured capacity unless an
    explicit total override is provided.

    Returns:
        The total capacity to assign to the MPM solver entry.
    """
    solver_cfg = _mpm_solver_cfg(cfg)
    override = cfg.mpm_cell_cap_override
    if override is not None:
        capacity = int(override)
    elif solver_cfg.grid_type == "sparse":
        alignment = int(cfg.mpm_cell_capacity_alignment)
        if alignment <= 0:
            raise ValueError(f"Franka Pour MPM cell-capacity alignment must be positive, got {alignment}.")
        particle_count = int(cup_cavity_lattice(cfg)[0].shape[0])
        per_world = ((particle_count + alignment - 1) // alignment) * alignment
        capacity = per_world * int(cfg.scene.num_envs)
    else:
        capacity = int(solver_cfg.max_active_cell_count)

    if capacity <= 0:
        raise ValueError(f"Franka Pour MPM capacity must be positive, got {capacity}.")
    return capacity


@configclass
class PourSceneCfg(InteractiveSceneCfg):
    """Lift-task scene assets plus resolved cups and MPM media."""

    # SeattleLab table (top at env z=0), exactly as the Isaac-Lift-Cube-Franka scene.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0, 0, 0.707, 0.707]),  # xyzw, matches Lift
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
    )
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.usd_path = FRANKA_POUR_ROBOT_USD_PATH
    robot.spawn.func = spawn_franka_with_arm_collisions
    # The task-specific spawner overrides the source USD's Newton-native value; retain the matching
    # backend-independent intent in the articulation configuration as well.
    robot.spawn.articulation_props.enabled_self_collisions = True
    # Resolve the implicit-actuator parameters from the requested USD rather than overwriting them
    # with Isaac Lab's generic Franka gains and limits.
    robot.actuators = {
        name: actuator_cfg.replace(
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness=None,
            damping=None,
            armature=None,
        )
        for name, actuator_cfg in robot.actuators.items()
    }
    robot.spawn.joint_drive_props = [MujocoJointCfg(actuatorgravcomp=True)]
    # Built by :meth:`FrankaPourEnvCfg.finalize` from the final override values.
    source_cup: RigidObjectCfg | None = None
    target_cup: RigidObjectCfg | None = None
    media: MPMObjectCfg | None = None


@configclass
class ActionsCfg:
    """Relative arm-joint increments and one continuous symmetric-gripper command."""

    arm_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[f"panda_joint{i}" for i in range(1, 8)],
        preserve_order=True,
        # Eight-hundredths of a radian per policy step gives useful reach authority without
        # bypassing the articulation position drives or encoding a demonstrated trajectory.
        scale=0.08,
        use_zero_offset=True,
    )
    gripper_action = mdp.CurriculumGripperPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        # Zero action holds the contact-safe preload. Negative actions close farther and positive
        # actions continuously open the fingers, so the policy—not a phase interlock—owns grasping.
        scale=0.016,
        alpha=0.2,
        close_position=0.021,
        neutral_position=0.04,
        open_position=0.04,
        default_position=0.024,
        limit_to_preload=False,
        force_open_before_phase_stage=-1,
        # The lift gate still requires persistent bilateral deflection and actual cup motion. A
        # 5 cm/s finger-settling threshold avoids spending most of a five-second attempt waiting
        # for sub-millimetre drive oscillations to decay.
        contact_max_velocity=0.05,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Sensor-compatible robot, gripper, and cup geometry available to the actor."""

        arm_q = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
            scale=0.3,
        )
        arm_qd = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
            scale=0.05,
        )
        time_remaining = ObsTerm(func=mdp.time_remaining_obs)
        pour_target_fraction = ObsTerm(func=mdp.pour_target_fraction_obs)
        tcp_pose = ObsTerm(func=mdp.tcp_pose_obs)
        cup_pose = ObsTerm(func=mdp.cup_pose_obs)
        target_pose = ObsTerm(func=mdp.target_pose_obs)
        tcp_to_grasp_position_c = ObsTerm(func=mdp.tcp_to_grasp_position_c_obs, scale=10.0)
        grasp_to_tcp_quat = ObsTerm(func=mdp.grasp_to_tcp_quat_obs)
        target_position_c = ObsTerm(func=mdp.target_position_c_obs, scale=5.0)
        finger_position = ObsTerm(func=mdp.finger_position_obs, scale=25.0)
        finger_velocity = ObsTerm(func=mdp.finger_velocity_obs, scale=5.0)
        gripper_target = ObsTerm(func=mdp.gripper_target_obs, scale=25.0)
        gripper_contact = ObsTerm(func=mdp.gripper_contact_obs, scale=250.0)
        last_action = ObsTerm(func=mdp.last_action, scale=0.2)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Exact simulation state available only to the asymmetric critic."""

        success_dwell = ObsTerm(func=mdp.success_dwell_obs)
        lost_grasp_dwell = ObsTerm(func=mdp.lost_grasp_dwell_obs)
        cup_velocity = ObsTerm(func=mdp.cup_velocity_obs, scale=0.1)
        particle_fractions = ObsTerm(func=mdp.particle_fractions_obs)
        particle_transfer = ObsTerm(func=mdp.particle_transfer_obs)
        held_delivery_history = ObsTerm(func=mdp.held_delivery_history_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class RewardsCfg:
    # One discounted hierarchical physical potential spans every backward-curriculum reset. It is
    # policy-invariant at PPO's gamma, so holding or wiggling cannot improve discounted return.
    task_progress = RewTerm(
        func=mdp.PourTaskProgress,
        weight=5.0,
        params={
            "target_height": 0.12,
            "reach_std": 0.07,
            "grasp_reach_std": 0.015,
            "grasp_preload_position": 0.024,
            "lift_height": 0.06,
            "align_std": 0.12,
            "source_offset_xy": (0.0, 0.05),
            "target_tilt": math.radians(140.0),
            "pour_direction_xy": (0.0, -1.0),
            "source_mouth_height": 0.099,
            "alignment_radius": 0.15,
            # Tilt is an exploration bootstrap only for the supplied-grasp stages. Full-task
            # policies are optimized by actual held particle transfer rather than a prescribed pose.
            "active_through_stage": 6,
            "min_lift_height": 0.05,
            "max_tcp_distance": 0.018,
            "max_gripper_width_error": 0.006,
            "max_gripper_command": 0.024,
            # Must match PPO gamma for policy-invariant discounted potential shaping.
            "discount_factor": 0.99,
        },
    )
    # The full-task reset starts outside the narrow reach kernel above. This broad physical-pose
    # potential preserves a Cartesian gradient after premature closure and rewards the cup-relative
    # side-grasp orientation without prescribing a trajectory or exposing a phase variable.
    approach_progress = RewTerm(
        func=mdp.ApproachProgress,
        weight=8.0,
        params={
            "position_std": 0.20,
            "orientation_std": 0.75,
            "open_hand_fraction": 0.35,
            "active_from_stage": 8,
            "discount_factor": 0.99,
        },
    )
    # Once approach reaches the contact neighborhood, distinguish a loaded side grasp from empty
    # closure and retain a monotonic signal until the glass has cleared the table.
    grasp_lift_progress = RewTerm(
        func=mdp.GraspLiftProgress,
        weight=10.0,
        params={
            "target_height": 0.10,
            "grasp_reach_std": 0.025,
            "grasp_preload_position": 0.024,
            "grasp_fraction": 0.40,
            "active_from_stage": 4,
            "discount_factor": 0.99,
        },
    )
    # Signed held-delivery progress is capped at the active success threshold. Particles leaving
    # the receiver repay their credit, and an unsuccessful episode repays any credit still held.
    delivered = RewTerm(
        func=mdp.HeldDeliveryProgress,
        weight=30.0,
        params={
            "min_lift_height": 0.05,
            "max_tcp_distance": 0.018,
            "max_gripper_width_error": 0.006,
            "max_gripper_command": 0.024,
        },
    )
    success = RewTerm(func=mdp.pour_success_bonus, weight=25.0)
    # Airborne transfer is excluded; each particle is penalized once after reaching the table
    # outside both cups. Termination bounds the failure at just over ten percent.
    spill = RewTerm(func=mdp.NewlySpilledParticles, weight=-30.0)
    # Count overlapping failures and an unsuccessful deadline once. This keeps a transient dump
    # that misses the stable-success predicate strictly worse than completing the task.
    failure = RewTerm(func=mdp.terminal_failure, weight=-35.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-4)
    # Penalize unnecessarily large relative arm increments and finger commands without prescribing
    # a reference trajectory.
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.05)


@configclass
class ResetDatasetRewardsCfg:
    """Stage-independent OmniReset-style rewards for reset-dataset training."""

    # Task rewards: generic reach and goal-set distance plus the strict particle success state.
    # The broad kernel remains informative across full-workspace reaching while the reset dataset
    # supplies close-contact precision without adding a task-specific grasp trajectory.
    reach = RewTerm(func=mdp.tcp_cup_distance_tanh, weight=0.1, params={"std": 0.3})
    # Particle distances span only a few decimetres; this preserves useful transfer contrast
    # while retaining the same task-independent tanh form used by OmniReset.
    goal_distance = RewTerm(func=mdp.media_target_distance_tanh, weight=0.1, params={"std": 0.2})
    # A target occupancy of 30% is an immediate terminal success. Dividing the terminal pulse by
    # the policy step keeps its integrated contribution equal to this unit weight.
    success = RewTerm(func=mdp.pour_success_bonus, weight=1.0)

    # Smoothness is one semantic reward group, kept as separate standard terms for diagnostics.
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-1.0e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-3)
    joint_velocity = RewTerm(
        func=mdp.finite_joint_velocity_l2,
        weight=-1.0e-2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
            "max_velocity": 20.0,
        },
    )

    # Ordinary fixed-horizon completion is neutral. ``terminal_failure`` is already normalized by
    # the policy step, so this weight produces one exact -1 abnormal-state pulse per episode.
    failure = RewTerm(func=mdp.terminal_failure, weight=-1.0, params={"include_time_out": False})


@configclass
class TerminationsCfg:
    failure = DoneTerm(func=mdp.nonfinite_failure)
    extreme_rigid_state = DoneTerm(func=mdp.extreme_rigid_state)
    lost_grasp = DoneTerm(
        func=mdp.lost_lifted_grasp,
        params={
            "dwell_time_s": 0.05,
            "max_tcp_distance": 0.018,
            "max_gripper_width_error": 0.006,
            "max_gripper_command": 0.024,
        },
    )
    spill = DoneTerm(func=mdp.excessive_spill)
    particle_out_of_bounds = DoneTerm(func=mdp.particle_out_of_bounds)
    # Success follows every failure predicate, then the custom timeout excludes same-step success.
    success = DoneTerm(
        func=mdp.stable_pour_success,
        params={
            "dwell_time_s": 0.15,
            "min_lift_height": 0.05,
            "max_tcp_distance": 0.018,
            "max_gripper_width_error": 0.006,
            "max_gripper_command": 0.024,
        },
    )
    time_out = DoneTerm(func=mdp.unsuccessful_time_out, time_out=True)


@configclass
class EventsCfg:
    reset_scene = EventTerm(func=mdp.reset_pour_scene, mode="reset")


@configclass
class CurriculumCfg:
    stage = CurrTerm(func=mdp.PourCurriculum)


@configclass
class ResetDatasetCurriculumCfg:
    """Adaptive curriculum over a validated reset-state dataset."""

    reset_dataset = CurrTerm(func=mdp.PourResetDatasetCurriculum)


@configclass
class FrankaPourEnvCfg(ManagerBasedRLEnvCfg):
    """Franka grasping a dynamic cup of MPM media on the lift foundation, proxy-coupled solver."""

    scene: PourSceneCfg = PourSceneCfg(num_envs=2, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # ---- Franka layout / reset (horizontal gripper behind the glass, ready to grasp) ----
    # The fingers approach parallel to the table instead of descending over the rim. This
    # configuration seeds the far-side Newton-IK reset bank; the policy must still acquire physical
    # contact, lift, carry, and pour through direct actions.
    arm_home: tuple[float, float, float, float, float, float, float] = (
        -1.07505691,
        0.76868522,
        0.53213346,
        -2.93226814,
        2.50670838,
        1.40047050,
        0.21146376,
    )
    # Task-space metadata is independent of the policy action representation. SpaceMouse teleop
    # uses the same frame for its input-only IK adapter, while PPO commands joint positions.
    tcp_body_name: str = "panda_hand"
    tcp_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.107)
    tcp_offset_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    gripper_open_pos: float = 0.04  # finger position the cup is grasped from (fingers start open)
    # A target inside the 0.028 m geometric contact position proves active squeeze/preload instead
    # of open fingers passively compressed by cup contact.
    # The zero-action target retained the tall, media-loaded glass in validation without the
    # instability caused by higher friction. The continuous command may close 3 mm farther or open
    # to the full 40 mm finger position.
    # The close target is derived from this value and ``gripper_close_offset`` during finalization.
    gripper_preload_pos: float = 0.024
    gripper_close_offset: float = 0.003

    # ---- source cup (dynamic, grasped by the fingers) ----
    # A 56 x 56 x 119 mm hollow glass leaves 24 mm of clearance in the Panda's 80 mm opening. Its
    # visible outer wall and solid grasp proxy have exactly the same extents. The taller profile
    # leaves 36 mm of wall above the horizontal grasp TCP, so the fingers engage the side without
    # pinching or entering the rim while the hand remains clear of the table.
    source_cup_inner_width: float = 0.042
    source_cup_inner_depth: float = 0.042
    source_cup_cavity_depth: float = 0.110
    source_cup_wall_thickness: float = 0.007
    source_cup_bottom_thickness: float = 0.009
    source_cup_friction: float = 0.9
    cup_mass: float = 0.05
    # Match the visible glass's exact 56 x 56 x 119 mm outer envelope. Direct contact instrumentation
    # showed that a 60 mm TCP made the horizontal hand collide with the table in randomized poses;
    # 83 mm preserves a side grasp while providing physical clearance for the complete approach.
    cup_grasp_box_half: tuple[float, float, float] = (0.028, 0.028, 0.0595)
    cup_grasp_height: float = 0.083
    # Cup-local TCP orientation for a true side grasp. A +90 degree rotation about cup +Y maps
    # Panda tool +Z to cup +X, leaving both the finger length and jaw axis parallel to the table.
    cup_grasp_tcp_quat_c: tuple[float, float, float, float] = (
        0.0,
        math.sqrt(0.5),
        0.0,
        math.sqrt(0.5),
    )
    # A/B grasp testing: mu=1 let the media-loaded cup roll out during the carry, while mu=2 kept
    # bilateral contact through the full carry/tilt path without raising tangential stiffness.
    cup_grasp_box_friction: float = 2.0
    # The Newton default (ke=2.5e3 N/m) permits several millimeters of penetration under the
    # finger position drive.  That is enough for the fingers to cross the narrow grasp proxy and
    # lose the contact manifold entirely.  Match Newton's rigid-robot contact recipe instead.
    # A/B sweep: 50 kN/m retained the cup but allowed ~4 mm corner penetration at mu=2;
    # 100 kN/m reduced typical penetration to ~2 mm and stayed stable; 200 kN/m ejected the cup.
    grasp_contact_ke: float = 1.0e5
    grasp_contact_kd: float = 5.0e2
    grasp_contact_kf: float = 1.0e3
    # Cup reset pose, in the env frame. The cup rests on the table (top z=0) directly under the home
    # gripper, opening up. z is the cup base height (the body local origin sits at the outer base).
    cup_reset_pos: tuple[float, float, float] = (0.5, 0.0, 0.0)

    # ---- receiving cup (fixed, represented once and proxied between solvers) ----
    # The receiver is wider than the source during the initial learning curriculum. It remains a
    # proper hollow cup while making early particle-delivery experience substantially less sparse.
    target_cup_inner_width: float = 0.140
    target_cup_inner_depth: float = 0.140
    target_cup_cavity_depth: float = 0.065
    target_cup_wall_thickness: float = 0.009
    target_cup_bottom_thickness: float = 0.009
    target_cup_friction: float = 0.8
    # Keep enough initial clearance for the Panda finger collision meshes. Moving this wide receiver
    # to y=-0.12 made its rigid rim touch the pre-grasp hand and explosively eject media at step 0.
    target_cup_reset_pos: tuple[float, float, float] = (0.5, -0.18, 0.0)
    # The grasped source origin moves about 5 cm toward environment -y during the deep +x tilt.
    # Starting behind the receiver keeps the draining mouth centered throughout that motion.
    pour_source_offset_xy: tuple[float, float] = (0.0, 0.05)
    collider_margin: float = 0.002

    # Full-task threshold kept as a standalone compatibility knob. Earlier curriculum stages use
    # the values below; :attr:`curriculum_target_frac` combines both sources.
    # A 30% transfer is 74 of the 245 particles. The validated side-pour motion reaches about 41%,
    # leaving enough margin that the success predicate measures manipulation rather than the
    # lower tail of large-batch MPM/contact variation.
    pour_target_frac: float = 0.30
    particle_count_margin: float = 0.003
    # Particle point samples resting on the z=0 MPM spill plane settle within the containment
    # margin above it. Only points in that contact band and outside both cups are true spills.
    spill_table_height: float = 0.0
    max_spill_fraction: float = 0.10
    # A transfer must remain above its stage threshold for this duration before successful
    # termination. This rejects transient particle crossings and aligns reward with curriculum.
    # Nine consecutive control steps reject transient particle crossings while leaving enough of
    # the finite horizon for the terminal event after a late, valid randomized grasp.
    success_dwell_time_s: float = 0.15
    lost_grasp_dwell_time_s: float = 0.05
    """Continuous post-lift grasp loss required before failure [s]."""
    success_min_lift_height: float = 0.05
    success_max_tcp_distance: float = 0.018
    # A contact-free hand reaches a 64 mm measured gap at the bounded 24 mm command, exactly 8 mm
    # wider than this 56 mm cup. Requiring <=6 mm distinguishes real bilateral cup contact while
    # retaining roughly 3 mm of measured true-grasp variation.
    success_max_gripper_width_error: float = 0.006
    # ``None`` derives the largest command that still guarantees the configured drive deflection
    # at geometric cup contact. This keeps continuous near-preload exploration eligible while the
    # physical bilateral-contact predicate rejects a genuinely opening or empty hand.
    success_max_gripper_command: float | None = None

    def _resolved_success_max_gripper_command(self) -> float:
        if self.success_max_gripper_command is not None:
            return float(self.success_max_gripper_command)
        contact_limit = float(self.cup_grasp_box_half[1]) - float(self.actions.gripper_action.contact_min_deflection)
        return max(float(self.gripper_preload_pos), contact_limit)

    # Reset extreme but finite rigid state before it can enter actor observation normalization.
    state_bound_joint_position_margin: float = 0.05
    state_bound_max_joint_velocity: float = 20.0
    state_bound_max_cup_linear_velocity: float = 10.0
    state_bound_max_cup_angular_velocity: float = 50.0
    # Keep finite escaped particles from expanding the sparse NanoVDB hierarchy throughout
    # an episode. Bounds are in each environment's local frame and comfortably contain both cups,
    # every curriculum reset, and the robot workspace.
    # The reset-dataset generator covers the central 90% of the Franka's full 360-degree reachable
    # workspace. Keep the sparse-grid safety envelope symmetric behind the base so valid rear
    # grasps and their cup-contained media are not mistaken for numerical escapes.
    particle_workspace_lower_bound: tuple[float, float, float] = (-1.0, -1.0, -0.5)
    particle_workspace_upper_bound: tuple[float, float, float] = (1.5, 1.0, 1.5)

    # ---- success-driven backward curriculum ----
    # The first reset starts with a grasped cup nearly drained over the receiver, then moves backward
    # through a partial tilt, an upright pour, a source-side carry, an open-finger grasp, and
    # progressively longer open-finger approaches. The final stage adds independently mixed arm,
    # source, and receiver resets. This provides dense direct-control experience without an
    # automatic trajectory.
    # Reset IK is solved once into a bank; asynchronous resets only select prevalidated rows.
    curriculum_stage_names: tuple[str, ...] = CURRICULUM_STAGE_NAMES
    curriculum_pour_arm_q: tuple[float, float, float, float, float, float, float] = (
        -1.47599292,
        0.33629909,
        0.99845403,
        -2.69460344,
        2.62228370,
        1.93315995,
        0.66680431,
    )
    # Collision-screened joint-space points on the authored upright-to-deep-pour segment. They are
    # reset states only: zero action holds the pose, and the policy must command all further motion.
    curriculum_tilt_arm_q: tuple[float, float, float, float, float, float, float] = (
        -1.77459520,
        0.79981079,
        1.18383252,
        -2.64059955,
        2.53800059,
        2.19650501,
        1.88206341,
    )
    curriculum_deep_tilt_arm_q: tuple[float, float, float, float, float, float, float] = (
        -1.86647283,
        0.94242977,
        1.24087206,
        -2.62398297,
        2.51206732,
        2.27753425,
        2.25598929,
    )
    curriculum_drain_arm_q: tuple[float, float, float, float, float, float, float] = (
        -1.91241164,
        1.01373926,
        1.26939183,
        -2.61567468,
        2.49910069,
        2.31804888,
        2.44295223,
    )
    # Roll the side-grasped glass 140 degrees about the horizontal approach axis. The 119 mm glass
    # retains its granular media at 120 degrees; this deeper but still natural wrist roll drains it
    # without the instability observed at 150 degrees.
    curriculum_pour_target_arm_q: tuple[float, float, float, float, float, float, float] = (
        -1.93538105,
        1.04939401,
        1.28365171,
        -2.61152053,
        2.49261737,
        2.33830619,
        2.53643370,
    )
    curriculum_carry_arm_q: tuple[float, float, float, float, float, float, float] = (
        -1.16845703,
        0.55803788,
        0.95656616,
        -2.75139022,
        2.87593412,
        1.73866940,
        0.36629686,
    )
    # Move backward from the receiver-side pour pose toward the source-side carry pose in two
    # collision-screened joint-space increments. Each reset still derives the held cup pose from
    # forward kinematics, so the arm, cup, media, and grasp remain exactly co-located.
    curriculum_transport_reset_fractions: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
    # The deterministic validation motion reaches ~41% on the full task. First-time particle
    # delivery remains rewarded above every success threshold.
    curriculum_early_target_frac: tuple[float, ...] = (
        0.05,
        0.08,
        0.10,
        0.15,
        0.15,
        0.18,
        0.20,
        0.30,
        0.30,
        0.30,
        0.30,
        0.30,
        0.30,
        0.30,
    )
    curriculum_randomized_pour_target_frac: float = 0.30
    # Conservative Cartesian half-extents containing the polar workspace below. This field stays
    # available for downstream rectangular-reset configurations and particle-workspace validation;
    # the Franka Pour preset uses the explicit polar domain when ``source_radius_range`` is set.
    curriculum_randomized_source_position_range: tuple[float, float] = (0.30, 0.45)
    # Cover the useful front-table reach sector instead of the former thin diagonal strip. At full
    # extent the source samples a candidate side-grasp sector from 40--78 cm radius and +/-35
    # degrees about the fixed Franka base. A startup sweep retained complete collision-free paths
    # across every radial ring and both angular boundaries with margin beyond the configured IK
    # thresholds. Lower curriculum levels interpolate these cells from the authored pose.
    curriculum_randomized_source_radius_range: tuple[float, float] | None = (0.40, 0.78)
    curriculum_randomized_source_azimuth_range: float = math.radians(35.0)
    # Retained for rectangular downstream presets. Zero means a full two-dimensional rectangle;
    # it is ignored by the polar Franka Pour preset.
    curriculum_randomized_source_xy_correlation: float = 0.0
    # After grasping anywhere in the polar source sector, pull the glass into the central +/-3 cm
    # carry corridor while lifting. This preserves the full reach problem without forcing an
    # edge-of-workspace upright carry against a joint limit.
    curriculum_randomized_carry_position_range: tuple[float, float] = (0.03, 0.03)
    # In the polar preset the cup's grasp face points approximately toward the robot. This optional
    # value adds a local yaw perturbation around that radial-facing direction so the policy cannot
    # rely on the source grasp face pointing toward the robot at the high-randomization frontier.
    curriculum_randomized_source_yaw_range: float = math.radians(30.0)
    # Keep the authored -Y pour direction while moving the receiver throughout the reachable table
    # region. The reset mixer enforces rectangular separation after progressively breaking the
    # source/receiver bank pairing.
    curriculum_randomized_target_center_xy: tuple[float, float] = (0.50, -0.18)
    curriculum_randomized_target_position_range: tuple[float, float] = (0.15, 0.44)
    curriculum_randomized_cup_clearance: float = 0.04
    # The randomized stage starts 12 cm behind the glass along cup -X. The runtime rotates this
    # cup-local offset and jitter by the source yaw, preserving the horizontal approach geometry.
    curriculum_randomized_reset_tcp_standoff: tuple[float, float, float] = (-0.12, 0.0, 0.0)
    curriculum_randomized_reset_tcp_jitter: tuple[float, float, float] = (0.02, 0.03, 0.0)
    # Optional asymmetric offset box applied on top of the centered pre-grasp standoff. At full
    # extent it varies approach depth from 10--28 cm, lateral displacement by +/-16 cm, and height
    # by 25 cm. Every pose is solved and collision-screened before entering the reset bank. ``None``
    # preserves the legacy symmetric-jitter design for downstream presets.
    curriculum_randomized_reset_tcp_offset_lower: tuple[float, float, float] | None = (-0.16, -0.16, 0.0)
    curriculum_randomized_reset_tcp_offset_upper: tuple[float, float, float] | None = (0.02, 0.16, 0.25)
    # Reset-only orientation error relative to the cup-aligned grasp frame. Fibonacci-sphere axes
    # avoid a preferred rotation direction, while a positive lower angle guarantees that the final
    # curriculum never begins perfectly grasp-aligned, so the direct policy must recover from the
    # observed pose error instead of receiving an aligned reach problem.
    curriculum_randomized_reset_tcp_rotation_angle_range: tuple[float, float] = (
        math.radians(20.0),
        math.radians(60.0),
    )
    curriculum_randomized_reset_tcp_min_grasp_distance: float = 0.09
    # Move backward along samples already exercised by the complete-path collision sweep. Values
    # are interpolation weights from the 6 cm midpoint toward the exact grasp, so decreasing
    # weights increase reset distance before ``full`` introduces the aligned 12 cm pre-grasp.
    curriculum_grasp_approach_fractions: tuple[float, ...] = (0.75, 0.50, 0.375, 0.25, 0.125, 0.0)
    # Keep the held source proxy above the receiver rim throughout the independently solved
    # pour-to-tilt joint interpolation. Without this reserve the source corner and two collider
    # margins overlap even though both endpoint IK poses are valid.
    curriculum_randomized_pour_clearance: float = 0.010
    # Center the closed-finger waypoint on the authored side-grasp point. A lower waypoint made
    # the TCP settle about 8 mm below the grasp point and prevented otherwise valid captures.
    curriculum_grasp_descent_overshoot: float = 0.0
    # An odd source grid includes the authored nominal pose and both configured XY extrema. Newton
    # IK solves this bank once at startup; asynchronous resets only gather prevalidated rows.
    curriculum_randomized_reset_ik_grid_size: int = 7
    curriculum_randomized_reset_ik_samples_per_source: int = 11
    # Require the feasible, posture-screened bank to retain broad workspace coverage. Runtime
    # sampling is uniform over these source XY cells rather than over their unequal IK row counts.
    # The polar grid is a reachability census, not a promise that every Cartesian cell is feasible
    # for the complete grasp-to-deep-tilt trajectory. Retain at least one fifth of its cells while
    # separately requiring every radius, both angular sides, and broad angular span.
    curriculum_randomized_min_source_cell_fraction: float = 0.2
    # Require more than one safe arm start to be available in every retained source cell. The
    # curriculum deliberately exposes only the closest of these paths at small extents, then grows
    # arm-start diversity with the extent and exposes every surviving path at the final level.
    # Screening the full reserve here prevents a broad Cartesian grid from silently collapsing to
    # one repeated Franka posture when the high-randomization levels are reached.
    curriculum_randomized_min_reset_variants_per_source: int = 2
    curriculum_randomized_reset_ik_iterations: int = 160
    curriculum_randomized_reset_ik_max_cost: float = 1.0e-3
    curriculum_randomized_reset_ik_joint_margin: float = 0.015
    # Backward-compatible upper safeguard for the folded panda_joint6 branch. The default lies just
    # inside the URDF limit, so joint-limit and complete-path collision screening perform the
    # effective filtering without imposing a narrow workspace-specific posture heuristic.
    curriculum_randomized_reset_joint6_max: float = 3.75
    # Introduce the final stage through nested, normalized extents across source pose, receiver pose,
    # reset-TCP offset, and the number of exposed arm-start variants. The exact zero-amplitude anchor
    # is behaviorally identical to the mastered full task; small early increments prevent an
    # IK/reset-bank discontinuity from masquerading as exploration difficulty. The last level
    # contains the complete prevalidated randomization bank.
    # Together with the fifteen preceding backward-reset stages, these nine frontiers form a
    # twenty-four-level curriculum. Small initial extents prevent the first randomized reset from
    # simultaneously changing reach, orientation, source placement, and receiver placement beyond
    # the support of the mastered nominal policy.
    curriculum_randomization_extent_levels: tuple[float, ...] = (
        0.0,
        0.05,
        0.10,
        0.20,
        0.35,
        0.50,
        0.70,
        0.85,
        1.0,
    )
    # Preserve paired collision-screened paths during the first geometry-only frontiers, then
    # gradually break arm/source and receiver/source correlations. The final frontier samples all
    # three independently, subject to conservative reset clearances.
    curriculum_independent_arm_fraction_levels: tuple[float, ...] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.10,
        0.25,
        0.50,
        0.75,
        1.0,
    )
    curriculum_independent_target_fraction_levels: tuple[float, ...] = (
        0.0,
        0.0,
        0.0,
        0.10,
        0.25,
        0.50,
        0.70,
        0.85,
        1.0,
    )
    curriculum_independent_sample_attempts: int = 8
    curriculum_independent_arm_min_tcp_distance: float = 0.12
    # Stateful manager progress is not part of RSL-RL checkpoints. Set both start controls to the
    # last logged values when resuming within the randomized stage.
    curriculum_randomization_start_level: int = 0
    curriculum_success_threshold: float = 0.8
    # Standard Franka manipulation tasks train directly on broad reset distributions. Use a lower
    # threshold only to expose the next nested randomization frontier; final task mastery still
    # requires ``curriculum_success_threshold`` at full amplitude.
    curriculum_randomization_promotion_threshold: float = 0.65
    # Use the most recent 4,096 frontier episodes for the success estimate. This is a statistical
    # window, not sufficient policy-training exposure when thousands of environments reset at once.
    curriculum_min_resets_per_stage: int = 4096
    # Require eight whole vectorized-environment reset cohorts before promotion. At the documented
    # 512-environment scale this equals the success window above; at 3,000 environments it prevents
    # a frontier from being promoted after only one or two PPO updates. The entry replay mixture
    # decays over this same exposure horizon so the predecessor skill remains represented while the
    # new prerequisite is consolidated. Set to zero only to retain the absolute-window behavior.
    curriculum_min_reset_cohorts_per_stage: float = 8.0
    # Retain the immediately preceding nested task after promotion so the policy does not forget
    # already-solved behavior while learning the newly introduced prerequisite.
    curriculum_previous_stage_replay_fraction: float = 0.1
    # At a newly introduced frontier, begin with a balanced predecessor mixture and decay toward
    # the retention fraction above over one evidence window. This avoids replacing nearly the
    # entire rollout distribution at the exact moment a new prerequisite is introduced.
    curriculum_frontier_entry_replay_fraction: float = 0.5
    # Set this to the last logged ``Curriculum/stage/stage`` value when resuming training.
    curriculum_start_stage: int = 0
    curriculum_freeze: bool = False

    # ---- media (granular sand inside the cup) ----
    # Preserve the former particle volume in the taller 110 mm cavity. This keeps the
    # particle count, material mass, sparse-grid capacity, and RL transfer threshold unchanged.
    media_fill_frac: float = 0.17181818181818181
    # Normal rollouts peak near 2.2 m/s. This generous clamp prevents a numerically launched
    # particle from crossing many NanoVDB upper regions within one manager step, before the
    # workspace termination can selectively reset its environment.
    particle_max_velocity: float = 10.0
    media_material: MPMParticleMaterialCfg = MPMParticleMaterialCfg(
        density=1500.0,
        friction=0.7,
        yield_pressure=1.0e12,
    )

    # ---- MPM ----
    voxel_size: float = 0.01
    particles_per_cell: float = 2.0
    mpm_iterations: int = 24
    # A rebuildable multi-world grid stores guard/topology voxels in addition to each particle's
    # occupied cell. Two aligned blocks per world prevent high world indices from exhausting the
    # global captured reserve after tilted fills spread, while retaining a sparse local grid.
    mpm_cell_capacity_alignment: int = 512
    mpm_cell_cap_override: int | None = None
    # Advance the complete coupled system once per 120 Hz simulation tick. Per-entry refinements
    # below avoid repeating the outer collision and proxy-coupling pipeline.
    physics_substeps: int = 1
    # Four rigid refinements preserve the former number of arm solves per simulation tick without
    # multiplying the coupled proxy exchange or MPM work.
    rigid_entry_substeps: int = 4
    # Refine only the implicit-MPM entry twice inside each coupled tick. This improves thin-wall
    # collision stability without duplicating the outer collision/coupling pipeline; rigid and MPM
    # accuracy remain independently configurable through their per-entry substeps.
    mpm_entry_substeps: int = 2
    proxy_iterations: int = 1
    # This scales only the virtual cup inertia inside the destination MPM solve. The rigid solver
    # retains the authored cup mass and receives the harvested MPM reaction wrench. A stiff proxy
    # prevents split-step cup yielding from letting resting media creep through its floor while
    # retaining one inexpensive two-way coupling pass.
    proxy_mass_scale: float = 100.0
    # Newton supports captured sparse rebuilds with one local MPM world per replicated environment.
    use_cuda_graph: bool = True

    @property
    def curriculum_target_frac(self) -> tuple[float, ...]:
        """Per-stage delivered-particle success fractions."""
        return (
            *self.curriculum_early_target_frac,
            float(self.pour_target_frac),
            float(self.curriculum_randomized_pour_target_frac),
        )

    @curriculum_target_frac.setter
    def curriculum_target_frac(self, values: tuple[float, ...]) -> None:
        if len(values) != len(CURRICULUM_STAGE_NAMES):
            raise ValueError(f"curriculum_target_frac must contain {len(CURRICULUM_STAGE_NAMES)} values.")
        self.curriculum_early_target_frac = tuple(values[:14])
        self.pour_target_frac = float(values[14])
        self.curriculum_randomized_pour_target_frac = float(values[15])

    def _curriculum_transport_arm_configs(self) -> tuple[tuple[float, ...], ...]:
        """Return joint configurations between the receiver-side pour and source-side carry poses."""
        return tuple(
            tuple(
                (1.0 - fraction) * pour_q + fraction * carry_q
                for pour_q, carry_q in zip(
                    self.curriculum_pour_arm_q,
                    self.curriculum_carry_arm_q,
                    strict=True,
                )
            )
            for fraction in self.curriculum_transport_reset_fractions
        )

    def _configure_reward_cfg(self, max_gripper_command: float, *, initialize_stage_gates: bool) -> None:
        """Propagate task controls into the reverse-curriculum reward terms."""
        progress_params = self.rewards.task_progress.params
        progress_params["grasp_preload_position"] = self.gripper_preload_pos
        progress_params["source_offset_xy"] = self.pour_source_offset_xy
        progress_params["source_mouth_height"] = self.source_cup_bottom_thickness + self.source_cup_cavity_depth
        progress_params["min_lift_height"] = self.success_min_lift_height
        progress_params["max_tcp_distance"] = self.success_max_tcp_distance
        progress_params["max_gripper_width_error"] = self.success_max_gripper_width_error
        progress_params["max_gripper_command"] = max_gripper_command
        grasp_lift_params = self.rewards.grasp_lift_progress.params
        grasp_lift_params["grasp_preload_position"] = self.gripper_preload_pos
        if initialize_stage_gates:
            progress_params["active_through_stage"] = self.curriculum_stage_names.index("carry")
            self.rewards.approach_progress.params["active_from_stage"] = self.curriculum_stage_names.index("approach_1")
            grasp_lift_params["active_from_stage"] = self.curriculum_stage_names.index("near_carry")
        delivered_params = self.rewards.delivered.params
        delivered_params["min_lift_height"] = self.success_min_lift_height
        delivered_params["max_tcp_distance"] = self.success_max_tcp_distance
        delivered_params["max_gripper_width_error"] = self.success_max_gripper_width_error
        delivered_params["max_gripper_command"] = max_gripper_command

    def __post_init__(self):
        self.actions.gripper_action.close_position = max(0.0, self.gripper_preload_pos - self.gripper_close_offset)
        self.actions.gripper_action.open_position = self.gripper_open_pos
        if self.actions.gripper_action.limit_to_preload:
            self.actions.gripper_action.neutral_position = self.gripper_preload_pos
        else:
            self.actions.gripper_action.neutral_position = self.gripper_open_pos
            self.actions.gripper_action.default_position = self.gripper_preload_pos
            if not self.actions.gripper_action.use_incremental_target:
                self.actions.gripper_action.scale = self.gripper_open_pos - self.gripper_preload_pos
        max_gripper_command = self._resolved_success_max_gripper_command()
        self._configure_reward_cfg(max_gripper_command, initialize_stage_gates=True)
        self.terminations.lost_grasp.params["dwell_time_s"] = self.lost_grasp_dwell_time_s
        self.terminations.lost_grasp.params["max_gripper_command"] = max_gripper_command
        self.terminations.success.params["max_gripper_command"] = max_gripper_command
        self.decimation = 2
        # Recycle failed attempts promptly after the expected manipulation sequence.
        self.episode_length_s = 5.0
        # The deadline is part of the task: an attempt that has not poured within five seconds is
        # a failed finite-horizon episode and must not be value-bootstrapped by RL wrappers.
        self.is_finite_horizon = True
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.use_newton_actuators = False
        self.viewer.eye = (1.4, 1.4, 0.9)
        self.viewer.lookat = (0.5, 0.0, 0.1)
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0

        self._validate_curriculum_cfg()
        self._validate_particle_workspace_cfg()
        self._apply_robot_cfg()

        self.sim.physics = NewtonCfg(
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(
                        name=RIGID_ENTRY,
                        # Proxy coupling keeps the MPM stable, so the arm integrator can be the faster
                        # "implicitfast" (unlike base coupling, which needed "euler"). The cup is a
                        # dynamic rigid body owned by this entry; Newton generates its contacts and
                        # the proxy bridges its cavity mesh to the MPM solver.
                        solver_cfg=MJWarpSolverCfg(
                            use_mujoco_contacts=False, integrator="implicitfast", njmax=510, nconmax=400
                        ),
                        bodies=[
                            ROBOT_BODY_LABEL_PATTERN,
                            SOURCE_CUP_BODY_LABEL_PATTERN,
                            TARGET_CUP_BODY_LABEL_PATTERN,
                        ],
                        include_static_shapes=True,
                        substeps=self.rigid_entry_substeps,
                    ),
                    CouplerEntryCfg(
                        name=MPM_ENTRY,
                        solver_cfg=MPMSolverCfg(
                            voxel_size=self.voxel_size,
                            grid_type="sparse",
                            grid_padding=0,
                            max_active_cell_count=-1,
                            strain_basis="P0",
                            transfer_scheme="apic",
                            max_iterations=self.mpm_iterations,
                            warmstart_mode="none",
                            # PIC27 bounds collider work by particle samples.
                            velocity_basis="Q1",
                            collider_basis="pic27",
                            # "forward": the moving cup carries its media ("backward" drains it).
                            collider_velocity_mode="forward",
                            # Keep the task's validated nonlinear solve while sparse topology is
                            # rebuilt eagerly around the physically separated environments.
                            solver="jacobi",
                        ),
                        all_particles=True,
                        bodies=[SPILL_FLOOR_LABEL_PATTERN],
                        include_static_shapes=False,
                        include_child_joints=False,
                        substeps=self.mpm_entry_substeps,
                        in_place=True,
                    ),
                ],
                proxies=[
                    CouplerProxyMappingCfg(
                        source=RIGID_ENTRY,
                        destination=MPM_ENTRY,
                        bodies=[SOURCE_CUP_BODY_LABEL_PATTERN, TARGET_CUP_BODY_LABEL_PATTERN],
                        mass_scale=self.proxy_mass_scale,
                        mode="lagged",
                        # Implicit MPM resolves its proxy colliders internally; the shared outer
                        # pipeline is only needed for rigid MJWarp contacts.
                        collision_pipeline=lambda _model: None,
                    )
                ],
                iterations=self.proxy_iterations,
            ),
            # Rigid contacts use Newton's outer pipeline. Implicit MPM handles particle/shape
            # collisions internally, so allocating outer soft contacts would waste O(P*S) work.
            collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
            num_substeps=self.physics_substeps,
            use_cuda_graph=self.use_cuda_graph,
        )

    def _validate_gripper_action_cfg(self) -> None:
        """Validate the reset and action targets against the Panda finger range."""
        gripper_action = self.actions.gripper_action
        if not isinstance(gripper_action.use_incremental_target, bool):
            raise TypeError("Gripper use_incremental_target must be a bool.")
        if gripper_action.binary_threshold is not None:
            if (
                isinstance(gripper_action.binary_threshold, bool)
                or not math.isfinite(gripper_action.binary_threshold)
                or not -1.0 < gripper_action.binary_threshold < 1.0
            ):
                raise ValueError("Gripper binary_threshold must be finite and lie strictly between -1 and 1.")
            if gripper_action.use_incremental_target:
                raise ValueError("Binary and incremental gripper targets are mutually exclusive.")
        if (
            not math.isfinite(self.gripper_open_pos)
            or not 0.0 < self.gripper_open_pos <= 0.04
            or not math.isfinite(gripper_action.close_position)
            or not 0.0 <= gripper_action.close_position < self.gripper_open_pos
            or not math.isfinite(gripper_action.scale)
            or gripper_action.scale <= 0.0
            or not math.isfinite(gripper_action.alpha)
            or not 0.0 < gripper_action.alpha <= 1.0
        ):
            raise ValueError(
                "Gripper action positions must fit the Panda finger range [0, 0.04] with positive scale and a "
                "moving-average weight in (0, 1]."
            )
        if not math.isclose(gripper_action.open_position, self.gripper_open_pos, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("The gripper action open position must match gripper_open_pos.")
        if (
            not math.isfinite(self.gripper_preload_pos)
            or not gripper_action.close_position <= self.gripper_preload_pos < self.cup_grasp_box_half[1]
        ):
            raise ValueError("gripper_preload_pos must lie between the closed and geometric contact positions.")
        if (
            not math.isfinite(self.gripper_close_offset)
            or not 0.0 <= self.gripper_close_offset <= self.gripper_preload_pos
        ):
            raise ValueError("gripper_close_offset must lie in [0, gripper_preload_pos].")
        contact_command_limit = self.cup_grasp_box_half[1] - gripper_action.contact_min_deflection
        max_gripper_command = self._resolved_success_max_gripper_command()
        if (
            not math.isfinite(max_gripper_command)
            or not self.gripper_preload_pos <= max_gripper_command <= contact_command_limit
        ):
            raise ValueError(
                "success_max_gripper_command must lie between the preload target and the largest command that "
                "retains contact_min_deflection at the cup."
            )
        max_action_position = self.gripper_preload_pos if gripper_action.limit_to_preload else self.gripper_open_pos
        if not math.isclose(
            gripper_action.neutral_position,
            max_action_position,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("Gripper action maximum does not match its configured operating interval.")
        action_span = max_action_position - gripper_action.close_position
        if gripper_action.scale > action_span + 1.0e-9:
            raise ValueError("Gripper action scale must not exceed its configured operating interval.")
        if gripper_action.default_position is not None and not (
            gripper_action.close_position <= gripper_action.default_position <= gripper_action.neutral_position
        ):
            raise ValueError("Gripper default position must lie within its configured operating interval.")

    def _validate_source_cup_cfg(self) -> None:
        """Validate source-cup geometry, grasp proxy, and media fill as one contract."""
        for field_name in (
            "source_cup_inner_width",
            "source_cup_inner_depth",
            "source_cup_cavity_depth",
            "source_cup_wall_thickness",
            "source_cup_bottom_thickness",
            "cup_mass",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive.")
        if not math.isfinite(self.media_fill_frac) or not 0.0 < self.media_fill_frac <= 1.0:
            raise ValueError("media_fill_frac must lie in (0, 1].")

        outer_size = (
            self.source_cup_inner_width + 2.0 * self.source_cup_wall_thickness,
            self.source_cup_inner_depth + 2.0 * self.source_cup_wall_thickness,
            self.source_cup_cavity_depth + self.source_cup_bottom_thickness,
        )
        if len(self.cup_grasp_box_half) != 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in self.cup_grasp_box_half
        ):
            raise ValueError("cup_grasp_box_half must contain three finite positive half-extents.")
        if any(
            not math.isclose(proxy_half, 0.5 * outer, rel_tol=0.0, abs_tol=1.0e-9)
            for proxy_half, outer in zip(self.cup_grasp_box_half, outer_size, strict=True)
        ):
            raise ValueError("cup_grasp_box_half must exactly match the visible source-cup outer envelope.")
        if (
            not math.isfinite(self.cup_grasp_height)
            or self.cup_grasp_height <= self.source_cup_bottom_thickness
            or self.cup_grasp_height >= outer_size[2]
        ):
            raise ValueError("cup_grasp_height must lie above the source bottom and below its rim.")
        if len(self.cup_grasp_tcp_quat_c) != 4 or any(not math.isfinite(value) for value in self.cup_grasp_tcp_quat_c):
            raise ValueError("cup_grasp_tcp_quat_c must contain four finite XYZW values.")
        quaternion_norm = math.sqrt(sum(value * value for value in self.cup_grasp_tcp_quat_c))
        if not math.isclose(quaternion_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError("cup_grasp_tcp_quat_c must be a unit quaternion.")

    def _validate_curriculum_progress_cfg(self, stage_count: int) -> None:
        """Validate success statistics and promotion controls."""
        if not 0.0 < self.curriculum_success_threshold <= 1.0:
            raise ValueError("curriculum_success_threshold must lie in (0, 1].")
        if not 0.0 < self.curriculum_randomization_promotion_threshold <= self.curriculum_success_threshold:
            raise ValueError(
                "curriculum_randomization_promotion_threshold must lie in (0, curriculum_success_threshold]."
            )
        if self.curriculum_min_resets_per_stage <= 0:
            raise ValueError("curriculum_min_resets_per_stage must be positive.")
        cohort_count = self.curriculum_min_reset_cohorts_per_stage
        if not math.isfinite(cohort_count) or cohort_count < 0.0:
            raise ValueError("curriculum_min_reset_cohorts_per_stage must be finite and nonnegative.")
        replay_fraction = self.curriculum_previous_stage_replay_fraction
        if not math.isfinite(replay_fraction) or replay_fraction < 0.0 or replay_fraction >= 1.0:
            raise ValueError("curriculum_previous_stage_replay_fraction must lie in [0, 1).")
        entry_replay_fraction = self.curriculum_frontier_entry_replay_fraction
        if (
            not math.isfinite(entry_replay_fraction)
            or entry_replay_fraction < replay_fraction
            or entry_replay_fraction >= 1.0
        ):
            raise ValueError(
                "curriculum_frontier_entry_replay_fraction must lie in [curriculum_previous_stage_replay_fraction, 1)."
            )
        if self.curriculum_start_stage < 0 or self.curriculum_start_stage >= stage_count:
            raise ValueError(f"curriculum_start_stage must lie in [0, {stage_count - 1}].")
        extent_levels = self.curriculum_randomization_extent_levels
        if not extent_levels:
            raise ValueError("curriculum_randomization_extent_levels must not be empty.")
        if any(not math.isfinite(level) or level < 0.0 or level > 1.0 for level in extent_levels):
            raise ValueError("curriculum_randomization_extent_levels must lie in [0, 1].")
        if any(previous >= current for previous, current in zip(extent_levels, extent_levels[1:])):
            raise ValueError("curriculum_randomization_extent_levels must be strictly increasing.")
        if extent_levels[0] != 0.0:
            raise ValueError("curriculum_randomization_extent_levels must start at 0.0.")
        if not math.isclose(extent_levels[-1], 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("curriculum_randomization_extent_levels must end at 1.0.")
        for field_name in (
            "curriculum_independent_arm_fraction_levels",
            "curriculum_independent_target_fraction_levels",
        ):
            fractions = getattr(self, field_name)
            if len(fractions) != len(extent_levels):
                raise ValueError(f"{field_name} must align with curriculum_randomization_extent_levels.")
            if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in fractions):
                raise ValueError(f"{field_name} must contain values in [0, 1].")
            if any(left > right for left, right in zip(fractions, fractions[1:])):
                raise ValueError(f"{field_name} must be nondecreasing.")
            if not math.isclose(fractions[0], 0.0, rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(f"{field_name} must start at 0.0.")
            if not math.isclose(fractions[-1], 1.0, rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(f"{field_name} must end at 1.0.")
        if self.curriculum_independent_sample_attempts <= 0:
            raise ValueError("curriculum_independent_sample_attempts must be positive.")
        if (
            not math.isfinite(self.curriculum_independent_arm_min_tcp_distance)
            or self.curriculum_independent_arm_min_tcp_distance <= 0.0
        ):
            raise ValueError("curriculum_independent_arm_min_tcp_distance must be finite and positive.")
        if self.curriculum_randomization_start_level < 0 or self.curriculum_randomization_start_level >= len(
            extent_levels
        ):
            raise ValueError("curriculum_randomization_start_level must index curriculum_randomization_extent_levels.")
        self._validate_reward_cfg(stage_count)

    def _validate_solver_cfg(self) -> None:
        """Validate the task-level controls copied into the coupled solver tree."""
        for name in (
            "physics_substeps",
            "rigid_entry_substeps",
            "mpm_entry_substeps",
            "mpm_iterations",
            "proxy_iterations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("voxel_size", "proxy_mass_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not isinstance(self.use_cuda_graph, bool):
            raise TypeError("use_cuda_graph must be a bool.")

    def _validate_reward_cfg(self, stage_count: int) -> None:
        """Validate the reverse-curriculum reward parameters."""
        tilt_params = self.rewards.task_progress.params
        target_tilt = float(tilt_params["target_tilt"])
        pour_direction_xy = tilt_params["pour_direction_xy"]
        source_offset_xy = self.pour_source_offset_xy
        source_mouth_height = float(tilt_params["source_mouth_height"])
        alignment_radius = float(tilt_params["alignment_radius"])
        active_through_stage = int(tilt_params["active_through_stage"])
        discount_factor = float(tilt_params["discount_factor"])
        if not math.isfinite(target_tilt) or not 0.0 < target_tilt < math.pi:
            raise ValueError("task_progress target_tilt must lie in (0, pi).")
        if (
            len(pour_direction_xy) != 2
            or any(not math.isfinite(value) for value in pour_direction_xy)
            or math.hypot(float(pour_direction_xy[0]), float(pour_direction_xy[1])) <= 0.0
        ):
            raise ValueError("task_progress pour_direction_xy must contain two finite values and be nonzero.")
        if len(source_offset_xy) != 2 or any(not math.isfinite(value) for value in source_offset_xy):
            raise ValueError("pour_source_offset_xy must contain two finite values.")
        if not math.isfinite(source_mouth_height) or source_mouth_height <= 0.0:
            raise ValueError("task_progress source_mouth_height must be finite and positive.")
        if not math.isfinite(alignment_radius) or alignment_radius <= 0.0:
            raise ValueError("task_progress alignment_radius must be finite and positive.")
        if active_through_stage < 0 or active_through_stage >= stage_count:
            raise ValueError(f"task_progress active_through_stage must lie in [0, {stage_count - 1}].")
        if active_through_stage != self.curriculum_stage_names.index("carry"):
            raise ValueError("task_progress active_through_stage must select the carry curriculum stage.")
        if not math.isfinite(discount_factor) or not 0.0 < discount_factor <= 1.0:
            raise ValueError("task_progress discount_factor must lie in (0, 1].")
        self._validate_guidance_reward_cfg()

    def _validate_guidance_reward_cfg(self) -> None:
        """Validate full-task approach and grasp-lift potential settings."""
        approach_params = self.rewards.approach_progress.params
        for parameter in ("position_std", "orientation_std"):
            value = float(approach_params[parameter])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"approach_progress {parameter} must be finite and positive.")
        open_hand_fraction = float(approach_params["open_hand_fraction"])
        if not math.isfinite(open_hand_fraction) or not 0.0 <= open_hand_fraction <= 1.0:
            raise ValueError("approach_progress open_hand_fraction must lie in [0, 1].")
        if int(approach_params["active_from_stage"]) != self.curriculum_stage_names.index("approach_1"):
            raise ValueError("approach_progress active_from_stage must select the approach_1 curriculum stage.")

        grasp_lift_params = self.rewards.grasp_lift_progress.params
        for parameter in ("target_height", "grasp_reach_std"):
            value = float(grasp_lift_params[parameter])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"grasp_lift_progress {parameter} must be finite and positive.")
        if not math.isfinite(float(grasp_lift_params["grasp_preload_position"])):
            raise ValueError("grasp_lift_progress grasp_preload_position must be finite.")
        grasp_fraction = float(grasp_lift_params["grasp_fraction"])
        if not math.isfinite(grasp_fraction) or not 0.0 <= grasp_fraction <= 1.0:
            raise ValueError("grasp_lift_progress grasp_fraction must lie in [0, 1].")
        if int(grasp_lift_params["active_from_stage"]) != self.curriculum_stage_names.index("near_carry"):
            raise ValueError("grasp_lift_progress active_from_stage must select the near_carry curriculum stage.")

        for name, params in (
            ("approach_progress", approach_params),
            ("grasp_lift_progress", grasp_lift_params),
        ):
            value = float(params["discount_factor"])
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} discount_factor must lie in (0, 1].")

    def _validate_arm_action_cfg(self) -> None:
        """Validate the task's direct relative arm-joint action."""
        arm_action = self.actions.arm_action
        if not isinstance(arm_action, mdp.RelativeJointPositionActionCfg):
            raise ValueError("Franka Pour requires RelativeJointPositionActionCfg for the arm.")
        if arm_action.joint_names != [f"panda_joint{i}" for i in range(1, 8)] or not arm_action.preserve_order:
            raise ValueError("Franka Pour requires all seven Panda arm joints in kinematic order.")
        if not isinstance(arm_action.scale, float) or not math.isfinite(arm_action.scale) or arm_action.scale <= 0.0:
            raise ValueError("Arm relative-action scale must be a finite positive scalar.")
        if not arm_action.use_zero_offset:
            raise ValueError("Franka Pour relative arm actions require use_zero_offset=True.")

    def _validate_curriculum_cfg(self) -> None:
        """Validate the aligned per-stage backward-curriculum settings."""
        self._validate_solver_cfg()
        self._validate_source_cup_cfg()
        self._validate_gripper_action_cfg()
        self._validate_arm_action_cfg()
        stage_count = len(self.curriculum_stage_names)
        if self.curriculum_stage_names != CURRICULUM_STAGE_NAMES:
            raise ValueError(f"curriculum_stage_names must be {CURRICULUM_STAGE_NAMES!r}.")
        if len(self.curriculum_target_frac) != stage_count:
            raise ValueError(
                f"curriculum_target_frac has {len(self.curriculum_target_frac)} values for {stage_count} stages."
            )
        if any(
            not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0 for fraction in self.curriculum_target_frac
        ):
            raise ValueError("Curriculum target fractions must lie in (0, 1].")
        if tuple(sorted(self.curriculum_target_frac)) != self.curriculum_target_frac:
            raise ValueError("Curriculum target fractions must be nondecreasing.")
        self._validate_transport_reset_cfg()
        self._validate_grasp_approach_reset_cfg()
        self._validate_curriculum_progress_cfg(stage_count)
        if self.cup_grasp_box_half[1] < 0.0 or self.cup_grasp_box_half[1] > self.gripper_open_pos:
            raise ValueError("The curriculum contact position must fit within the open gripper.")
        self._validate_randomized_reset_cfg()
        self._validate_curriculum_arm_configs()

    def _validate_transport_reset_cfg(self) -> None:
        """Validate intermediate held-cup reset locations between pour and carry."""
        pour_stage = self.curriculum_stage_names.index("pour")
        carry_stage = self.curriculum_stage_names.index("carry")
        fractions = self.curriculum_transport_reset_fractions
        if len(fractions) != carry_stage - pour_stage - 1:
            raise ValueError(
                "curriculum_transport_reset_fractions must provide one value for every stage between pour and carry."
            )
        if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in fractions) or any(
            left >= right for left, right in zip(fractions, fractions[1:])
        ):
            raise ValueError("curriculum_transport_reset_fractions must increase strictly within (0, 1).")

    def _validate_grasp_approach_reset_cfg(self) -> None:
        """Validate collision-screened reset samples between exact grasp and pre-grasp."""
        grasp_stage = self.curriculum_stage_names.index("grasp")
        full_stage = self.curriculum_stage_names.index("full")
        fractions = self.curriculum_grasp_approach_fractions
        if len(fractions) != full_stage - grasp_stage - 1:
            raise ValueError(
                "curriculum_grasp_approach_fractions must provide one value for every stage between grasp and full."
            )
        if any(not math.isfinite(value) or value < 0.0 or value >= 1.0 for value in fractions) or any(
            left <= right for left, right in zip(fractions, fractions[1:])
        ):
            raise ValueError("curriculum_grasp_approach_fractions must decrease strictly within [0, 1).")
        if any(not math.isclose(8.0 * value, round(8.0 * value), abs_tol=1.0e-9) for value in fractions):
            raise ValueError(
                "curriculum_grasp_approach_fractions must select eighth-segment samples covered by collision screening."
            )

    def _validate_randomized_reset_cfg(self) -> None:
        """Validate randomized cup poses and their precomputed IK reset bank."""
        self._validate_randomized_workspace_cfg()
        self._validate_randomized_reset_offset_bounds_cfg()
        self._validate_randomized_reset_selection_cfg()
        self._validate_randomized_grasp_approach_cfg()
        self._validate_randomized_ik_solver_cfg()
        self._validate_randomized_cup_separation_cfg()

    def _validate_randomized_workspace_cfg(self) -> None:
        """Validate randomized source, carry, and receiver workspace geometry."""
        for field_name in (
            "curriculum_randomized_source_position_range",
            "curriculum_randomized_carry_position_range",
            "curriculum_randomized_target_position_range",
        ):
            values = getattr(self, field_name)
            if len(values) != 2 or any(not math.isfinite(value) or value < 0.0 for value in values):
                raise ValueError(f"{field_name} must contain two finite nonnegative values.")
        if (
            not math.isfinite(self.curriculum_randomized_source_xy_correlation)
            or not 0.0 <= self.curriculum_randomized_source_xy_correlation < 1.0
        ):
            raise ValueError("curriculum_randomized_source_xy_correlation must lie in [0, 1).")
        radius_range = self.curriculum_randomized_source_radius_range
        if radius_range is not None:
            if self.cup_reset_pos[0] <= 0.0 or not math.isclose(
                self.cup_reset_pos[1],
                0.0,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError(
                    "The polar source workspace requires cup_reset_pos to lie on the positive robot-base X axis."
                )
            if (
                len(radius_range) != 2
                or any(not math.isfinite(value) or value <= 0.0 for value in radius_range)
                or radius_range[0] >= radius_range[1]
            ):
                raise ValueError(
                    "curriculum_randomized_source_radius_range must contain two finite positive values "
                    "in increasing order."
                )
            if (
                not math.isfinite(self.curriculum_randomized_source_azimuth_range)
                or not 0.0 < self.curriculum_randomized_source_azimuth_range < math.pi / 2.0
            ):
                raise ValueError("curriculum_randomized_source_azimuth_range must lie in (0, pi / 2).")
            min_radius, max_radius = radius_range
            azimuth = self.curriculum_randomized_source_azimuth_range
            minimum_x = min_radius * math.cos(azimuth)
            maximum_x = max_radius
            maximum_abs_y = max_radius * math.sin(azimuth)
            required_half_range = (
                max(abs(minimum_x - self.cup_reset_pos[0]), abs(maximum_x - self.cup_reset_pos[0])),
                maximum_abs_y,
            )
            if any(
                configured + 1.0e-9 < required
                for configured, required in zip(
                    self.curriculum_randomized_source_position_range,
                    required_half_range,
                    strict=True,
                )
            ):
                raise ValueError(
                    "curriculum_randomized_source_position_range must contain the configured polar workspace; "
                    f"required at least {required_half_range}."
                )
        elif not math.isfinite(self.curriculum_randomized_source_azimuth_range):
            raise ValueError("curriculum_randomized_source_azimuth_range must be finite.")
        if any(
            carry_range > source_range
            for carry_range, source_range in zip(
                self.curriculum_randomized_carry_position_range,
                self.curriculum_randomized_source_position_range,
                strict=True,
            )
        ):
            raise ValueError(
                "curriculum_randomized_carry_position_range must not exceed "
                "curriculum_randomized_source_position_range."
            )
        if len(self.curriculum_randomized_target_center_xy) != 2 or any(
            not math.isfinite(value) for value in self.curriculum_randomized_target_center_xy
        ):
            raise ValueError("curriculum_randomized_target_center_xy must contain two finite values.")
        if (
            not math.isfinite(self.curriculum_randomized_source_yaw_range)
            or self.curriculum_randomized_source_yaw_range < 0.0
            or self.curriculum_randomized_source_yaw_range > math.pi / 4.0
        ):
            raise ValueError("curriculum_randomized_source_yaw_range must lie in [0, pi / 4].")
        if not math.isfinite(self.curriculum_randomized_cup_clearance) or self.curriculum_randomized_cup_clearance < 0:
            raise ValueError("curriculum_randomized_cup_clearance must be finite and nonnegative.")

    def _validate_randomized_reset_offset_bounds_cfg(self) -> None:
        """Validate legacy jitter and optional asymmetric reset-TCP offset bounds."""
        for field_name in (
            "curriculum_randomized_reset_tcp_standoff",
            "curriculum_randomized_reset_tcp_jitter",
        ):
            values = getattr(self, field_name)
            if len(values) != 3 or any(not math.isfinite(value) for value in values):
                raise ValueError(f"{field_name} must contain three finite values.")
        if any(value < 0.0 for value in self.curriculum_randomized_reset_tcp_jitter):
            raise ValueError("curriculum_randomized_reset_tcp_jitter must contain three finite nonnegative values.")
        offset_lower = self.curriculum_randomized_reset_tcp_offset_lower
        offset_upper = self.curriculum_randomized_reset_tcp_offset_upper
        if (offset_lower is None) != (offset_upper is None):
            raise ValueError(
                "curriculum_randomized_reset_tcp_offset_lower and "
                "curriculum_randomized_reset_tcp_offset_upper must either both be set or both be None."
            )
        if offset_lower is not None and offset_upper is not None:
            if len(offset_lower) != 3 or len(offset_upper) != 3:
                raise ValueError("Randomized reset TCP offset bounds must each contain three values.")
            if any(not math.isfinite(value) for value in (*offset_lower, *offset_upper)):
                raise ValueError("Randomized reset TCP offset bounds must be finite.")
            if any(lower > 0.0 or upper < 0.0 or lower > upper for lower, upper in zip(offset_lower, offset_upper)):
                raise ValueError("Randomized reset TCP offset bounds must be ordered and contain zero.")
            if offset_lower[2] < 0.0:
                raise ValueError("Randomized reset TCP offsets must not place the initial TCP below its pre-grasp.")
        rotation_range = self.curriculum_randomized_reset_tcp_rotation_angle_range
        if len(rotation_range) != 2 or any(not math.isfinite(value) for value in rotation_range):
            raise ValueError("curriculum_randomized_reset_tcp_rotation_angle_range must contain two finite values.")
        rotation_lower, rotation_upper = rotation_range
        if rotation_lower < 0.0 or rotation_lower > rotation_upper or rotation_upper > math.pi / 2.0:
            raise ValueError("curriculum_randomized_reset_tcp_rotation_angle_range must be ordered within [0, pi / 2].")

    def _validate_randomized_reset_selection_cfg(self) -> None:
        """Validate reset-bank filtering thresholds and row-coverage requirements."""
        if (
            not math.isfinite(self.curriculum_randomized_reset_tcp_min_grasp_distance)
            or self.curriculum_randomized_reset_tcp_min_grasp_distance <= 0.0
        ):
            raise ValueError("curriculum_randomized_reset_tcp_min_grasp_distance must be finite and positive.")
        if (
            not math.isfinite(self.curriculum_randomized_reset_joint6_max)
            or self.curriculum_randomized_reset_joint6_max <= 0.0
        ):
            raise ValueError("curriculum_randomized_reset_joint6_max must be finite and positive.")
        if (
            not math.isfinite(self.curriculum_randomized_min_source_cell_fraction)
            or not 0.0 < self.curriculum_randomized_min_source_cell_fraction <= 1.0
        ):
            raise ValueError("curriculum_randomized_min_source_cell_fraction must lie in (0, 1].")
        if (
            self.curriculum_randomized_min_reset_variants_per_source < 1
            or self.curriculum_randomized_min_reset_variants_per_source
            > self.curriculum_randomized_reset_ik_samples_per_source
        ):
            raise ValueError(
                "curriculum_randomized_min_reset_variants_per_source must lie between one and "
                "curriculum_randomized_reset_ik_samples_per_source."
            )
        if (
            not math.isfinite(self.curriculum_randomized_pour_clearance)
            or self.curriculum_randomized_pour_clearance < 0.0
        ):
            raise ValueError("curriculum_randomized_pour_clearance must be finite and nonnegative.")
        if not math.isfinite(self.curriculum_grasp_descent_overshoot) or self.curriculum_grasp_descent_overshoot < 0.0:
            raise ValueError("curriculum_grasp_descent_overshoot must be finite and nonnegative.")

    def _validate_randomized_grasp_approach_cfg(self) -> None:
        """Validate horizontal grasp orientation, standoff, and minimum clearance."""
        offset_lower = self.curriculum_randomized_reset_tcp_offset_lower
        offset_upper = self.curriculum_randomized_reset_tcp_offset_upper
        grasp_qx, grasp_qy, grasp_qz, grasp_qw = self.cup_grasp_tcp_quat_c
        tool_axis_c = (
            2.0 * (grasp_qx * grasp_qz + grasp_qy * grasp_qw),
            2.0 * (grasp_qy * grasp_qz - grasp_qx * grasp_qw),
            1.0 - 2.0 * (grasp_qx * grasp_qx + grasp_qy * grasp_qy),
        )
        if abs(tool_axis_c[2]) > 1.0e-6:
            raise ValueError("cup_grasp_tcp_quat_c must keep Panda tool +Z parallel to the table.")
        jaw_axis_z = 2.0 * (grasp_qy * grasp_qz + grasp_qx * grasp_qw)
        if abs(jaw_axis_z) > 1.0e-6:
            raise ValueError("cup_grasp_tcp_quat_c must keep the Panda jaw axis parallel to the table.")
        standoff_norm = math.sqrt(sum(value * value for value in self.curriculum_randomized_reset_tcp_standoff))
        if standoff_norm <= 1.0e-9:
            raise ValueError("curriculum_randomized_reset_tcp_standoff must be nonzero.")
        standoff_alignment = sum(
            tool_axis * standoff / standoff_norm
            for tool_axis, standoff in zip(
                tool_axis_c,
                self.curriculum_randomized_reset_tcp_standoff,
                strict=True,
            )
        )
        if not math.isclose(standoff_alignment, -1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError(
                "curriculum_randomized_reset_tcp_standoff must be antiparallel to Panda tool +Z "
                "from cup_grasp_tcp_quat_c."
            )
        if (
            offset_lower is None
            and self.cup_grasp_height - self.curriculum_randomized_reset_tcp_jitter[2] <= self.collider_margin
        ):
            raise ValueError(
                "cup_grasp_height and curriculum_randomized_reset_tcp_jitter must keep every reset TCP "
                "above the table by more than collider_margin."
            )
        if offset_lower is None or offset_upper is None:
            minimum_standoff = math.sqrt(
                sum(
                    max(abs(offset) - jitter, 0.0) ** 2
                    for offset, jitter in zip(
                        self.curriculum_randomized_reset_tcp_standoff,
                        self.curriculum_randomized_reset_tcp_jitter,
                        strict=True,
                    )
                )
            )
        else:
            closest_offset = tuple(
                min(max(-standoff, lower), upper)
                for standoff, lower, upper in zip(
                    self.curriculum_randomized_reset_tcp_standoff,
                    offset_lower,
                    offset_upper,
                    strict=True,
                )
            )
            minimum_standoff = math.sqrt(
                sum(
                    (standoff + offset) ** 2
                    for standoff, offset in zip(
                        self.curriculum_randomized_reset_tcp_standoff,
                        closest_offset,
                        strict=True,
                    )
                )
            )
        if minimum_standoff + 1.0e-9 < self.curriculum_randomized_reset_tcp_min_grasp_distance:
            raise ValueError(
                "curriculum_randomized_reset_tcp_standoff and curriculum_randomized_reset_tcp_jitter "
                "cannot guarantee curriculum_randomized_reset_tcp_min_grasp_distance."
            )

    def _validate_randomized_ik_solver_cfg(self) -> None:
        """Validate randomized reset-bank IK discretization and solver tolerances."""
        if self.curriculum_randomized_reset_ik_grid_size < 3 or self.curriculum_randomized_reset_ik_grid_size % 2 == 0:
            raise ValueError("curriculum_randomized_reset_ik_grid_size must be an odd integer of at least three.")
        if (
            self.curriculum_randomized_reset_ik_samples_per_source < 3
            or self.curriculum_randomized_reset_ik_samples_per_source % 2 == 0
        ):
            raise ValueError(
                "curriculum_randomized_reset_ik_samples_per_source must be an odd integer of at least three."
            )
        if self.curriculum_randomized_reset_ik_iterations <= 0:
            raise ValueError("curriculum_randomized_reset_ik_iterations must be positive.")
        for field_name in (
            "curriculum_randomized_reset_ik_max_cost",
            "curriculum_randomized_reset_ik_joint_margin",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and nonnegative.")

    def _validate_randomized_cup_separation_cfg(self) -> None:
        """Validate that every randomized source pose admits a separated receiver pose."""
        source_outer_half_x = self.source_cup_inner_width / 2.0 + self.source_cup_wall_thickness
        source_outer_half_y = self.source_cup_inner_depth / 2.0 + self.source_cup_wall_thickness
        target_outer_half_y = self.target_cup_inner_depth / 2.0 + self.target_cup_wall_thickness
        if self.curriculum_randomized_source_radius_range is None:
            maximum_projection_yaw = min(
                self.curriculum_randomized_source_yaw_range,
                math.atan2(source_outer_half_x, source_outer_half_y),
            )
            maximum_source_half_y = source_outer_half_x * math.sin(
                maximum_projection_yaw
            ) + source_outer_half_y * math.cos(maximum_projection_yaw)
        else:
            # A radial-facing cup can reach any global yaw in the polar sector. The diagonal is the
            # conservative support of the square source under arbitrary upright yaw.
            maximum_source_half_y = math.hypot(source_outer_half_x, source_outer_half_y)
        minimum_separation = maximum_source_half_y + target_outer_half_y + self.curriculum_randomized_cup_clearance
        if self.curriculum_randomized_source_radius_range is None:
            minimum_source_y = self.cup_reset_pos[1] - self.curriculum_randomized_source_position_range[1]
        else:
            minimum_source_y = -self.curriculum_randomized_source_radius_range[1] * math.sin(
                self.curriculum_randomized_source_azimuth_range
            )
        minimum_target_y = (
            self.curriculum_randomized_target_center_xy[1] - self.curriculum_randomized_target_position_range[1]
        )
        if minimum_source_y - minimum_separation < minimum_target_y - 1.0e-6:
            raise ValueError(
                "curriculum_randomized_target_position_range leaves no collision-free target y-position "
                "at the minimum randomized source y-position."
            )

    def _validate_curriculum_arm_configs(self) -> None:
        """Validate authored arm waypoints against the action limits."""
        arm_configs = (
            self.curriculum_drain_arm_q,
            self.curriculum_deep_tilt_arm_q,
            self.curriculum_tilt_arm_q,
            self.curriculum_pour_arm_q,
            *self._curriculum_transport_arm_configs(),
            self.curriculum_pour_target_arm_q,
            self.curriculum_carry_arm_q,
            self.arm_home,
        )
        for arm_q in arm_configs:
            if len(arm_q) != 7:
                raise ValueError("Every curriculum arm configuration must contain seven joint positions.")
            for joint_name, position, (lower, upper) in zip(
                self.actions.arm_action.joint_names,
                arm_q,
                PANDA_ARM_JOINT_LIMITS,
                strict=True,
            ):
                if not math.isfinite(position) or position < lower or position > upper:
                    raise ValueError(
                        f"Curriculum joint position {joint_name}={position} lies outside [{lower}, {upper}]."
                    )

    def _validate_particle_workspace_cfg(self) -> None:
        """Validate finite local particle bounds and all configured media reset poses."""
        if not math.isfinite(self.particle_max_velocity) or self.particle_max_velocity <= 0.0:
            raise ValueError("particle_max_velocity must be finite and positive.")
        if not math.isfinite(self.spill_table_height):
            raise ValueError("spill_table_height must be finite.")
        if not 0.0 < self.max_spill_fraction < 1.0:
            raise ValueError("max_spill_fraction must lie in (0, 1).")
        if not math.isfinite(self.success_dwell_time_s) or self.success_dwell_time_s <= 0.0:
            raise ValueError("success_dwell_time_s must be finite and positive.")
        if not math.isfinite(self.lost_grasp_dwell_time_s) or self.lost_grasp_dwell_time_s <= 0.0:
            raise ValueError("lost_grasp_dwell_time_s must be finite and positive.")
        for field_name in (
            "success_min_lift_height",
            "success_max_tcp_distance",
            "success_max_gripper_width_error",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive.")
        if not math.isfinite(self.state_bound_joint_position_margin) or self.state_bound_joint_position_margin < 0.0:
            raise ValueError("state_bound_joint_position_margin must be finite and nonnegative.")
        for field_name in (
            "state_bound_max_joint_velocity",
            "state_bound_max_cup_linear_velocity",
            "state_bound_max_cup_angular_velocity",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive.")
        if not math.isfinite(self.particle_count_margin) or self.particle_count_margin < 0.0:
            raise ValueError("particle_count_margin must be finite and nonnegative.")
        lower = self.particle_workspace_lower_bound
        upper = self.particle_workspace_upper_bound
        if len(lower) != 3 or len(upper) != 3:
            raise ValueError("particle_workspace bounds must each contain three coordinates.")
        if any(not math.isfinite(value) for value in (*lower, *upper)):
            raise ValueError("particle_workspace bounds must be finite.")
        if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
            raise ValueError("particle_workspace lower bounds must be smaller than upper bounds.")
        if not lower[2] <= self.spill_table_height <= upper[2]:
            raise ValueError("spill_table_height must lie inside the particle workspace z bounds.")

        local_points = cup_cavity_lattice(self)[0]
        local_lo = local_points.min(axis=0)
        local_hi = local_points.max(axis=0)
        source_range = (*self.curriculum_randomized_source_position_range, 0.0)
        # A radial XY envelope is conservative for every configured upright yaw and keeps this
        # validation independent of the reset bank's finite angular samples.
        local_xy_radius = max(math.hypot(float(point[0]), float(point[1])) for point in local_points)
        source_lo = (
            self.cup_reset_pos[0] - source_range[0] - local_xy_radius,
            self.cup_reset_pos[1] - source_range[1] - local_xy_radius,
            float(local_lo[2] + self.cup_reset_pos[2]),
        )
        source_hi = (
            self.cup_reset_pos[0] + source_range[0] + local_xy_radius,
            self.cup_reset_pos[1] + source_range[1] + local_xy_radius,
            float(local_hi[2] + self.cup_reset_pos[2]),
        )
        target_local_lo, target_local_hi = cube_bowl_inner_bounds(
            self.target_cup_inner_width,
            self.target_cup_inner_depth,
            self.target_cup_cavity_depth,
            self.target_cup_bottom_thickness,
        )
        target_range = (*self.curriculum_randomized_target_position_range, 0.0)
        target_lo = tuple(
            float(point + position - extent - self.particle_count_margin)
            for point, position, extent in zip(
                target_local_lo,
                self.target_cup_reset_pos,
                target_range,
                strict=True,
            )
        )
        target_hi = tuple(
            float(point + position + extent + self.particle_count_margin)
            for point, position, extent in zip(
                target_local_hi,
                self.target_cup_reset_pos,
                target_range,
                strict=True,
            )
        )
        for region_name, region_lo, region_hi in (
            ("randomized source media", source_lo, source_hi),
            ("randomized target cavity", target_lo, target_hi),
        ):
            if any(value < bound for value, bound in zip(region_lo, lower, strict=True)) or any(
                value > bound for value, bound in zip(region_hi, upper, strict=True)
            ):
                raise ValueError(f"particle_workspace bounds do not contain the {region_name}.")

    def _apply_robot_cfg(self) -> None:
        """Apply final task reset positions to the scene robot."""
        self.scene.robot.init_state.joint_pos.update(
            dict(zip([f"panda_joint{i}" for i in range(1, 8)], self.arm_home, strict=True))
        )
        self.scene.robot.init_state.joint_pos["panda_finger_joint.*"] = self.gripper_open_pos

    def _apply_solver_cfg_overrides(self) -> None:
        """Propagate final top-level controls into the constructed coupled-solver config."""
        coupled_cfg = self.sim.physics.solver_cfg
        arm_entries = [entry for entry in coupled_cfg.entries if entry.name == RIGID_ENTRY]
        if len(arm_entries) != 1:
            raise ValueError(f"Expected exactly one {RIGID_ENTRY!r} solver entry, found {len(arm_entries)}.")
        media_entries = [entry for entry in coupled_cfg.entries if entry.name == MPM_ENTRY]
        if len(media_entries) != 1:
            raise ValueError(f"Expected exactly one {MPM_ENTRY!r} solver entry, found {len(media_entries)}.")
        proxies = [
            proxy for proxy in coupled_cfg.proxies if proxy.source == RIGID_ENTRY and proxy.destination == MPM_ENTRY
        ]
        if len(proxies) != 1:
            raise ValueError(f"Expected exactly one {RIGID_ENTRY!r}-to-{MPM_ENTRY!r} proxy, found {len(proxies)}.")

        mpm_solver_cfg = _mpm_solver_cfg(self)
        mpm_solver_cfg.voxel_size = self.voxel_size
        mpm_solver_cfg.max_iterations = self.mpm_iterations
        arm_entries[0].substeps = self.rigid_entry_substeps
        media_entries[0].substeps = self.mpm_entry_substeps
        self.sim.physics.num_substeps = self.physics_substeps
        self.sim.physics.use_cuda_graph = self.use_cuda_graph
        coupled_cfg.iterations = self.proxy_iterations
        proxies[0].mass_scale = self.proxy_mass_scale

    def finalize(self) -> FrankaPourEnvCfg:
        """Return an independent config with all derived scene assets resolved."""
        resolved = deepcopy(self)
        resolved.sim.render_interval = resolved.decimation
        # Hydra and command-line overrides are applied after ``__post_init__`` constructs the
        # nested Newton solver tree. Reapply every public top-level solver control before resolving
        # derived configuration.
        resolved._apply_solver_cfg_overrides()
        # Command-line overrides are applied after ``__post_init__``. Re-resolve the custom action
        # bound so its open target cannot diverge from the physical reset configuration.
        resolved.actions.gripper_action.close_position = max(
            0.0, resolved.gripper_preload_pos - resolved.gripper_close_offset
        )
        resolved.actions.gripper_action.open_position = resolved.gripper_open_pos
        if resolved.actions.gripper_action.limit_to_preload:
            resolved.actions.gripper_action.neutral_position = resolved.gripper_preload_pos
        else:
            resolved.actions.gripper_action.neutral_position = resolved.gripper_open_pos
            resolved.actions.gripper_action.default_position = resolved.gripper_preload_pos
            if not resolved.actions.gripper_action.use_incremental_target:
                resolved.actions.gripper_action.scale = resolved.gripper_open_pos - resolved.gripper_preload_pos
        max_gripper_command = resolved._resolved_success_max_gripper_command()
        resolved._configure_reward_cfg(max_gripper_command, initialize_stage_gates=False)
        resolved.terminations.lost_grasp.params["dwell_time_s"] = resolved.lost_grasp_dwell_time_s
        resolved.terminations.lost_grasp.params["max_tcp_distance"] = resolved.success_max_tcp_distance
        resolved.terminations.lost_grasp.params["max_gripper_width_error"] = resolved.success_max_gripper_width_error
        resolved.terminations.lost_grasp.params["max_gripper_command"] = max_gripper_command
        resolved._validate_curriculum_cfg()
        resolved._validate_particle_workspace_cfg()
        resolved._apply_robot_cfg()
        if resolved.terminations.success.func is mdp.immediate_pour_success:
            resolved.terminations.success.params = {}
        else:
            resolved.terminations.success.params["dwell_time_s"] = resolved.success_dwell_time_s
            resolved.terminations.success.params["min_lift_height"] = resolved.success_min_lift_height
            resolved.terminations.success.params["max_tcp_distance"] = resolved.success_max_tcp_distance
            resolved.terminations.success.params["max_gripper_width_error"] = resolved.success_max_gripper_width_error
            resolved.terminations.success.params["max_gripper_command"] = max_gripper_command
        resolved.scene.source_cup = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/SourceCup",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=resolved.cup_reset_pos,
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
            spawn=CubeBowlSpawnerCfg(
                inner_width=resolved.source_cup_inner_width,
                inner_depth=resolved.source_cup_inner_depth,
                cavity_depth=resolved.source_cup_cavity_depth,
                wall_thickness=resolved.source_cup_wall_thickness,
                bottom_thickness=resolved.source_cup_bottom_thickness,
                display_color=(0.95, 0.82, 0.16),
                grasp_proxy_half_extents=resolved.cup_grasp_box_half,
                mass_props=MassCfg(mass=resolved.cup_mass),
                rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=False),
                collision_props=UsdPhysicsCollisionCfg(collision_enabled=True),
                physics_material=RigidBodyMaterialBaseCfg(
                    static_friction=resolved.cup_grasp_box_friction,
                    dynamic_friction=resolved.cup_grasp_box_friction,
                ),
            ),
        )
        resolved.scene.target_cup = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TargetCup",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=resolved.target_cup_reset_pos,
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
            spawn=CubeBowlSpawnerCfg(
                inner_width=resolved.target_cup_inner_width,
                inner_depth=resolved.target_cup_inner_depth,
                cavity_depth=resolved.target_cup_cavity_depth,
                wall_thickness=resolved.target_cup_wall_thickness,
                bottom_thickness=resolved.target_cup_bottom_thickness,
                display_color=(0.20, 0.55, 0.90),
                grasp_proxy_half_extents=None,
                rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
                physics_material=RigidBodyMaterialBaseCfg(
                    static_friction=resolved.target_cup_friction,
                    dynamic_friction=resolved.target_cup_friction,
                ),
            ),
        )
        resolved.scene.media = build_media_object_cfg(
            resolved,
            resolved.cup_reset_pos,
            (0.0, 0.0, 0.0, 1.0),
        )
        mpm_solver_cfg = _mpm_solver_cfg(resolved)
        mpm_solver_cfg.max_active_cell_count = _resolve_mpm_cell_cap(resolved)
        return resolved


@configclass
class FrankaPourEnvCfg_RESET_DATASET(FrankaPourEnvCfg):
    """Validated reset-dataset curriculum with a stage-independent reward."""

    curriculum: ResetDatasetCurriculumCfg = ResetDatasetCurriculumCfg()
    rewards: ResetDatasetRewardsCfg = ResetDatasetRewardsCfg()
    curriculum_early_target_frac: tuple[float, ...] = (0.30,) * 14
    # The dataset is generated and collision-validated offline, then restored directly at reset.
    # Training samples it through the adaptive curriculum below; frozen evaluation samples either
    # every row or the requested highest-objective grasp subset.
    reset_dataset_path: str = "datasets/franka_pour/reset_dataset.pt"
    reset_dataset_content_sha256: str | None = None
    reset_dataset_top_grasp_count: int | None = None
    # Each process retains bounded online success evidence. Training begins on the easiest rows,
    # calibrates a broad replay distribution near 50% success, and probes only the adjacent harder
    # frontier. The state is intentionally rank-local and starts fresh on RSL-RL resume because
    # environment state is not currently part of its checkpoints. The reusable sampler config
    # remains directly overridable through Hydra.
    reset_dataset_sampler: AdaptiveResetSamplerCfg = AdaptiveResetSamplerCfg()
    # A spill is irreversible once more than 30% of the media rests on the table outside both
    # vessels. At 245 particles the strict threshold terminates on particle 74.
    max_spill_fraction: float = 0.30

    def __post_init__(self):
        # Use the standard Isaac Lab relative joint-position action with the position drives
        # authored by the selected Franka USD. A 0.015 rad tracking error keeps the stiff proximal
        # USD drives within useful torque authority while improving contact-phase precision.
        self.actions.arm_action.scale = 0.015
        # OmniReset uses one region-independent binary gripper action. Keep the existing moving-
        # average filter to damp IID exploration chatter without accumulating stale open commands.
        # Adjust its coefficient to retain the same physical-time response at three times the rate.
        self.actions.gripper_action.use_incremental_target = False
        self.actions.gripper_action.binary_threshold = 0.0
        self.actions.gripper_action.alpha = 1.0 - (1.0 - self.actions.gripper_action.alpha) ** (1.0 / 3.0)
        super().__post_init__()
        # Run the policy at 30 Hz over the 120 Hz simulation. Thirteen frames span 0.4 seconds from
        # oldest to newest, matching five frames at 10 Hz, while 32 PPO steps span 1.067 seconds.
        self.decimation = 4
        self.sim.render_interval = self.decimation
        # Seven seconds gives broad reaching rows time to act while turning failed attempts over
        # more quickly. At 30 Hz this is exactly 210 policy steps.
        self.episode_length_s = 7.0
        self.observations.policy.history_length = 13
        self.is_finite_horizon = False
        # The first step with at least ``pour_target_frac`` (30%) of the media in the receiver is
        # terminal success. It deliberately has no dwell, retained-grasp, lift, or trajectory-history
        # requirement. Failure predicates are evaluated first and retain same-step precedence.
        self.terminations.success.func = mdp.immediate_pour_success
        self.terminations.success.params = {}
        self.terminations.lost_grasp.params["terminate"] = False
        self.terminations.spill.params["terminate"] = True
        self.terminations.time_out.func = mdp.unsuccessful_time_out

    def _configure_reward_cfg(self, max_gripper_command: float, *, initialize_stage_gates: bool) -> None:
        """Keep the general reward independent of curriculum and grasp thresholds."""
        del max_gripper_command, initialize_stage_gates

    def _validate_reward_cfg(self, stage_count: int) -> None:
        """Validate the general reward without introducing reset-stage dependencies."""
        for name in ("reach", "goal_distance"):
            std = float(getattr(self.rewards, name).params["std"])
            if not math.isfinite(std) or std <= 0.0:
                raise ValueError(f"{name} std must be finite and positive.")
        max_velocity = float(self.rewards.joint_velocity.params["max_velocity"])
        if not math.isfinite(max_velocity) or max_velocity <= 0.0:
            raise ValueError("joint_velocity max_velocity must be finite and positive.")

    def _validate_curriculum_cfg(self) -> None:
        super()._validate_curriculum_cfg()
        if not isinstance(self.reset_dataset_path, str):
            raise TypeError("reset_dataset_path must be a string.")
        if not self.reset_dataset_path:
            raise ValueError("reset_dataset_path must be nonempty.")
        top_grasp_count = self.reset_dataset_top_grasp_count
        if top_grasp_count is not None:
            if not isinstance(top_grasp_count, int) or isinstance(top_grasp_count, bool) or top_grasp_count <= 0:
                raise ValueError("reset_dataset_top_grasp_count must be a positive integer or None.")
            if not self.curriculum_freeze:
                raise ValueError("reset_dataset_top_grasp_count requires curriculum_freeze=True.")
        expected_hash = self.reset_dataset_content_sha256
        if expected_hash is not None and (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError("reset_dataset_content_sha256 must be a lowercase SHA-256 or None.")
        self.reset_dataset_sampler.validate_values()


@configclass
class FrankaPourEnvCfg_RESET_DATASET_EVAL(FrankaPourEnvCfg_RESET_DATASET):
    """Frozen full-distribution reset-dataset evaluation."""

    curriculum_freeze: bool = True


@configclass
class FrankaPourEnvCfg_RESET_DATASET_PLAY(FrankaPourEnvCfg_RESET_DATASET_EVAL):
    """Reset-dataset playback using the captured sparse multi-world MPM configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.use_cuda_graph = True
        self.sim.physics.use_cuda_graph = True
        # Retain the base view direction while moving the camera about one metre closer.
        self.viewer.eye = (0.9, 0.65, 0.5)
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=self.viewer.eye, lookat=self.viewer.lookat)


# Compatibility aliases for checkpoints and commands produced while this task was experimental.
@configclass
class FrankaPourEnvCfg_PLAY(FrankaPourEnvCfg):
    """Playback using the training task's captured sparse multi-world MPM configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.use_cuda_graph = True
        self.sim.physics.use_cuda_graph = True
        self.curriculum_start_stage = len(self.curriculum_stage_names) - 1
        self.curriculum_randomization_start_level = len(self.curriculum_randomization_extent_levels) - 1
        self.curriculum_freeze = True


@configclass
class FrankaPourEnvCfg_TELEOP(FrankaPourEnvCfg_PLAY):
    """Teleop preset: 1 env, no RL time-out (operator resets manually)."""

    def __post_init__(self):
        super().__post_init__()
        # SpaceMouse IK emits direct seven-joint targets; keep that operator-only interface
        # separate from the policy's relative-joint action representation.
        joint_clip = {
            joint_name: limits
            for joint_name, limits in zip(self.actions.arm_action.joint_names, PANDA_ARM_JOINT_LIMITS, strict=True)
        }
        self.actions.arm_action = mdp.CurriculumJointPositionActionCfg(
            asset_name="robot",
            joint_names=[f"panda_joint{i}" for i in range(1, 8)],
            scale=0.5,
            alpha=0.2,
            project_reference_through_stage=-1,
            use_default_offset=True,
            preserve_order=True,
            clip=joint_clip,
        )
        self.actions.gripper_action.force_open_before_phase_stage = -1
        self.actions.gripper_action.limit_to_preload = False
        self.actions.gripper_action.neutral_position = self.gripper_open_pos
        self.actions.gripper_action.default_position = self.gripper_open_pos
        self.actions.gripper_action.scale = self.gripper_open_pos - self.actions.gripper_action.close_position
        self.scene.num_envs = 1
        self.terminations.time_out = None
        self.episode_length_s = 3600.0

    def finalize(self) -> FrankaPourEnvCfg_TELEOP:
        """Preserve the operator preset's full open-to-close gripper range."""
        resolved = super().finalize()
        resolved.actions.gripper_action.default_position = resolved.gripper_open_pos
        resolved.actions.gripper_action.scale = (
            resolved.gripper_open_pos - resolved.actions.gripper_action.close_position
        )
        return resolved

    def _validate_arm_action_cfg(self) -> None:
        """Validate the operator-only absolute joint-position action."""
        arm_action = self.actions.arm_action
        # ``configclass`` validates the inherited policy action before this preset's post-init
        # replaces it with the operator controller.
        if isinstance(arm_action, mdp.RelativeJointPositionActionCfg):
            FrankaPourEnvCfg._validate_arm_action_cfg(self)
            return
        if not isinstance(arm_action, mdp.CurriculumJointPositionActionCfg):
            raise ValueError("Franka Pour teleoperation requires CurriculumJointPositionActionCfg for the arm.")
        if arm_action.joint_names != [f"panda_joint{i}" for i in range(1, 8)] or not arm_action.preserve_order:
            raise ValueError("Franka Pour teleoperation requires all seven Panda arm joints in kinematic order.")
        expected_clip = {
            joint_name: limits
            for joint_name, limits in zip(arm_action.joint_names, PANDA_ARM_JOINT_LIMITS, strict=True)
        }
        if arm_action.clip != expected_clip:
            raise ValueError("Franka Pour teleoperation requires Panda joint-limit clipping.")
        return


# Deprecated compatibility names; use the corresponding ``RESET_DATASET`` configurations. These
# aliases preserve configuration and task lookup, not compatibility with older 7-action policies.
ResetMixtureRewardsCfg = ResetDatasetRewardsCfg
ResetMixtureCurriculumCfg = ResetDatasetCurriculumCfg
FrankaPourEnvCfg_RESET_MIXTURE = FrankaPourEnvCfg_RESET_DATASET
FrankaPourEnvCfg_RESET_MIXTURE_EVAL = FrankaPourEnvCfg_RESET_DATASET_EVAL
FrankaPourEnvCfg_RESET_MIXTURE_PLAY = FrankaPourEnvCfg_RESET_DATASET_PLAY
