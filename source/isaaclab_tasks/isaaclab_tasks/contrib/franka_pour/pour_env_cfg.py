# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for reset-dataset Franka pouring with proxy-coupled MPM media."""

from __future__ import annotations

import math
import os
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
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
from isaaclab.sim.schemas import MassCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from . import mdp
from .geometry import (
    CUP_GRASP_HEIGHT,
    CUP_GRASP_TCP_QUAT_C,
    MPM_COLLIDER_MARGIN,
    SOURCE_CUP_GEOMETRY,
    TARGET_CUP_GEOMETRY,
)
from .media import build_media_object_cfg, configure_media_object_fill, media_particle_count
from .reset_sampler import ResetDatasetSamplerCfg

RIGID_ENTRY = "arm"
MPM_ENTRY = "media"

FRANKA_POUR_ROBOT_ASSET_PATH = "Robots/FrankaEmika/franka_panda.usda"
FRANKA_POUR_CUPS_ASSET_PATH = "Contrib/MPM/Pour/franka_pour_cups.usda"
FRANKA_POUR_RESET_DATASET_ASSET_PATH = "Contrib/MPM/Pour/reset_dataset.pt"
FRANKA_POUR_ROBOT_ASSET_ID = f"{ISAACLAB_NUCLEUS_DIR}/{FRANKA_POUR_ROBOT_ASSET_PATH}"
FRANKA_POUR_ROBOT_PHYSICS_PAYLOAD_SHA256 = "0dc38454f02ea14d9ddd2437995fdc7c4a65634443cacdcc2a04e3de25655e00"
FRANKA_POUR_CUPS_ASSET_ID = f"{ISAACLAB_NUCLEUS_DIR}/{FRANKA_POUR_CUPS_ASSET_PATH}"
FRANKA_POUR_RESET_DATASET_ASSET_ID = f"{ISAACLAB_NUCLEUS_DIR}/{FRANKA_POUR_RESET_DATASET_ASSET_PATH}"
FRANKA_POUR_CUPS_USD_PATH = os.environ.get("ISAACLAB_FRANKA_POUR_CUPS_USD_PATH", FRANKA_POUR_CUPS_ASSET_ID)
FRANKA_POUR_ARM_DRIVE_STIFFNESS = MappingProxyType(
    {
        "panda_joint[1-2]": 1000.0,
        "panda_joint[3-4]": 750.0,
        "panda_joint[5-7]": 300.0,
    }
)
FRANKA_POUR_ARM_DRIVE_DAMPING = MappingProxyType(
    {
        "panda_joint[1-2]": 20.0,
        "panda_joint[3-4]": 4.0,
        "panda_joint[5-7]": 2.0,
    }
)

_ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
_ARM_HOME = (
    -1.07505691,
    0.76868522,
    0.53213346,
    -2.93226814,
    2.50670838,
    1.40047050,
    0.21146376,
)
_GRIPPER_OPEN_POSITION = 0.04
_SOURCE_CUP_POSITION = (0.5, 0.0, 0.0)
_TARGET_CUP_POSITION = (0.5, -0.18, 0.0)
_SOURCE_CUP_MASS = 0.05
_SOURCE_CUP_GRASP_FRICTION = 2.0
_TARGET_CUP_FRICTION = 0.8
_MEDIA_FILL_LEVEL = 0.70
_MEDIA_FILL_RESOLUTION = 0.005
_MPM_VOXEL_SIZE = 0.015
_MPM_PARTICLES_PER_CELL = 3.0


def _source_cup_asset_cfg() -> RigidObjectCfg:
    """Build the source-cup asset authored by the task scene."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SourceCup",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_SOURCE_CUP_POSITION, rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=UsdFileCfg(
            usd_path=FRANKA_POUR_CUPS_USD_PATH,
            variants={"cupType": "source"},
            mass_props=MassCfg(mass=_SOURCE_CUP_MASS),
            rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=False),
            physics_material=RigidBodyMaterialBaseCfg(
                static_friction=_SOURCE_CUP_GRASP_FRICTION,
                dynamic_friction=_SOURCE_CUP_GRASP_FRICTION,
            ),
        ),
    )


def _target_cup_asset_cfg() -> RigidObjectCfg:
    """Build the receiver asset authored by the task scene."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetCup",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_TARGET_CUP_POSITION, rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=UsdFileCfg(
            usd_path=FRANKA_POUR_CUPS_USD_PATH,
            variants={"cupType": "target"},
            rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
            physics_material=RigidBodyMaterialBaseCfg(
                static_friction=_TARGET_CUP_FRICTION,
                dynamic_friction=_TARGET_CUP_FRICTION,
            ),
        ),
    )


def _media_asset_cfg() -> MPMObjectCfg:
    """Build the source-cup particle grid authored by the task scene."""
    return build_media_object_cfg(
        cup_pos=_SOURCE_CUP_POSITION,
        cup_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        source_inner_width=SOURCE_CUP_GEOMETRY.inner_width,
        source_inner_depth=SOURCE_CUP_GEOMETRY.inner_depth,
        source_cavity_depth=SOURCE_CUP_GEOMETRY.cavity_depth,
        source_bottom_thickness=SOURCE_CUP_GEOMETRY.bottom_thickness,
        fill_level=_MEDIA_FILL_LEVEL,
        fill_resolution=_MEDIA_FILL_RESOLUTION,
        voxel_size=_MPM_VOXEL_SIZE,
        particles_per_cell=_MPM_PARTICLES_PER_CELL,
        collider_margin=MPM_COLLIDER_MARGIN,
        material=MPMParticleMaterialCfg(
            density=1500.0,
            friction=0.7,
            yield_pressure=1.0e12,
        ),
    )


FRANKA_POUR_ROBOT_USD_PATH = os.environ.get("ISAACLAB_FRANKA_POUR_ROBOT_USD_PATH", FRANKA_POUR_ROBOT_ASSET_ID)
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


def spawn_franka_with_arm_collisions(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the canonical Franka with the intended mimic and arm-collision schemas."""
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    robot_prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for root_prim in sim_utils.find_matching_prims(prim_path, stage=robot_prim.GetStage()):
        self_collision = root_prim.GetAttribute("newton:selfCollisionEnabled")
        if not self_collision or not self_collision.Set(True):
            raise RuntimeError(f"Franka asset at {root_prim.GetPath()} has no writable Newton self-collision flag.")

        finger_joints = {
            prim.GetName(): prim
            for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies())
            if prim.GetName() in {"panda_finger_joint1", "panda_finger_joint2"}
        }
        if finger_joints.keys() != {"panda_finger_joint1", "panda_finger_joint2"}:
            missing = sorted({"panda_finger_joint1", "panda_finger_joint2"}.difference(finger_joints))
            raise RuntimeError(f"Franka asset at {root_prim.GetPath()} is missing finger joints: {missing}.")

        leader_schemas = finger_joints["panda_finger_joint1"].GetMetadata("apiSchemas")
        follower_schemas = finger_joints["panda_finger_joint2"].GetMetadata("apiSchemas")
        leader_schema_names = [] if leader_schemas is None else list(leader_schemas.GetAppliedItems())
        follower_schema_names = [] if follower_schemas is None else list(follower_schemas.GetAppliedItems())
        if "NewtonMimicAPI" not in leader_schema_names or "PhysxMimicJointAPI:linear" not in follower_schema_names:
            raise RuntimeError(
                "The pinned Franka physics payload no longer has the expected duplicate finger-mimic schemas. "
                "Update the asset hash and task-side override together."
            )
        if not finger_joints["panda_finger_joint1"].RemoveAppliedSchema("NewtonMimicAPI"):
            raise RuntimeError("Failed to remove the redundant Franka Newton finger-mimic schema.")
        if finger_joints["panda_finger_joint1"].HasAPI("NewtonMimicAPI"):
            raise RuntimeError("The redundant Franka Newton finger-mimic schema is still composed after removal.")

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


@dataclass(frozen=True)
class _PourSolverTree:
    """Typed references into the task's coupled Newton configuration."""

    coupled: CouplerProxyCfg
    arm_entry: CouplerEntryCfg
    media_entry: CouplerEntryCfg
    proxy: CouplerProxyMappingCfg
    media_solver: MPMSolverCfg


def _resolve_pour_solver_tree(cfg: FrankaPourResetDatasetEnvCfg) -> _PourSolverTree:
    """Resolve and type-check the task's unique rigid, MPM, and proxy entries."""
    physics_cfg = cfg.sim.physics
    if not isinstance(physics_cfg, NewtonCfg):
        raise TypeError(f"Franka Pour requires NewtonCfg, got {type(physics_cfg).__name__}.")
    coupled = physics_cfg.solver_cfg
    if not isinstance(coupled, CouplerProxyCfg):
        raise TypeError(f"Franka Pour requires CouplerProxyCfg, got {type(coupled).__name__}.")

    arm_entries = [entry for entry in coupled.entries if entry.name == RIGID_ENTRY]
    media_entries = [entry for entry in coupled.entries if entry.name == MPM_ENTRY]
    proxies = [proxy for proxy in coupled.proxies if proxy.source == RIGID_ENTRY and proxy.destination == MPM_ENTRY]
    if len(arm_entries) != 1 or len(media_entries) != 1 or len(proxies) != 1:
        raise ValueError("Franka Pour requires exactly one arm entry, one media entry, and one arm-to-media proxy.")
    if not isinstance(arm_entries[0].solver_cfg, MJWarpSolverCfg):
        raise TypeError(f"The {RIGID_ENTRY!r} entry requires MJWarpSolverCfg.")
    if not isinstance(media_entries[0].solver_cfg, MPMSolverCfg):
        raise TypeError(f"The {MPM_ENTRY!r} entry requires MPMSolverCfg.")
    return _PourSolverTree(
        coupled=coupled,
        arm_entry=arm_entries[0],
        media_entry=media_entries[0],
        proxy=proxies[0],
        media_solver=media_entries[0].solver_cfg,
    )


def _mpm_solver_cfg(cfg: FrankaPourResetDatasetEnvCfg) -> MPMSolverCfg:
    """Return the task's MPM solver configuration."""
    return _resolve_pour_solver_tree(cfg).media_solver


def _resolve_mpm_cell_cap(cfg: FrankaPourResetDatasetEnvCfg) -> int:
    """Resolve the total MPM active-cell capacity for all independent worlds."""
    solver_cfg = _mpm_solver_cfg(cfg)
    if cfg.mpm_cell_cap_override is not None:
        capacity = int(cfg.mpm_cell_cap_override)
    elif solver_cfg.grid_type == "sparse":
        alignment = int(cfg.mpm_cell_capacity_alignment)
        if alignment <= 0:
            raise ValueError("mpm_cell_capacity_alignment must be positive.")
        particle_count = media_particle_count(cfg.scene.media)
        per_world = ((particle_count + alignment - 1) // alignment) * alignment
        capacity = per_world * int(cfg.scene.num_envs)
    else:
        capacity = int(solver_cfg.max_active_cell_count)
    if capacity <= 0:
        raise ValueError("The resolved MPM active-cell capacity must be positive.")
    return capacity


def _configure_mpm_capacities(cfg: FrankaPourResetDatasetEnvCfg) -> None:
    """Resolve world-count-dependent MPM capacities after command-line overrides."""
    _configure_media_fill(cfg)
    solver_cfg = _mpm_solver_cfg(cfg)
    solver_cfg.max_active_cell_count = _resolve_mpm_cell_cap(cfg)
    # Leave sparse hierarchy caps automatic: spilled particles can require one leaf per active cell.


def _configure_media_fill(cfg: FrankaPourResetDatasetEnvCfg) -> None:
    """Resolve fill-level-dependent media bounds after command-line overrides."""
    cfg.scene.media.init_state.pos = tuple(cfg.scene.source_cup.init_state.pos)
    cfg.scene.media.init_state.rot = tuple(cfg.scene.source_cup.init_state.rot)
    configure_media_object_fill(
        cfg.scene.media,
        source_inner_width=SOURCE_CUP_GEOMETRY.inner_width,
        source_inner_depth=SOURCE_CUP_GEOMETRY.inner_depth,
        source_cavity_depth=SOURCE_CUP_GEOMETRY.cavity_depth,
        source_bottom_thickness=SOURCE_CUP_GEOMETRY.bottom_thickness,
        fill_level=cfg.source_fill_level,
        fill_resolution=_MEDIA_FILL_RESOLUTION,
        collider_margin=MPM_COLLIDER_MARGIN,
    )


@configclass
class PourSceneCfg(InteractiveSceneCfg):
    """Lift-table scene with resolved source cup, receiver, and MPM media."""

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0, 0, 0.707, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
        ),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.usd_path = FRANKA_POUR_ROBOT_USD_PATH
    robot.spawn.func = spawn_franka_with_arm_collisions
    robot.spawn.articulation_props.enabled_self_collisions = True
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
    robot.actuators["panda_shoulder"].stiffness = {
        key: FRANKA_POUR_ARM_DRIVE_STIFFNESS[key] for key in ("panda_joint[1-2]", "panda_joint[3-4]")
    }
    robot.actuators["panda_shoulder"].damping = {
        key: FRANKA_POUR_ARM_DRIVE_DAMPING[key] for key in ("panda_joint[1-2]", "panda_joint[3-4]")
    }
    robot.actuators["panda_forearm"].stiffness = {
        "panda_joint[5-7]": FRANKA_POUR_ARM_DRIVE_STIFFNESS["panda_joint[5-7]"]
    }
    robot.actuators["panda_forearm"].damping = {"panda_joint[5-7]": FRANKA_POUR_ARM_DRIVE_DAMPING["panda_joint[5-7]"]}
    robot.spawn.joint_drive_props = [MujocoJointCfg(actuatorgravcomp=True)]
    robot.init_state.joint_pos.update(dict(zip(_ARM_JOINT_NAMES, _ARM_HOME, strict=True)))
    robot.init_state.joint_pos["panda_finger_joint.*"] = _GRIPPER_OPEN_POSITION

    source_cup: RigidObjectCfg = _source_cup_asset_cfg()
    target_cup: RigidObjectCfg = _target_cup_asset_cfg()
    media: MPMObjectCfg = _media_asset_cfg()


@configclass
class ActionsCfg:
    """Filtered arm deltas and a binary gripper command."""

    arm_action = mdp.EMARelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[f"panda_joint{index}" for index in range(1, 8)],
        preserve_order=True,
        scale=0.03,
        use_zero_offset=True,
        alpha=0.20,
    )
    gripper_action = mdp.CurriculumGripperPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        alpha=1.0 - (1.0 - 0.2) ** (1.0 / 3.0),
        close_position=0.021,
        neutral_position=0.04,
        default_position=0.024,
        contact_min_deflection=0.001,
    )


@configclass
class ResetDatasetObservationsCfg:
    """Current robot and task state with compact MPM summaries."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Robot, cup, and target state available to the actor."""

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
            self.history_length = 0

    @configclass
    class MediaCfg(ObsGroup):
        """Permutation-invariant particle and cup-motion state."""

        cup_velocity = ObsTerm(
            func=mdp.normalized_cup_velocity_obs,
            params={"max_surface_speed": 0.5, "surface_radius": 0.13},
            clip=(-2.0, 2.0),
        )
        particle_fractions = ObsTerm(func=mdp.particle_fractions_obs)
        particle_source_state = ObsTerm(func=mdp.particle_source_state_obs)
        particle_transfer = ObsTerm(func=mdp.particle_transfer_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Exact simulation state available only to the asymmetric critic."""

        success_dwell = ObsTerm(func=mdp.success_dwell_obs)
        lost_grasp_dwell = ObsTerm(func=mdp.lost_grasp_dwell_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    media: MediaCfg = MediaCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class ResetDatasetRewardsCfg:
    """Terminal-sparse pouring reward with small action regularizers."""

    success = RewTerm(func=mdp.pour_success_bonus, weight=5.0)
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-1.0e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-4)
    failure = RewTerm(func=mdp.terminal_failure, weight=-1.0, params={"include_time_out": False})


@configclass
class TerminationsCfg:
    """Safety failures, immediate particle-transfer success, and neutral timeout."""

    failure = DoneTerm(func=mdp.nonfinite_failure)
    extreme_rigid_state = DoneTerm(func=mdp.extreme_rigid_state)
    source_receiver_overlap = DoneTerm(func=mdp.source_receiver_overlap, params={"clearance": 0.001})
    lost_grasp = DoneTerm(
        func=mdp.lost_lifted_grasp,
        params={
            "dwell_time_s": 0.05,
            "max_tcp_distance": 0.018,
            "max_gripper_width_error": 0.006,
            "max_gripper_command": 0.027,
            "terminate": False,
        },
    )
    spill = DoneTerm(func=mdp.excessive_spill, params={"terminate": True})
    particle_out_of_bounds = DoneTerm(func=mdp.particle_out_of_bounds)
    success = DoneTerm(func=mdp.immediate_pour_success, params={})
    learning_progress_context = DoneTerm(
        func=mdp.PourResetLearningProgress,
        params={
            "minimum_progress": 0.08,
            "minimum_episode_steps": 3,
            "potential_params": {
                "approach_position_std": 0.20,
                "approach_orientation_std": 0.75,
                "approach_open_hand_fraction": 0.15,
                "grasp_target_height": 0.10,
                "grasp_reach_std": 0.025,
                "grasp_preload_position": 0.024,
                "grasp_fraction": 0.40,
                "source_mouth_height": 0.119,
                "target_rim_height": 0.074,
                "target_clearance_height": 0.15,
                "alignment_std": 0.20,
                "tilt_alignment_radius": 0.15,
                "target_tilt": math.radians(140.0),
            },
        },
    )
    time_out = DoneTerm(func=mdp.unsuccessful_time_out, time_out=True)


@configclass
class EventsCfg:
    """Reset the scene from the row selected by the curriculum term."""

    reset_scene = EventTerm(func=mdp.reset_pour_scene, mode="reset")


@configclass
class ResetDatasetCurriculumCfg:
    """Adaptive replay over the validated external reset dataset."""

    reset_dataset = CurrTerm(func=mdp.PourResetDatasetCurriculum)


@configclass
class FrankaPourResetDatasetEnvCfg(ManagerBasedRLEnvCfg):
    """Registered Franka Pour task using an externally generated reset dataset."""

    scene: PourSceneCfg = PourSceneCfg(num_envs=2, env_spacing=2.5)
    observations: ResetDatasetObservationsCfg = ResetDatasetObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: ResetDatasetRewardsCfg = ResetDatasetRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: ResetDatasetCurriculumCfg = ResetDatasetCurriculumCfg()

    tcp_body_name: str = "panda_hand"
    tcp_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.107)
    tcp_offset_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    grasp_contact_ke: float = 1.0e5
    grasp_contact_kd: float = 5.0e2
    grasp_contact_kf: float = 1.0e3
    collider_margin: float = 0.002

    source_fill_level: float = _MEDIA_FILL_LEVEL
    """Initial source-cup fill height as a fraction of the usable cavity height."""

    pour_target_frac: float = 0.70
    """Fraction of the initial particles that must reach the receiver for success."""

    particle_count_margin: float = 0.003
    spill_table_height: float = 0.0
    max_spill_fraction: float = 0.10
    success_dwell_time_s: float = 0.15
    success_min_lift_height: float = 0.05

    state_bound_joint_position_margin: float = 0.05
    state_bound_max_joint_velocity: float = 20.0
    state_bound_max_cup_linear_velocity: float = 10.0
    state_bound_max_cup_angular_velocity: float = 50.0
    particle_workspace_lower_bound: tuple[float, float, float] = (-1.0, -1.0, -0.5)
    particle_workspace_upper_bound: tuple[float, float, float] = (1.5, 1.0, 1.5)

    particle_max_velocity: float = 10.0
    mpm_cell_capacity_alignment: int = 512
    mpm_cell_cap_override: int | None = None

    reset_dataset_path: str = FRANKA_POUR_RESET_DATASET_ASSET_ID
    reset_dataset_content_sha256: str | None = None
    reset_dataset_top_grasp_count: int | None = None
    reset_dataset_sampling_mode: Literal["adaptive", "uniform"] = "adaptive"
    reset_dataset_sampler: ResetDatasetSamplerCfg = ResetDatasetSamplerCfg(
        monitored_history_len=50,
        target_success_rate=0.50,
        kappa=1.0,
        epsilon=1.0e-4,
        # Reserve 50% of assignments for cyclic replay.
        uniform_fraction=0.50,
    )
    curriculum_freeze: bool = False

    def __post_init__(self):
        """Set the canonical control rate, horizon, viewer, and coupled Newton solver."""
        from isaaclab_visualizers.kit import KitVisualizerCfg

        self.decimation = 4
        self.episode_length_s = 12.0
        self.is_finite_horizon = False
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.use_newton_actuators = True
        self.sim.default_visualizer_cfg = KitVisualizerCfg(
            eye=(1.4, 1.4, 0.9),
            lookat=(0.5, 0.0, 0.1),
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
                            # njmax must exceed the maximum per-world constraint count.
                            njmax=2560,
                            nconmax=400,
                        ),
                        bodies=[
                            r"/World/envs/env_.*/Robot",
                            r"/World/envs/env_.*/Table",
                            r"/World/envs/env_.*/SourceCup",
                            r"/World/envs/env_.*/TargetCup",
                        ],
                        include_static_shapes=True,
                        substeps=3,
                    ),
                    CouplerEntryCfg(
                        name=MPM_ENTRY,
                        solver_cfg=MPMSolverCfg(
                            voxel_size=_MPM_VOXEL_SIZE,
                            grid_type="sparse",
                            grid_padding=0,
                            max_active_cell_count=-1,
                            strain_basis="P0",
                            transfer_scheme="apic",
                            max_iterations=24,
                            tolerance=1.0e-4,
                            warmstart_mode="auto",
                            velocity_basis="Q1",
                            collider_basis="pic27",
                            collider_velocity_mode="forward",
                            solver="jacobi",
                            separate_worlds=True,
                        ),
                        all_particles=True,
                        bodies=[SPILL_FLOOR_LABEL_PATTERN],
                        include_static_shapes=False,
                        include_child_joints=False,
                        # The tall source payload needs a smaller MPM step than the coupled rigid solve.
                        substeps=2,
                        in_place=True,
                    ),
                ],
                proxies=[
                    CouplerProxyMappingCfg(
                        source=RIGID_ENTRY,
                        destination=MPM_ENTRY,
                        bodies=[r"/World/envs/env_.*/SourceCup", r"/World/envs/env_.*/TargetCup"],
                        # Represent the supported or grasp-constrained 50 g cup as 50 kg only in
                        # MPM's proxy view so the roughly 140 g payload cannot create unphysical recoil.
                        mass_scale=1000.0,
                        # Synchronize MPM colliders from the rigid solver's end pose so particle
                        # reactions do not lag behind a cup resting on the table.
                        mode="staggered",
                        collision_pipeline=None,
                    )
                ],
                iterations=1,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
            num_substeps=2,
            use_cuda_graph=True,
        )

    def validate_config(self) -> None:
        """Validate runtime values that otherwise fail with opaque errors."""
        self._validate_runtime_values()
        _configure_media_fill(self)
        self._validate_mdp_term_values()
        self._validate_reset_dataset_values()

    def _validate_runtime_values(self) -> None:
        """Validate allocation and observation bounds consumed directly at runtime."""
        for name, value in (
            ("source_fill_level", self.source_fill_level),
            ("pour_target_frac", self.pour_target_frac),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 < float(value) <= 1.0
            ):
                raise ValueError(f"{name} must lie in (0, 1].")
        if (
            isinstance(self.mpm_cell_capacity_alignment, bool)
            or not isinstance(self.mpm_cell_capacity_alignment, int)
            or self.mpm_cell_capacity_alignment <= 0
        ):
            raise ValueError("mpm_cell_capacity_alignment must be a positive integer.")
        if self.mpm_cell_cap_override is not None and (
            isinstance(self.mpm_cell_cap_override, bool)
            or not isinstance(self.mpm_cell_cap_override, int)
            or self.mpm_cell_cap_override <= 0
        ):
            raise ValueError("mpm_cell_cap_override must be a positive integer or None.")
        if not math.isfinite(float(self.particle_max_velocity)) or self.particle_max_velocity <= 0.0:
            raise ValueError("particle_max_velocity must be finite and positive.")
        lower = self.particle_workspace_lower_bound
        upper = self.particle_workspace_upper_bound
        if len(lower) != 3 or len(upper) != 3 or any(not math.isfinite(value) for value in (*lower, *upper)):
            raise ValueError("particle workspace bounds must each contain three finite values.")
        if any(low >= high for low, high in zip(lower, upper, strict=True)):
            raise ValueError("particle workspace lower bounds must be smaller than upper bounds.")

    def _validate_mdp_term_values(self) -> None:
        """Validate constant manager-term parameters once instead of on every policy step."""

        def params(term: Any, name: str) -> Mapping[str, Any]:
            value = term.params
            if not isinstance(value, Mapping):
                raise TypeError(f"{name}.params must be a mapping.")
            return value

        def numbers(
            values: Mapping[str, Any],
            names: tuple[str, ...],
            lower: float = -math.inf,
            upper: float = math.inf,
            *,
            lower_open: bool = False,
            upper_open: bool = False,
        ) -> None:
            for name in names:
                value = values.get(name)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                    raise ValueError(f"{name} must be finite.")
                value = float(value)
                lower_valid = value > lower if lower_open else value >= lower
                upper_valid = value < upper if upper_open else value <= upper
                if not lower_valid or not upper_valid:
                    left, right = ("(" if lower_open else "["), (")" if upper_open else "]")
                    raise ValueError(f"{name} must lie in {left}{lower}, {upper}{right}.")

        velocity = params(self.observations.media.cup_velocity, "observations.media.cup_velocity")
        numbers(velocity, ("max_surface_speed", "surface_radius"), 0.0, lower_open=True)

        failure = params(self.rewards.failure, "rewards.failure")
        overlap = params(self.terminations.source_receiver_overlap, "terminations.source_receiver_overlap")
        lost_grasp = params(self.terminations.lost_grasp, "terminations.lost_grasp")
        spill = params(self.terminations.spill, "terminations.spill")
        for values, name, label in (
            (failure, "include_time_out", "include_time_out"),
            (lost_grasp, "terminate", "lost_grasp.terminate"),
            (spill, "terminate", "spill.terminate"),
        ):
            if not isinstance(values.get(name), bool):
                raise TypeError(f"{label} must be a bool.")
        numbers(overlap, ("clearance",), 0.0)
        numbers(lost_grasp, ("dwell_time_s",), 0.0, lower_open=True)
        numbers(
            lost_grasp,
            ("max_tcp_distance", "max_gripper_width_error", "max_gripper_command"),
            0.0,
        )

        progress = params(self.terminations.learning_progress_context, "terminations.learning_progress_context")
        numbers(progress, ("minimum_progress",), 0.0, 1.0, lower_open=True, upper_open=True)
        minimum_steps = progress.get("minimum_episode_steps")
        if not isinstance(minimum_steps, int) or isinstance(minimum_steps, bool):
            raise TypeError("minimum_episode_steps must be an integer.")
        if minimum_steps < 0:
            raise ValueError("minimum_episode_steps must be nonnegative.")
        potential = progress.get("potential_params")
        if not isinstance(potential, Mapping):
            raise TypeError("potential_params must be a mapping.")
        numbers(
            potential,
            (
                "approach_position_std",
                "approach_orientation_std",
                "grasp_target_height",
                "grasp_reach_std",
                "target_clearance_height",
                "alignment_std",
                "tilt_alignment_radius",
            ),
            0.0,
            lower_open=True,
        )
        numbers(potential, ("source_mouth_height", "target_rim_height"), 0.0)
        numbers(potential, ("grasp_preload_position",))
        numbers(potential, ("approach_open_hand_fraction", "grasp_fraction"), 0.0, 1.0)
        numbers(potential, ("target_tilt",), 0.0, math.pi, lower_open=True, upper_open=True)

    def _validate_reset_dataset_values(self) -> None:
        """Validate the external reset-dataset and sampler contract."""
        if not isinstance(self.reset_dataset_path, str) or not self.reset_dataset_path:
            raise ValueError("reset_dataset_path must be a non-empty string.")
        if self.reset_dataset_sampling_mode not in ("adaptive", "uniform"):
            raise ValueError("reset_dataset_sampling_mode must be 'adaptive' or 'uniform'.")
        if self.reset_dataset_content_sha256 is not None and not _is_sha256(self.reset_dataset_content_sha256):
            raise ValueError("reset_dataset_content_sha256 must be a lowercase SHA-256 digest or None.")
        if not isinstance(self.curriculum_freeze, bool):
            raise TypeError("curriculum_freeze must be a bool.")
        if self.reset_dataset_top_grasp_count is not None:
            if (
                not isinstance(self.reset_dataset_top_grasp_count, int)
                or isinstance(self.reset_dataset_top_grasp_count, bool)
                or self.reset_dataset_top_grasp_count <= 0
            ):
                raise ValueError("reset_dataset_top_grasp_count must be a positive integer or None.")
            if not self.curriculum_freeze:
                raise ValueError("reset_dataset_top_grasp_count requires curriculum_freeze=True.")
        self.reset_dataset_sampler.validate_values()

    def play_mode(self) -> None:
        """Configure frozen single-world dataset playback."""
        from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

        super().play_mode()
        self.scene.num_envs = 1
        self.sim.default_visualizer_cfg = NewtonRTXVisualizerCfg(
            eye=(0.9, 0.65, 0.5),
            lookat=(0.5, 0.0, 0.1),
            show_particles=True,
            particle_color=tuple(self.scene.media.spawn.visual_color),
        )
        self.curriculum_freeze = True


def _float32(value: float) -> float:
    """Round one Python float to the representation stored by a float32 tensor."""
    return struct.unpack("=f", struct.pack("=f", float(value)))[0]


def _reset_dataset_task_contract(cfg: FrankaPourResetDatasetEnvCfg) -> dict[str, Any]:
    """Return the runtime values required to interpret generated reset states."""
    source_half = tuple(_float32(value) for value in SOURCE_CUP_GEOMETRY.outer_half_extents)
    target_half = tuple(_float32(value) for value in TARGET_CUP_GEOMETRY.outer_half_extents)
    gripper_action = cfg.actions.gripper_action
    gripper_open_pos = cfg.scene.robot.init_state.joint_pos["panda_finger_joint.*"]

    return {
        "robot_asset": FRANKA_POUR_ROBOT_ASSET_PATH,
        "robot_physics_payload_sha256": FRANKA_POUR_ROBOT_PHYSICS_PAYLOAD_SHA256,
        "source_box_half": source_half,
        "source_mouth_height": SOURCE_CUP_GEOMETRY.rim_height,
        "target_box_half": target_half,
        "target_rim_height": float(2.0 * target_half[2]),
        "cup_grasp_tcp_quat_c": CUP_GRASP_TCP_QUAT_C,
        "cup_grasp_height": CUP_GRASP_HEIGHT,
        "tcp_body_name": str(cfg.tcp_body_name),
        "tcp_offset_pos": tuple(float(value) for value in cfg.tcp_offset_pos),
        "tcp_offset_rot": tuple(float(value) for value in cfg.tcp_offset_rot),
        "gripper_open_pos": float(gripper_open_pos),
        "gripper_grasp_reset_target": float(gripper_action.close_position),
        "gripper_contact_min_deflection": float(gripper_action.contact_min_deflection),
        "collider_margin": float(cfg.collider_margin),
        "mpm_collider_margin": MPM_COLLIDER_MARGIN,
    }


def _is_sha256(value: object) -> bool:
    """Return whether ``value`` is a lowercase hexadecimal SHA-256 digest."""
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
