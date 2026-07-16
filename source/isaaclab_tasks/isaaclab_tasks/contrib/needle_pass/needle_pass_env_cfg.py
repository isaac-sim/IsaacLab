# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""General scene and measured task semantics for dVRK needle pass."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import Any

from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg, PhysxRigidBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg
from isaaclab_teleop import XrCfg

from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg, spawn_from_usd
from isaaclab.sim.utils import bind_physics_material, find_matching_prim_paths, get_current_stage
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from . import mdp
from .assets import (
    NEEDLE_ASSET,
    NEEDLE_DYNAMIC_FRICTION,
    NEEDLE_FRICTION_COMBINE_MODE,
    NEEDLE_MASS_KG,
    NEEDLE_RESTITUTION,
    NEEDLE_RESTITUTION_COMBINE_MODE,
    NEEDLE_SCALE,
    NEEDLE_STATIC_FRICTION,
    SUTURE_PAD_ASSET,
)

MAXIMUM_COMMANDED_ACCELERATION_M_S2 = 0.5
RETENTION_LOAD_SAFETY_FACTOR = 2.0
RETENTION_FRICTION_CONE_FACETS = mdp.FORCE_CLOSURE_CONE_FACETS
# The pinned PSM material at /psm/Looks/PhysicsMaterial authors these
# coefficients on both jaw collision bindings and no combine mode (PhysX's
# default is average).  Its dynamic coefficient of 10.0 is not a defensible
# steel/steel task input.  The needle material therefore declares PhysX's
# higher-priority ``min`` mode, resolving the pair to the task's dry
# steel/steel coefficients while leaving the shared PSM asset untouched.
# Static force closure uses the resolved static coefficient; the dynamic value
# is retained for runtime provenance.
DVRK_JAW_AUTHORED_STATIC_FRICTION = 1.0
DVRK_JAW_AUTHORED_DYNAMIC_FRICTION = 10.0
RESOLVED_JAW_NEEDLE_STATIC_FRICTION = min(NEEDLE_STATIC_FRICTION, DVRK_JAW_AUTHORED_STATIC_FRICTION)
RESOLVED_JAW_NEEDLE_DYNAMIC_FRICTION = min(NEEDLE_DYNAMIC_FRICTION, DVRK_JAW_AUTHORED_DYNAMIC_FRICTION)
REQUIRED_RETENTION_LOAD = mdp.required_retention_load(
    mass_kg=NEEDLE_MASS_KG,
    gravity_m_s2=9.81,
    maximum_commanded_acceleration_m_s2=MAXIMUM_COMMANDED_ACCELERATION_M_S2,
    friction_coefficient=RESOLVED_JAW_NEEDLE_STATIC_FRICTION,
    safety_factor=RETENTION_LOAD_SAFETY_FACTOR,
)
"""Conservative per-jaw load implied by the declared mass/friction inputs."""

HANDOFF_PHASE_CFG = mdp.HandoffPhaseCfg(
    engage_force_n=REQUIRED_RETENTION_LOAD.normal_force_per_jaw_n,
    disengage_force_n=0.5 * REQUIRED_RETENTION_LOAD.normal_force_per_jaw_n,
)
"""One load-backed threshold object shared by every phase-dependent term."""


@configclass
class NeedlePassPhysicsCfg(PresetCfg):
    """PhysX presets for the contact-driven needle-pass task."""

    default: PhysxCfg = PhysxCfg(
        solver_type=1,
        solve_articulation_contact_last=True,
        enable_external_forces_every_iteration=True,
        enable_enhanced_determinism=True,
        min_position_iteration_count=4,
        min_velocity_iteration_count=1,
        bounce_threshold_velocity=0.01,
        friction_correlation_distance=0.002,
    )
    physx: PhysxCfg = default


def spawn_usd_with_rigid_material(
    prim_path: str,
    cfg: UsdFileWithRigidMaterialCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: Any,
) -> Usd.Prim:
    """Spawn a USD and strongly bind one explicit rigid-body material.

    Current Isaac Lab ``UsdFileCfg`` does not expose a physics-material field.
    The stock USD spawner first creates/clones the asset; this wrapper then
    creates one material beneath every resolved clone and recursively binds it
    to collision descendants.  Binding happens during scene construction, not
    during reset, and therefore cannot write or adapt needle state.
    """

    # The pinned needle authors its rigid body on a descendant without a
    # ``MassAPI``.  The stock USD spawner only modifies existing mass schemas,
    # so applying ``mass_props`` at the referenced asset root is a no-op.  Defer
    # mass authoring until the unique rigid-body descendant has been resolved.
    spawn_cfg = cfg.replace(mass_props=None)
    prim = spawn_from_usd(prim_path, spawn_cfg, translation, orientation, **kwargs)
    resolved_prim_paths = find_matching_prim_paths(prim_path)
    if not resolved_prim_paths:
        raise RuntimeError(f"USD material binding resolved no prims for {prim_path!r}")
    stage = get_current_stage()
    for resolved_prim_path in resolved_prim_paths:
        material_path = f"{resolved_prim_path}/physicsMaterial"
        cfg.physics_material.func(material_path, cfg.physics_material)
        bind_physics_material(
            resolved_prim_path,
            material_path,
            stronger_than_descendants=True,
        )
        if cfg.mass_props is not None:
            root_prim = stage.GetPrimAtPath(resolved_prim_path)
            rigid_body_prims = [prim for prim in Usd.PrimRange(root_prim) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(rigid_body_prims) != 1:
                raise RuntimeError(
                    f"needle physical-property binding expected one rigid body beneath {resolved_prim_path!r}, "
                    f"found {[str(prim.GetPath()) for prim in rigid_body_prims]}"
                )
            rigid_body_prim = rigid_body_prims[0]
            sim_utils.define_mass_properties(str(rigid_body_prim.GetPath()), cfg.mass_props, stage=stage)
    return prim


@configclass
class UsdFileWithRigidMaterialCfg(UsdFileCfg):
    """Task-local USD spawner with an explicit rigid-body material binding."""

    func: Callable = spawn_usd_with_rigid_material
    physics_material: PhysxRigidBodyMaterialCfg = MISSING


@configclass
class NeedlePassSceneCfg(InteractiveSceneCfg):
    """Two fixed PSMs, one free needle, and four filtered jaw sensors."""

    left_psm: ArticulationCfg = MISSING
    right_psm: ArticulationCfg = MISSING

    # Keep the spawned root name distinct from the pinned USD's nested
    # ``Needle/Needle`` children. PhysX globs allow ``*`` to span path
    # separators, so a repeated leaf name makes the environment wildcard
    # resolve the root and both descendants as separate filter entries.
    needle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SutureNeedle",
        spawn=UsdFileWithRigidMaterialCfg(
            usd_path=NEEDLE_ASSET.url,
            scale=NEEDLE_SCALE,
            activate_contact_sensors=True,
            rigid_props=PhysxRigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=0.01,
                angular_damping=0.01,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            collision_props=PhysxCollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=NEEDLE_MASS_KG),
            physics_material=PhysxRigidBodyMaterialCfg(
                static_friction=NEEDLE_STATIC_FRICTION,
                dynamic_friction=NEEDLE_DYNAMIC_FRICTION,
                restitution=NEEDLE_RESTITUTION,
                friction_combine_mode=NEEDLE_FRICTION_COMBINE_MODE,
                restitution_combine_mode=NEEDLE_RESTITUTION_COMBINE_MODE,
            ),
        ),
        # The dVRK-specific configuration replaces this with its pinned native
        # grasp-generator pose inside the donor's closed jaws.
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.10),
            rot=(0.0, 0.0, 0.0, 1.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    # The pad is deliberately outside the reset, hand-off, and vertical drop
    # regions.  It cannot support an open-jaw counterfactual needle.
    suture_pad = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SuturePad",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0.45, -0.20)),
        spawn=sim_utils.UsdFileCfg(usd_path=SUTURE_PAD_ASSET.url),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.50)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.85, 0.85, 0.85), intensity=2500.0),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 500.0)),
    )

    left_jaw_1_needle_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/LeftPSM/psm_tool_gripper1_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/SutureNeedle"],
        track_pose=True,
        update_period=0.0,
    )
    left_jaw_2_needle_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/LeftPSM/psm_tool_gripper2_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/SutureNeedle"],
        track_pose=True,
        update_period=0.0,
    )
    right_jaw_1_needle_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RightPSM/psm_tool_gripper1_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/SutureNeedle"],
        track_pose=True,
        update_period=0.0,
    )
    right_jaw_2_needle_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RightPSM/psm_tool_gripper2_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/SutureNeedle"],
        track_pose=True,
        update_period=0.0,
    )


@configclass
class ActionsCfg:
    """Stable ``7 + 2 + 7 + 2`` dVRK bimanual action declaration."""

    left_arm_action: mdp.WorldFrameDifferentialInverseKinematicsActionCfg = MISSING
    left_jaw_action: mdp.PairedJawJointPositionActionCfg = MISSING
    right_arm_action: mdp.WorldFrameDifferentialInverseKinematicsActionCfg = MISSING
    right_jaw_action: mdp.PairedJawJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Unconcatenated observations for policy recording and subtask display."""

    @configclass
    class PolicyCfg(ObsGroup):
        left_joint_pos = ObsTerm(
            func=mdp.joint_position,
            params={"asset_cfg": SceneEntityCfg("left_psm")},
        )
        left_joint_vel = ObsTerm(
            func=mdp.joint_velocity,
            params={"asset_cfg": SceneEntityCfg("left_psm")},
        )
        right_joint_pos = ObsTerm(
            func=mdp.joint_position,
            params={"asset_cfg": SceneEntityCfg("right_psm")},
        )
        right_joint_vel = ObsTerm(
            func=mdp.joint_velocity,
            params={"asset_cfg": SceneEntityCfg("right_psm")},
        )
        left_ee_pose_w = ObsTerm(
            func=mdp.end_effector_pose_w,
            params={"asset_cfg": SceneEntityCfg("left_psm")},
        )
        right_ee_pose_w = ObsTerm(
            func=mdp.end_effector_pose_w,
            params={"asset_cfg": SceneEntityCfg("right_psm")},
        )
        needle_pose_w = ObsTerm(func=mdp.needle_pose_w)
        needle_velocity_w = ObsTerm(func=mdp.needle_velocity_w)
        jaw_needle_contact_force = ObsTerm(func=mdp.jaw_needle_contact_force)
        handoff_phase = ObsTerm(func=mdp.handoff_phase, params={"phase_cfg": HANDOFF_PHASE_CFG})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        donor_hold = ObsTerm(
            func=mdp.phase_at_least,
            params={"phase_cfg": HANDOFF_PHASE_CFG, "phase": mdp.HandoffPhase.DONOR_HOLD},
        )
        co_hold = ObsTerm(
            func=mdp.phase_at_least,
            params={"phase_cfg": HANDOFF_PHASE_CFG, "phase": mdp.HandoffPhase.CO_HOLD},
        )
        receiver_only_hold = ObsTerm(
            func=mdp.phase_at_least,
            params={"phase_cfg": HANDOFF_PHASE_CFG, "phase": mdp.HandoffPhase.RECEIVER_ONLY_HOLD},
        )
        retained_lift = ObsTerm(
            func=mdp.phase_at_least,
            params={"phase_cfg": HANDOFF_PHASE_CFG, "phase": mdp.HandoffPhase.RETAINED_LIFT},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class EventCfg:
    """Deterministic reset with no physics settling inside the event."""

    reset_all = EventTerm(
        func=mdp.reset_needle_pass_to_default,
        mode="reset",
        params={"phase_cfg": HANDOFF_PHASE_CFG},
    )


@configclass
class RewardsCfg:
    """Measured phase progress rewards; reward is not used to establish success."""

    phase_progress = RewTerm(
        func=mdp.handoff_phase_progress,
        weight=1.0,
        params={"phase_cfg": HANDOFF_PHASE_CFG},
    )
    retained_lift = RewTerm(
        func=mdp.retained_lift_bonus,
        weight=5.0,
        params={"phase_cfg": HANDOFF_PHASE_CFG},
    )


@configclass
class TerminationsCfg:
    """Recorder-compatible success and separate physical failure terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.success, params={"phase_cfg": HANDOFF_PHASE_CFG})
    needle_dropped_or_out_of_bounds = DoneTerm(
        func=mdp.needle_dropped_or_out_of_bounds,
        params={"phase_cfg": HANDOFF_PHASE_CFG},
    )


@configclass
class NeedlePassEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based needle pass that starts held by the donor and transfers by contact."""

    scene: NeedlePassSceneCfg = NeedlePassSceneCfg(
        num_envs=256,
        env_spacing=1.25,
        replicate_physics=True,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    commands = None
    curriculum = None

    xr: XrCfg = XrCfg(anchor_pos=(0.0, -0.45, -0.10))

    def __post_init__(self):
        self.seed = 42
        self.decimation = 1
        self.episode_length_s = 30.0
        self.sim.dt = 1.0 / 240.0
        self.sim.render_interval = 1
        self.sim.physics = NeedlePassPhysicsCfg()
        # A near-overhead surgical view keeps both PSM jaws and the free needle
        # visible during the exchange; the previous oblique view let the right
        # arm occlude the channel contact in recorded validation episodes.
        self.viewer.eye = (0.0, -0.12, 0.45)
        self.viewer.lookat = (0.0, 0.0, 0.055)


__all__ = [
    "ActionsCfg",
    "EventCfg",
    "HANDOFF_PHASE_CFG",
    "DVRK_JAW_AUTHORED_DYNAMIC_FRICTION",
    "DVRK_JAW_AUTHORED_STATIC_FRICTION",
    "MAXIMUM_COMMANDED_ACCELERATION_M_S2",
    "NeedlePassEnvCfg",
    "NeedlePassPhysicsCfg",
    "NeedlePassSceneCfg",
    "ObservationsCfg",
    "RewardsCfg",
    "REQUIRED_RETENTION_LOAD",
    "RETENTION_FRICTION_CONE_FACETS",
    "RESOLVED_JAW_NEEDLE_DYNAMIC_FRICTION",
    "RESOLVED_JAW_NEEDLE_STATIC_FRICTION",
    "RETENTION_LOAD_SAFETY_FACTOR",
    "TerminationsCfg",
    "UsdFileWithRigidMaterialCfg",
    "spawn_usd_with_rigid_material",
]
