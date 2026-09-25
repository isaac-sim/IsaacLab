# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the force-driven conveyor and Franka demonstration scene."""

from __future__ import annotations

import math
import re

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_newton.sim.schemas import MujocoCollisionCfg, NewtonCollisionCfg, NewtonMaterialPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas import CollisionFragment, UsdPhysicsCollisionCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from . import mdp
from .conveyor_geometry import (
    BELT_COLOR,
    BELT_INNER_STRAIGHT_Y,
    BELT_OUTER_STRAIGHT_Y,
    CUBE_COLORS,
    CUBE_INNER_SLOT_X,
    CUBE_OUTER_SLOT_X,
    GUARD_COLOR,
    CuboidSpec,
    MeshSpec,
    belt_collision_geometry_specs,
    belt_mesh_spec,
    guard_mesh_specs,
)
from .franka_robot_cfg import FRANKA_PANDA_CONVEYOR_CFG
from .mdp.terminations import invalid_action as invalid_policy_action

_DYNAMIC_PROPERTIES = sim_utils.RigidBodyBaseCfg()
_CONTACT_GAP = 0.01
_CUBE_CONTACT_MARGIN = 0.003
_MUJOCO_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
_MUJOCO_SOLREF = (0.02, 1.0)
_SUBGOAL_TIMEOUT_S = 20.0
_TRANSFER_SEQUENCE_LENGTH = 8
_ARM_JOINT_NAMES = tuple(f"panda_joint{joint_id}" for joint_id in range(1, 8))
_FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")


def _validate_common_config(cfg: ConveyorFrankaEnvCfg) -> None:
    """Validate backend-independent timing and policy tensor contracts."""
    cfg.conveyor_force.validate_config()
    if not cfg.sim.use_newton_actuators:
        raise ValueError("The conveyor Franka requires the shared Newton-actuator execution path.")
    if not math.isfinite(cfg.sim.dt) or cfg.sim.dt <= 0.0 or cfg.decimation <= 0:
        raise ValueError("Simulation dt and environment decimation must be positive.")
    arm_action = cfg.actions.arm_action
    if arm_action.joint_names != list(_ARM_JOINT_NAMES) or not arm_action.preserve_order:
        raise ValueError("Arm actions must preserve the explicit panda_joint1-to-panda_joint7 ordering.")
    if not math.isfinite(arm_action.max_delta) or arm_action.max_delta <= 0.0:
        raise ValueError("Arm max_delta must be finite and positive.")
    if not math.isfinite(arm_action.joint_limit_margin) or arm_action.joint_limit_margin < 0.0:
        raise ValueError("Arm joint_limit_margin must be finite and non-negative.")
    lower = arm_action.workspace_lower
    upper = arm_action.workspace_upper
    if len(lower) != len(_ARM_JOINT_NAMES) or len(upper) != len(_ARM_JOINT_NAMES):
        raise ValueError("Arm workspace bounds must contain one value per controlled joint.")
    if any(not math.isfinite(value) for value in (*lower, *upper)):
        raise ValueError("Arm workspace bounds must be finite.")
    if any(low >= high for low, high in zip(lower, upper, strict=True)):
        raise ValueError("Every arm workspace lower bound must be less than its upper bound.")


def _collision_properties(contact_margin: float = 0.0, mujoco_priority: int = 0) -> list[CollisionFragment]:
    """Build explicit Newton and MuJoCo contact properties for one collider."""
    return [
        UsdPhysicsCollisionCfg(collision_enabled=True),
        NewtonCollisionCfg(contact_margin=contact_margin, contact_gap=_CONTACT_GAP),
        MujocoCollisionCfg(
            condim=3,
            priority=mujoco_priority,
            solimp=_MUJOCO_SOLIMP,
            solmix=1.0,
            solref=_MUJOCO_SOLREF,
        ),
    ]


def _srgb_to_linear_channel(value: float) -> float:
    """Convert an sRGB channel to the linear value expected by USD displayColor."""
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


@configclass
class ActionsCfg:
    """Relative arm and binary gripper actions."""

    arm_action = mdp.ConveyorRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(_ARM_JOINT_NAMES),
        preserve_order=True,
        scale=0.12,
        max_delta=0.12,
    )
    gripper_action = mdp.ResetBufferedGripperActionCfg(
        asset_name="robot",
        joint_names=list(_FINGER_JOINT_NAMES),
        open_command_expr={"panda_finger_joint.*": 0.04},
        close_command_expr={"panda_finger_joint.*": 0.0},
        force_close_steps=5,
    )


@configclass
class CommandsCfg:
    """Success-driven cube-transfer command."""

    transfer = mdp.ConveyorTransferCommandCfg(
        reset_event_name="reset_from_state_table",
        minimum_subgoal_steps=2,
        hold_steps=3,
        lateral_tolerance=0.055,
        maximum_cube_speed=0.65,
        minimum_finger_position=0.027,
        minimum_tool_clearance=0.055,
        minimum_progress_steps=3,
        minimum_progress=0.35,
        maximum_target_potential=5.0,
        minimum_acquisition_lift=0.025,
        maximum_acquisition_tool_distance=0.075,
        maximum_acquisition_finger_position=0.030,
    )


@configclass
class ObservationsCfg:
    """Policy observations with stable cube identity and transfer commands."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Fully observed transfer policy input."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(_ARM_JOINT_NAMES), preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(_ARM_JOINT_NAMES), preserve_order=True)},
        )
        gripper_pos = ObsTerm(
            func=mdp.gripper_joint_positions,
            params={"robot_cfg": SceneEntityCfg("robot", joint_names=list(_FINGER_JOINT_NAMES), preserve_order=True)},
        )
        objects = ObsTerm(func=mdp.transfer_object_observation)
        active_transfer = ObsTerm(func=mdp.active_transfer_features)
        target_cube = ObsTerm(func=mdp.target_cube_one_hot)
        cube_conveyors = ObsTerm(func=mdp.cube_conveyor_state)
        target_side = ObsTerm(func=mdp.target_side_one_hot)
        eef_velocity = ObsTerm(func=mdp.end_effector_velocity)
        eef_axes = ObsTerm(func=mdp.end_effector_axes)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Restore validated physical reset states."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_from_state_table = EventTerm(
        func=mdp.ConveyorResetStateTable,
        mode="reset",
        params={
            "fixed_recipe": None,
            "fixed_variant_id": None,
            "fixed_target_cube_id": None,
            "fixed_source_side_id": None,
            "belt_start_x_range": (0.30, 0.82),
            "cube_position_noise": 0.015,
            "arm_joint_noise": 0.015,
        },
    )


@configclass
class RewardsCfg:
    """Sparse transfer completion plus safety regularization rewards."""

    success = RewTerm(
        func=mdp.transfer_success_reward,
        params={"command_name": "transfer"},
        weight=600.0,
    )
    failure = RewTerm(
        func=mdp.terminal_failure,
        params={"command_name": "transfer"},
        weight=-60.0,
    )
    arm_action_l2 = RewTerm(
        func=mdp.action_term_l2,
        params={"action_name": "arm_action"},
        weight=-1.0e-3,
    )
    action_rate_l2 = RewTerm(
        func=mdp.finite_action_rate_l2,
        params={"action_names": ("arm_action", "gripper_action")},
        weight=-1.0e-3,
    )
    joint_velocity_l2 = RewTerm(
        func=mdp.finite_joint_velocity_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(_ARM_JOINT_NAMES), preserve_order=True)},
        weight=-1.0e-4,
    )


@configclass
class TerminationsCfg:
    """Safety failures and bounded training sequences."""

    cube_out_of_workspace = DoneTerm(func=mdp.cube_out_of_workspace)
    invalid_action = DoneTerm(
        func=invalid_policy_action,
        params={"action_names": ("arm_action", "gripper_action")},
    )
    nonfinite_scene_state = DoneTerm(func=mdp.nonfinite_scene_state)
    subgoal_time_out = DoneTerm(
        func=mdp.subgoal_time_out,
        params={"timeout_s": _SUBGOAL_TIMEOUT_S, "command_name": "transfer"},
        time_out=True,
    )
    transfer_sequence_time_out = DoneTerm(
        func=mdp.transfer_sequence_time_out,
        params={"maximum_transfers": _TRANSFER_SEQUENCE_LENGTH, "command_name": "transfer"},
        time_out=True,
    )


@configclass
class CurriculumCfg:
    """Adaptive phase-balanced reset-state sampling."""

    reset_sampling = CurrTerm(
        func=mdp.ConveyorResetCurriculum,
        params={
            "command_name": "transfer",
            # Shared target-rate monitor keeps each physical reset row near the
            # policy's 50% competence frontier without stale early outcomes.
            "success_monitor": mdp.SuccessMonitorCfg(
                monitored_history_len=50,
                target_success_rate=0.5,
                kappa=1.0,
                temperature=1.0,
            ),
            # Keep a deployment-facing stream while the remaining starts
            # adapt around the rolling pickup-to-placement frontier. Every
            # recipe, cube identity, and direction retains equal total mass.
            "deployment_probability_initial": 0.35,
            "deployment_probability_final": 0.90,
            "deployment_progress_start": 0.45,
            "deployment_progress_end": 0.80,
            "deployment_coverage_target": 0.50,
            # Optional staged-training control. None keeps the deployable
            # bidirectional task; a side id can focus the same adaptive reset
            # distribution on one weak direction without changing reset rows.
            "fixed_source_side_id": None,
        },
    )


@configclass
class ConveyorForceCfg:
    """Configuration for force-based conveyor traction."""

    speed: float = 0.35
    """Tangential conveyor surface speed [m/s]."""

    friction: float = 0.5
    """Coulomb friction coefficient used to limit traction."""

    normal_threshold: float = 0.997
    """Minimum upward contact-normal alignment in the range [0, 1]."""

    startup_duration_s: float = 1.0
    """Duration over which conveyor traction ramps to full speed [s]."""

    transported_body_pattern: str = r"(?:^|/)Cube_?[0-3](?:/|$)"
    """Regular expression selecting rigid bodies that receive conveyor forces."""

    transported_body_count_per_env: int = 4
    """Expected number of transported rigid bodies in each environment."""

    def __post_init__(self) -> None:
        """Validate conveyor force parameters."""
        self.validate_config()

    def validate_config(self) -> None:
        """Validate final conveyor-force values after overrides are applied."""
        if not math.isfinite(self.speed) or self.speed < 0.0:
            raise ValueError(f"Conveyor speed must be non-negative, got {self.speed}.")
        if not math.isfinite(self.friction) or self.friction < 0.0:
            raise ValueError(f"Conveyor friction must be non-negative, got {self.friction}.")
        if not math.isfinite(self.normal_threshold) or not 0.0 <= self.normal_threshold <= 1.0:
            raise ValueError(f"Conveyor normal threshold must be in [0, 1], got {self.normal_threshold}.")
        if not math.isfinite(self.startup_duration_s) or self.startup_duration_s <= 0.0:
            raise ValueError(f"Conveyor startup duration must be positive, got {self.startup_duration_s}.")
        if self.transported_body_count_per_env <= 0:
            raise ValueError(
                "Conveyor transported body count per environment must be positive, got "
                f"{self.transported_body_count_per_env}."
            )
        try:
            re.compile(self.transported_body_pattern)
        except re.error as exc:
            raise ValueError(f"Invalid conveyor transported-body pattern: {self.transported_body_pattern!r}.") from exc


@sim_utils.clone
def _spawn_shape_with_display_color(
    prim_path: str,
    cfg: sim_utils.ShapeCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn a primitive and author a renderer-independent USD display color."""
    if isinstance(cfg, sim_utils.CuboidCfg):
        prim = sim_utils.spawn_cuboid(prim_path, cfg, translation, orientation, **kwargs)
    elif isinstance(cfg, sim_utils.CylinderCfg):
        prim = sim_utils.spawn_cylinder(prim_path, cfg, translation, orientation, **kwargs)
    elif isinstance(cfg, sim_utils.CapsuleCfg):
        prim = sim_utils.spawn_capsule(prim_path, cfg, translation, orientation, **kwargs)
    elif isinstance(cfg, sim_utils.SphereCfg):
        prim = sim_utils.spawn_sphere(prim_path, cfg, translation, orientation, **kwargs)
    else:
        raise TypeError(f"Unsupported colored primitive configuration: {type(cfg).__name__}")

    if cfg.visual_material is not None:
        from pxr import Usd, UsdGeom

        display_color = tuple(_srgb_to_linear_channel(value) for value in cfg.visual_material.diffuse_color)
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Gprim):
                UsdGeom.Gprim(child).CreateDisplayColorAttr([display_color])
    return prim


@sim_utils.clone
def _spawn_hidden_collision_mesh(
    prim_path: str,
    cfg: sim_utils.MeshCustomCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn a collision-only custom mesh and hide it from all visualizers."""
    prim = sim_utils.spawn_mesh_custom(prim_path, cfg, translation, orientation, **kwargs)
    sim_utils.set_prim_visibility(prim, False)
    return prim


def _static_cuboid(
    prim_path: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
    rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    friction: float = 0.7,
    roughness: float = 0.75,
    metallic: float = 0.0,
) -> AssetBaseCfg:
    """Build a static colliding cuboid configuration."""
    spawn = sim_utils.CuboidCfg(
        size=size,
        collision_props=_collision_properties(),
        physics_material=RigidBodyMaterialBaseCfg(
            static_friction=friction,
            dynamic_friction=friction,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            roughness=roughness,
            metallic=metallic,
        ),
    )
    spawn.func = _spawn_shape_with_display_color
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
        spawn=spawn,
    )


def _visual_mesh(
    prim_path: str,
    spec: MeshSpec,
    color: tuple[float, float, float],
    roughness: float,
    metallic: float,
) -> AssetBaseCfg:
    """Build a non-colliding custom mesh used only for rendering."""
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.MeshCustomCfg(
            vertices=spec.vertices,
            faces=spec.faces,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=roughness,
                metallic=metallic,
            ),
        ),
    )


def _collision_material(friction: float) -> NewtonMaterialPropertiesCfg:
    """Build a contact material for surfaces using the task-wide raw MuJoCo response."""
    return NewtonMaterialPropertiesCfg(
        static_friction=friction,
        dynamic_friction=friction,
        restitution=0.0,
    )


def _hidden_collision_mesh(
    prim_path: str,
    spec: MeshSpec,
    friction: float,
    mujoco_priority: int,
) -> AssetBaseCfg:
    """Build a hidden static triangle-mesh collider."""
    spawn = sim_utils.MeshCustomCfg(
        vertices=spec.vertices,
        faces=spec.faces,
        visible=False,
        collision_props=_collision_properties(mujoco_priority=mujoco_priority),
        physics_material=_collision_material(friction),
    )
    spawn.func = _spawn_hidden_collision_mesh
    return AssetBaseCfg(prim_path=prim_path, spawn=spawn)


def _hidden_collision_cuboid(
    prim_path: str,
    spec: CuboidSpec,
    friction: float,
    mujoco_priority: int,
) -> AssetBaseCfg:
    """Build a hidden native cuboid collider."""
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=spec.position),
        spawn=sim_utils.CuboidCfg(
            size=spec.size,
            visible=False,
            collision_props=_collision_properties(mujoco_priority=mujoco_priority),
            physics_material=_collision_material(friction),
        ),
    )


def _hidden_collision_geometry(
    prim_path: str,
    spec: MeshSpec | CuboidSpec,
    friction: float,
    mujoco_priority: int,
) -> AssetBaseCfg:
    """Build hidden collision geometry while preferring native primitives where possible."""
    if isinstance(spec, CuboidSpec):
        return _hidden_collision_cuboid(
            prim_path=prim_path,
            spec=spec,
            friction=friction,
            mujoco_priority=mujoco_priority,
        )
    return _hidden_collision_mesh(
        prim_path=prim_path,
        spec=spec,
        friction=friction,
        mujoco_priority=mujoco_priority,
    )


def _cube(
    name: str,
    color: tuple[float, float, float],
    pos: tuple[float, float, float],
) -> RigidObjectCfg:
    """Build one numbered dynamic transfer cube."""
    spawn = sim_utils.CuboidCfg(
        size=(0.04, 0.04, 0.04),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.75),
    )
    spawn.rigid_props = _DYNAMIC_PROPERTIES
    spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.05)
    spawn.collision_props = _collision_properties(contact_margin=_CUBE_CONTACT_MARGIN)
    spawn.physics_material = NewtonMaterialPropertiesCfg(
        # The belt's higher MuJoCo contact priority overrides this friction
        # only for belt/cube pairs, leaving physical finger/cube friction.
        static_friction=0.8,
        dynamic_friction=0.6,
        restitution=0.0,
    )
    spawn.func = _spawn_shape_with_display_color
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
        spawn=spawn,
    )


@configclass
class ConveyorFrankaSceneCfg(InteractiveSceneCfg):
    """Scene with two counter-rotating racetrack conveyors around a table-mounted Franka."""

    # Use the MuJoCo Menagerie-derived model with explicit manipulation gains.
    robot = FRANKA_PANDA_CONVEYOR_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.35,
                "panda_joint3": 0.0,
                "panda_joint4": -2.35,
                "panda_joint5": 0.0,
                "panda_joint6": 2.0,
                "panda_joint7": 0.78,
                "panda_finger_joint.*": 0.04,
            }
        ),
    )

    tabletop = _static_cuboid(
        prim_path="{ENV_REGEX_NS}/Tabletop",
        size=(2.0, 1.9, 0.08),
        pos=(0.50, 0.0, -0.04),
        color=(0.32, 0.34, 0.37),
    )
    table_pedestal = _static_cuboid(
        prim_path="{ENV_REGEX_NS}/TablePedestal",
        size=(0.75, 0.55, 0.76),
        pos=(0.25, 0.0, -0.46),
        color=(0.18, 0.20, 0.23),
    )

    cube_0 = _cube("Cube0", CUBE_COLORS[0], (CUBE_INNER_SLOT_X, BELT_INNER_STRAIGHT_Y, 0.06))
    cube_1 = _cube("Cube1", CUBE_COLORS[1], (CUBE_OUTER_SLOT_X, BELT_OUTER_STRAIGHT_Y, 0.06))
    cube_2 = _cube("Cube2", CUBE_COLORS[2], (CUBE_INNER_SLOT_X, -BELT_INNER_STRAIGHT_Y, 0.06))
    cube_3 = _cube("Cube3", CUBE_COLORS[3], (CUBE_OUTER_SLOT_X, -BELT_OUTER_STRAIGHT_Y, 0.06))

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.85)),
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )

    def __post_init__(self) -> None:
        """Generate visual belts, velocity-field collision sections, and guardrails."""
        for side in ("Left", "Right"):
            belt_spec = belt_mesh_spec(side)
            setattr(
                self,
                f"conveyor_{side.lower()}_belt_visual",
                _visual_mesh(
                    prim_path=f"{{ENV_REGEX_NS}}/{belt_spec.name}",
                    spec=belt_spec,
                    color=BELT_COLOR,
                    roughness=0.9,
                    metallic=0.0,
                ),
            )

            section_keys = ("top_straight", "bottom_straight", "right_turn", "left_turn")
            for section_key, spec in zip(section_keys, belt_collision_geometry_specs(side), strict=True):
                setattr(
                    self,
                    f"conveyor_{side.lower()}_{section_key}_collision",
                    _hidden_collision_geometry(
                        prim_path=f"{{ENV_REGEX_NS}}/{spec.name}",
                        spec=spec,
                        # MuJoCo requires a tiny positive value even though the force driver,
                        # rather than solver friction, supplies the belt motion.
                        friction=1.1e-5,
                        # Override cube friction only for collision-section/cube pairs.
                        mujoco_priority=1,
                    ),
                )

            for spec in guard_mesh_specs(side):
                boundary = "inner" if spec.name.endswith("Inner") else "outer"
                setattr(
                    self,
                    f"guard_{side.lower()}_{boundary}_visual",
                    _visual_mesh(
                        prim_path=f"{{ENV_REGEX_NS}}/{spec.name}Visual",
                        spec=spec,
                        color=GUARD_COLOR,
                        roughness=0.3,
                        metallic=0.8,
                    ),
                )
                setattr(
                    self,
                    f"guard_{side.lower()}_{boundary}_collision",
                    _hidden_collision_mesh(
                        prim_path=f"{{ENV_REGEX_NS}}/{spec.name}Collision",
                        spec=spec,
                        # The compact turns need freely sliding guide contacts;
                        # tangential rail friction can wedge a cube against the
                        # wall even though its belt drive remains valid.
                        friction=1.1e-5,
                        # Override the cube's grasp friction only for rail contacts.
                        mujoco_priority=1,
                    ),
                )


@configclass
class ConveyorFrankaEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL task for commanded conveyor-to-conveyor cube transfer."""

    scene: ConveyorFrankaSceneCfg = ConveyorFrankaSceneCfg(num_envs=256, env_spacing=3.0, replicate_physics=True)
    conveyor_force: ConveyorForceCfg = ConveyorForceCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    decimation: int = 2
    episode_length_s: float = _SUBGOAL_TIMEOUT_S * _TRANSFER_SEQUENCE_LENGTH

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                # Per-environment capacities include headroom for all four cubes,
                # the gripper, table, belts, and guard contacts without allocating
                # the previous 256k-contact scene-wide driver buffers.
                njmax=300,
                nconmax=200,
                impratio=1.0,
                cone="elliptic",
                update_data_interval=1,
                iterations=100,
                ls_iterations=50,
                use_mujoco_contacts=False,
                ccd_iterations=35,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(),
            # Manager decimation supplies the reference's two 120 Hz solves
            # per 60 Hz policy step, with contacts refreshed before each solve.
            collision_decimation=0,
            default_shape_cfg=NewtonShapeCfg(margin=0.0, gap=_CONTACT_GAP, ke=2.5e3, kd=100.0),
            num_substeps=1,
            use_cuda_graph=True,
            # Import render-only geometry only when a visualizer or camera needs it.
            load_visual_shapes=None,
        ),
        use_newton_actuators=True,
    )

    def __post_init__(self) -> None:
        # Any visualizer selected at runtime receives these shared camera hints.
        # Newton camera pose: position (2.13, 0.0, 1.0), pitch -23.9 degrees,
        # yaw 180 degrees. The look-at point is one unit along that view ray.
        self.sim.default_visualizer_cfg = VisualizerCfg(
            eye=(2.13, 0.0, 1.0),
            lookat=(1.2157460448, 0.0, 0.5948584132),
            max_visible_envs=1,
            randomly_sample_visible_envs=False,
        )

    def validate_config(self) -> None:
        """Validate the final task configuration after command-line overrides are applied."""
        _validate_common_config(self)
        physics = self.sim.physics
        if not isinstance(physics, NewtonCfg) or not isinstance(physics.solver_cfg, MJWarpSolverCfg):
            raise ValueError("The conveyor force driver requires the Newton MJWarp backend.")
        if physics.solver_cfg.use_mujoco_contacts:
            raise ValueError("The conveyor force driver requires the Newton collision-pipeline contact path.")
        if physics.collision_cfg is None:
            raise ValueError("The conveyor force driver requires an explicit Newton collision pipeline.")

    def play_mode(self) -> None:
        """Run continuing transfers from evenly distributed moving-belt starts."""
        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 8)
        self.events.reset_from_state_table.params["fixed_recipe"] = int(mdp.ConveyorResetRecipe.BELT)
        self.events.reset_from_state_table.params["fixed_variant_id"] = mdp.BELT_DEPLOYMENT_VARIANT
        self.events.reset_from_state_table.params["cube_position_noise"] = 0.0
        # Successful placements already transition to a new commanded cube.
        # Playback removes training-only refreshes and runs until physics leaves
        # the recoverable workspace.
        self.terminations.subgoal_time_out = None
        self.terminations.transfer_sequence_time_out = None
        self.curriculum = None
