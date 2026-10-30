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
from isaaclab.sim.schemas import CollisionFragment, UsdPhysicsCollisionCfg, UsdPhysicsMeshCollisionCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_contrib.conveyors.surface_velocity import SurfaceVelocitySpec

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
    belt_collision_section_specs,
    belt_mesh_spec,
    guard_mesh_specs,
)
from .franka_robot_cfg import FRANKA_PANDA_CONVEYOR_CFG
from .mdp.terminations import invalid_action as invalid_policy_action

_CONTACT_GAP = 0.01
_CUBE_CONTACT_MARGIN = 0.003
_MUJOCO_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
_MUJOCO_SOLREF = (0.02, 1.0)
_SUBGOAL_TIMEOUT_S = 20.0
_TRANSFER_SEQUENCE_LENGTH = 8
_ARM_JOINT_NAMES = tuple(f"panda_joint{joint_id}" for joint_id in range(1, 8))
_FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")
_ROBOT_INITIAL_STATE = ArticulationCfg.InitialStateCfg(
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
)


def _validate_common_config(cfg: ConveyorFrankaEnvCfg) -> None:
    """Validate backend-independent timing and policy tensor contracts."""
    if not cfg.sim.use_newton_actuators:
        raise ValueError("The conveyor Franka requires the shared Newton-actuator execution path.")
    if not math.isfinite(cfg.sim.dt) or cfg.sim.dt <= 0.0 or cfg.decimation <= 0:
        raise ValueError("Simulation dt and environment decimation must be positive.")
    arm_action = cfg.actions.arm_action
    if arm_action.joint_names != list(_ARM_JOINT_NAMES) or not arm_action.preserve_order:
        raise ValueError("Arm actions must preserve the explicit panda_joint1-to-panda_joint7 ordering.")


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
    """Practice reset states near the measured transfer-success frontier."""

    reset_sampling = CurrTerm(
        func=mdp.ConveyorResetCurriculum,
        params={
            "command_name": "transfer",
            "success_monitor": mdp.SuccessMonitorCfg(monitored_history_len=50),
            "sampling": mdp.SamplerCfg(
                strategies=[
                    mdp.BetaSamplingStrategyCfg(target=0.5, kappa=2.0),
                    mdp.UniformSamplingStrategyCfg(weight=0.05),
                ],
                eps=1.0e-4,
            ),
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


def _static_cuboid(
    prim_path: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
) -> AssetBaseCfg:
    """Build a colliding table component."""
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=_collision_properties(),
            physics_material=RigidBodyMaterialBaseCfg(static_friction=0.7, dynamic_friction=0.7, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.75, metallic=0.0),
        ),
    )


def _conveyor_asset(
    prim_path: str,
    spec: MeshSpec | CuboidSpec,
    *,
    visual_material: sim_utils.PreviewSurfaceCfg | None = None,
    colliding: bool = True,
) -> AssetBaseCfg:
    """Build conveyor geometry with freely sliding contacts and optional rendering."""
    if isinstance(spec, CuboidSpec):
        spawn = sim_utils.CuboidCfg(size=spec.size)
        position = spec.position
    else:
        spawn = sim_utils.MeshFileCfg(
            mesh=sim_utils.MeshFileCfg.TriangleMeshCfg(vertices=spec.vertices, faces=spec.faces),
            mesh_collision_props=UsdPhysicsMeshCollisionCfg(mesh_approximation_name="none"),
        )
        position = (0.0, 0.0, 0.0)
    spawn.visible = visual_material is not None
    spawn.visual_material = visual_material
    if colliding:
        spawn.collision_props = _collision_properties(mujoco_priority=1)
        # Override cube/finger grasp friction at the belts and guide rails. MuJoCo
        # requires positive friction; the force model supplies belt traction.
        spawn.physics_material = NewtonMaterialPropertiesCfg(
            static_friction=1.1e-5, dynamic_friction=1.1e-5, restitution=0.0
        )
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        spawn=spawn,
    )


def _cube(
    name: str,
    color: tuple[float, float, float],
    pos: tuple[float, float, float],
) -> RigidObjectCfg:
    """Build one numbered dynamic transfer cube."""
    spawn = sim_utils.CuboidCfg(
        size=(0.04, 0.04, 0.04),
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
        mass_props=sim_utils.MassCfg(mass=0.05),
        collision_props=_collision_properties(contact_margin=_CUBE_CONTACT_MARGIN),
        # Belt priority overrides friction only for belt/cube contacts.
        physics_material=NewtonMaterialPropertiesCfg(static_friction=0.8, dynamic_friction=0.6, restitution=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.75),
    )
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
        init_state=_ROBOT_INITIAL_STATE,
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
                _conveyor_asset(
                    prim_path=f"{{ENV_REGEX_NS}}/{belt_spec.name}",
                    spec=belt_spec,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=BELT_COLOR, roughness=0.9, metallic=0.0),
                    colliding=False,
                ),
            )

            section_keys = ("top_straight", "bottom_straight", "right_turn", "left_turn")
            for section_key, spec in zip(section_keys, belt_collision_geometry_specs(side), strict=True):
                setattr(
                    self,
                    f"conveyor_{side.lower()}_{section_key}_collision",
                    _conveyor_asset(
                        prim_path=f"{{ENV_REGEX_NS}}/{spec.name}",
                        spec=spec,
                    ),
                )

            for spec in guard_mesh_specs(side):
                boundary = "inner" if spec.name.endswith("Inner") else "outer"
                setattr(
                    self,
                    f"guard_{side.lower()}_{boundary}",
                    _conveyor_asset(
                        prim_path=f"{{ENV_REGEX_NS}}/{spec.name}Collision",
                        spec=spec,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=GUARD_COLOR, roughness=0.3, metallic=0.8
                        ),
                    ),
                )

    def build_conveyor_belt_specs(self, **kwargs: float) -> tuple[SurfaceVelocitySpec, ...]:
        """Describe the velocity field on each authored collision section."""
        return tuple(
            section.belt for side in ("Left", "Right") for section in belt_collision_section_specs(side, **kwargs)
        )


@configclass
class ConveyorFrankaEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL task for commanded conveyor-to-conveyor cube transfer."""

    class_type: type | str = "{DIR}.conveyor_franka_env:ConveyorFrankaEnv"
    scene: ConveyorFrankaSceneCfg = ConveyorFrankaSceneCfg(num_envs=256, env_spacing=3.0)
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
                # the gripper, table, belts, and guard contacts.
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
        self.sim.visualizer_cfgs = VisualizerCfg(
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
