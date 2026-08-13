# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based bimanual YAM cable routing with Newton MJWarp/VBD coupling."""

from __future__ import annotations

import math
import os
from pathlib import Path

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.sim.schemas import MujocoJointCfg, MujocoRigidBodyCfg
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg

import isaaclab.envs.mdp as env_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, CableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
from isaaclab_contrib.deformable import VBDSolverCfg

from . import mdp

YAM_USD_PATH = os.environ.get(
    "ISAACLAB_CABLE_ROUTING_YAM_USD_PATH",
    str(Path(__file__).resolve().parent / "assets" / "yam" / "i2rt_yam_default.usda"),
)
"""Pinned Robot Menagerie YAM, optionally replaced by an explicit local asset path."""

MANIPULATIONNET_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "manipulationnet"
BOARD_USD_PATH = str(MANIPULATIONNET_ASSET_DIR / "board.usdc")
ROUND_PEG_USD_PATH = str(MANIPULATIONNET_ASSET_DIR / "round_peg.usdc")

TABLE_TOP_Z = 0.75
BOARD_SIZE = (0.30, 0.40)
BOARD_THICKNESS = 0.00635
BOARD_TOP_Z = TABLE_TOP_Z + BOARD_THICKNESS
PEG_HEIGHT = 0.0235
PEG_RADIUS = 0.0125
PEG_SHAFT_RADIUS = 0.00475
PEG_CENTER_Z = BOARD_TOP_Z + 0.5 * PEG_HEIGHT
CABLE_LENGTH = 1.0
CABLE_SEGMENT_LENGTH = 0.01
CABLE_NUM_SEGMENTS = round(CABLE_LENGTH / CABLE_SEGMENT_LENGTH)
CABLE_THICKNESS = 0.006
CABLE_RADIUS = 0.5 * CABLE_THICKNESS
ROUTE_AXIAL_CUTOFF = 0.5 * PEG_HEIGHT + CABLE_RADIUS
"""Cable-center height range whose surface can overlap the finite peg [m]."""
CABLE_CENTER_Z = BOARD_TOP_Z + CABLE_RADIUS + 0.002
CABLE_CONTACT_FRICTION = 5.0
YAM_CONTACT_FRICTION = 5.0
FIXTURE_CONTACT_FRICTION = 0.5
CONTACT_STIFFNESS = 4.0e4
CONTACT_DAMPING = 1.0e-5
YAM_BASE_COLLISION_DEPTH = 0.017
YAM_BASE_Z = TABLE_TOP_Z + YAM_BASE_COLLISION_DEPTH
YAM_VISUAL_BASE_DEPTH = 0.07
YAM_VISUAL_BASE_WIDTH = 0.20
YAM_BOARD_GAP = 0.15
# Place the bases in front of and outside the board. Their inner visual edges are
# collinear with the board's prolonged long sides, while their board-facing visual
# edges retain a clear approach aisle. The wide local-Y base axes consequently stay
# parallel to the board's wide direction.
YAM_FRONT_X = -0.5 * (BOARD_SIZE[0] + YAM_VISUAL_BASE_DEPTH) - YAM_BOARD_GAP
YAM_LATERAL_OFFSET = 0.5 * (BOARD_SIZE[1] + YAM_VISUAL_BASE_WIDTH)
# The Menagerie YAM fingers move apart as ``left_finger`` increases, while the
# ``right_finger`` equality constraint mirrors that displacement. Consequently,
# the maximum joint target is open and zero is closed.
YAM_GRIPPER_OPEN_POS = 0.0375
YAM_GRIPPER_CLOSED_POS = 0.0

# Convert the solver-level targets to the elastic moduli authored by CableMaterialCfg.
CABLE_STRETCH_MODULUS = 2.0e5 * CABLE_SEGMENT_LENGTH / (math.pi * CABLE_RADIUS**2)
CABLE_BEND_MODULUS = 0.08 * CABLE_SEGMENT_LENGTH / (0.25 * math.pi * CABLE_RADIUS**4)

# ManipulationNet Tier-1 round pegs, mapped from the 10 mm board lattice into a centered board frame.
PEG_BASE_POSITIONS_B = (
    ((20.0 - 15.5) * 0.01, (15.0 - 20.5) * 0.01, PEG_CENTER_Z),
    ((12.0 - 15.5) * 0.01, (29.0 - 20.5) * 0.01, PEG_CENTER_Z),
)


def _make_rigid_contact_material(
    friction: float,
) -> list[UsdPhysicsRigidBodyMaterialCfg | NewtonMaterialCfg]:
    """Create one explicit rigid-contact material for Newton-backed assets."""
    return [
        UsdPhysicsRigidBodyMaterialCfg(
            static_friction=friction,
            dynamic_friction=friction,
        ),
        NewtonMaterialCfg(
            contact_stiffness=CONTACT_STIFFNESS,
            contact_damping=CONTACT_DAMPING,
        ),
    ]


def _make_neutral_rounded_cable_positions() -> list[tuple[float, float, float]]:
    """Create a smooth, exact-length and self-avoiding cable rest curve.

    The open curve follows most of a rounded rectangle near the board perimeter. Its closed-form
    construction is exact-length and self-avoiding without running the replay-bank projector at
    environment import time. Every chord is exactly one segment long, and each corner distributes
    a 90-degree turn over six segments instead of folding the cable into a raster bundle.
    """
    corner_segments = 6
    corner_step = 0.5 * math.pi / corner_segments
    corner_radius = CABLE_SEGMENT_LENGTH / (2.0 * math.sin(0.5 * corner_step))
    horizontal_segments = 18
    vertical_segments = 30
    half_horizontal = 0.5 * horizontal_segments * CABLE_SEGMENT_LENGTH
    half_vertical = 0.5 * vertical_segments * CABLE_SEGMENT_LENGTH
    positions = [(-half_horizontal, -half_vertical - corner_radius, 0.0)]

    def append_straight(heading: float, count: int) -> None:
        for _ in range(count):
            x, y, z = positions[-1]
            positions.append(
                (
                    x + CABLE_SEGMENT_LENGTH * math.cos(heading),
                    y + CABLE_SEGMENT_LENGTH * math.sin(heading),
                    z,
                )
            )

    def append_corner(center_x: float, center_y: float, start_angle: float) -> None:
        for step in range(1, corner_segments + 1):
            angle = start_angle + step * corner_step
            positions.append(
                (
                    center_x + corner_radius * math.cos(angle),
                    center_y + corner_radius * math.sin(angle),
                    0.0,
                )
            )

    append_straight(0.0, horizontal_segments)
    append_corner(half_horizontal, -half_vertical, -0.5 * math.pi)
    append_straight(0.5 * math.pi, vertical_segments)
    append_corner(half_horizontal, half_vertical, 0.0)
    append_straight(math.pi, horizontal_segments)
    append_corner(-half_horizontal, half_vertical, 0.5 * math.pi)
    append_straight(-0.5 * math.pi, vertical_segments)
    append_corner(-half_horizontal, -half_vertical, math.pi)

    if len(positions) <= CABLE_NUM_SEGMENTS:
        raise RuntimeError("Rounded cable template is shorter than the requested cable length.")
    return positions[: CABLE_NUM_SEGMENTS + 1]


def _make_yam_cfg(prim_path: str, position: tuple[float, float, float], yaw: float) -> ArticulationCfg:
    """Create one fixed-base YAM configuration at a front-edge pose."""
    return ArticulationCfg(
        prim_path=prim_path,
        articulation_root_prim_path="/Geometry/arm",
        spawn=sim_utils.UsdFileCfg(
            usd_path=YAM_USD_PATH,
            copy_from_source=False,
            physics_material=_make_rigid_contact_material(YAM_CONTACT_FRICTION),
            # The converted Menagerie package retains legacy
            # ``mjc:body:gravcomp`` metadata.  Author the current Newton schema
            # spelling explicitly and route compensation through each drive so
            # a fixed relative-joint target is a true static hold.
            rigid_props=[MujocoRigidBodyCfg(gravcomp=1.0)],
            joint_drive_props=[MujocoJointCfg(actuatorgravcomp=True)],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=position,
            rot=(0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)),
            joint_pos={
                "joint1": 0.0,
                "joint2": 0.85,
                "joint3": 0.60,
                "joint4": 0.0,
                "joint5": 0.0,
                "joint6": 0.0,
                "left_finger": YAM_GRIPPER_OPEN_POS,
                "right_finger": -YAM_GRIPPER_OPEN_POS,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=2.0,
                stiffness=400.0,
                damping=40.0,
                armature=0.02,
            ),
            "gripper_drive": ImplicitActuatorCfg(
                joint_names_expr=["left_finger"],
                effort_limit_sim=80.0,
                velocity_limit_sim=0.2,
                stiffness=2000.0,
                damping=100.0,
                armature=0.1,
            ),
            # Robot Menagerie authors right_finger as a -1 mimic of left_finger.
            # Keeping its drive passive avoids fighting the Newton equality constraint.
            "gripper_passive": ImplicitActuatorCfg(
                joint_names_expr=["right_finger"],
                effort_limit_sim=1.0,
                velocity_limit_sim=0.2,
                stiffness=0.0,
                damping=0.0,
                armature=0.1,
            ),
        },
        soft_joint_pos_limit_factor=0.95,
    )


def _make_peg_cfg(name: str, position: tuple[float, float, float]) -> RigidObjectCfg:
    """Create a kinematic visual-mesh F1 peg with primitive spool collision."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROUND_PEG_USD_PATH,
            copy_from_source=False,
            physics_material=_make_rigid_contact_material(FIXTURE_CONTACT_FRICTION),
            rigid_props=sim_utils.RigidBodyBaseCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionBaseCfg(contact_offset=0.002, rest_offset=0.0),
        ),
    )


@configclass
class CableRoutingSceneCfg(InteractiveSceneCfg):
    """Full table, board, dual YAM, two round pegs, and one Newton cable."""

    yam_left: ArticulationCfg = _make_yam_cfg(
        "{ENV_REGEX_NS}/YamLeft",
        (YAM_FRONT_X, YAM_LATERAL_OFFSET, YAM_BASE_Z),
        0.0,
    )
    yam_right: ArticulationCfg = _make_yam_cfg(
        "{ENV_REGEX_NS}/YamRight",
        (YAM_FRONT_X, -YAM_LATERAL_OFFSET, YAM_BASE_Z),
        0.0,
    )

    # The table and board are kinematic bodies instead of static shapes. Coupled Newton
    # requires every shape to have exactly one owning entry, so a static collider cannot
    # simultaneously belong to MJWarp for robot contact and VBD for cable contact. Owning
    # these fixtures in MJWarp and exposing them to VBD as proxies gives both solvers the
    # required collision geometry without duplicate shape ownership.
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5 * TABLE_TOP_Z)),
        spawn=sim_utils.CuboidCfg(
            size=(1.10, 0.80, TABLE_TOP_Z),
            rigid_props=sim_utils.RigidBodyBaseCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionBaseCfg(),
            physics_material=_make_rigid_contact_material(FIXTURE_CONTACT_FRICTION),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.23, 0.16)),
        ),
    )
    board: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Board",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_TOP_Z + 0.5 * BOARD_THICKNESS)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=BOARD_USD_PATH,
            copy_from_source=False,
            physics_material=_make_rigid_contact_material(FIXTURE_CONTACT_FRICTION),
            rigid_props=sim_utils.RigidBodyBaseCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionBaseCfg(contact_offset=0.002, rest_offset=0.0),
        ),
    )
    peg_0: RigidObjectCfg = _make_peg_cfg("Peg0", PEG_BASE_POSITIONS_B[0])
    peg_1: RigidObjectCfg = _make_peg_cfg("Peg1", PEG_BASE_POSITIONS_B[1])

    cable = CableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cable",
        init_state=CableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, CABLE_CENTER_Z)),
        spawn=sim_utils.CableCfg(
            positions=_make_neutral_rounded_cable_positions(),
            physics_material=sim_utils.CableMaterialCfg(
                thickness=CABLE_THICKNESS,
                density=1200.0,
                stretch_stiffness=CABLE_STRETCH_MODULUS,
                bend_stiffness=CABLE_BEND_MODULUS,
            ),
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.07, 0.07, 0.08)),
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(
            color=(0.20, 0.20, 0.20),
            physics_material=_make_rigid_contact_material(FIXTURE_CONTACT_FRICTION),
        ),
        collision_group=-1,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=1200.0, color=(0.85, 0.85, 0.85)),
    )


@configclass
class ActionsCfg:
    """Fourteen actions: six relative arm joints and one binary gripper per YAM."""

    left_arm = mdp.FiniteRelativeJointPositionActionCfg(
        asset_name="yam_left",
        joint_names=["joint[1-6]"],
        scale=0.04,
        use_zero_offset=True,
        preserve_order=True,
    )
    left_gripper = mdp.FiniteBinaryJointPositionActionCfg(
        asset_name="yam_left",
        joint_names=["left_finger"],
        open_command_expr={"left_finger": YAM_GRIPPER_OPEN_POS},
        close_command_expr={"left_finger": YAM_GRIPPER_CLOSED_POS},
    )
    right_arm = mdp.FiniteRelativeJointPositionActionCfg(
        asset_name="yam_right",
        joint_names=["joint[1-6]"],
        scale=0.04,
        use_zero_offset=True,
        preserve_order=True,
    )
    right_gripper = mdp.FiniteBinaryJointPositionActionCfg(
        asset_name="yam_right",
        joint_names=["left_finger"],
        open_command_expr={"left_finger": YAM_GRIPPER_OPEN_POS},
        close_command_expr={"left_finger": YAM_GRIPPER_CLOSED_POS},
    )


@configclass
class CommandsCfg:
    """Geometry-grounded route programs with staged sampling subsets."""

    route = mdp.CableRoutingCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        board_origin_b=(0.0, 0.0, BOARD_TOP_Z),
        radial_cutoff=0.05,
        axial_cutoff=ROUTE_AXIAL_CUTOFF,
    )


@configclass
class ObservationsCfg:
    """Structured privileged observations for the first learning milestone."""

    @configclass
    class GoalCfg(ObsGroup):
        route_program = ObsTerm(func=env_mdp.generated_commands, params={"command_name": "route"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PolicyCfg(ObsGroup):
        task_state = ObsTerm(func=mdp.route_task_state, params={"command_name": "route"})
        active_geometry = ObsTerm(
            func=mdp.active_goal_geometry,
            params={
                "command_name": "route",
                "cable_cfg": SceneEntityCfg("cable"),
                "left_ee_cfg": SceneEntityCfg("yam_left", body_names=["link_6"]),
                "right_ee_cfg": SceneEntityCfg("yam_right", body_names=["link_6"]),
            },
        )
        actions = ObsTerm(
            func=mdp.finite_last_action,
            params={"binary_action_names": ("left_gripper", "right_gripper")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ProprioCfg(ObsGroup):
        left_joint_pos = ObsTerm(
            func=mdp.finite_joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("yam_left", joint_names=["joint[1-6]", "left_finger", "right_finger"])},
        )
        left_joint_vel = ObsTerm(
            func=mdp.finite_joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("yam_left", joint_names=["joint[1-6]", "left_finger", "right_finger"])},
        )
        right_joint_pos = ObsTerm(
            func=mdp.finite_joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("yam_right", joint_names=["joint[1-6]", "left_finger", "right_finger"])
            },
        )
        right_joint_vel = ObsTerm(
            func=mdp.finite_joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("yam_right", joint_names=["joint[1-6]", "left_finger", "right_finger"])
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CableStateCfg(ObsGroup):
        sampled_state = ObsTerm(
            func=mdp.sampled_cable_state_b,
            params={"asset_cfg": SceneEntityCfg("cable"), "num_samples": 32},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    goal: GoalCfg = GoalCfg()
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioCfg = ProprioCfg()
    cable_state: CableStateCfg = CableStateCfg()


@configclass
class EventCfg:
    """Heterogeneous robot, fixture, and cable resets."""

    reset_scene = EventTerm(
        func=env_mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    reset_left_arm = EventTerm(
        func=env_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("yam_left", joint_names=["joint[1-6]"]),
        },
    )
    reset_right_arm = EventTerm(
        func=env_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("yam_right", joint_names=["joint[1-6]"]),
        },
    )
    reset_pegs = EventTerm(
        func=mdp.reset_peg_offsets,
        mode="reset",
        params={
            "asset_names": ("peg_0", "peg_1"),
            "base_positions_b": PEG_BASE_POSITIONS_B,
            "grid_pitch": 0.01,
        },
    )
    reset_cable = EventTerm(
        func=mdp.reset_cable_state,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cable"),
            "translation_jitter": ((-0.002, 0.002), (-0.002, 0.002)),
            "yaw_jitter": (-0.02, 0.02),
            "rest_length": CABLE_SEGMENT_LENGTH,
            # Preserve the authored VBD rest angles. Continuous SE(2) jitter and
            # independently randomized fixtures still provide heterogeneous resets.
            "max_heading_offset": 0.0,
            "num_shape_modes": 3,
            "winding_radial_cutoff": 0.05,
            "winding_axial_cutoff": ROUTE_AXIAL_CUTOFF,
            # The route command restores cable, fixtures, and robots atomically
            # when its training replay is enabled. Avoid generating a cable
            # curve here that the subsequent command reset would overwrite.
            "full_scene_replay_command_name": "route",
        },
    )


@configclass
class RewardsCfg:
    """Sparse ordered-route success reward with safety and control penalties."""

    success = RewTerm(
        func=mdp.route_success,
        weight=20.0,
        params={
            "command_name": "route",
            "failure_termination_names": ("invalid_cable", "invalid_robot_or_action"),
        },
    )
    failure = RewTerm(
        func=mdp.route_failure,
        weight=-20.0,
        params={"termination_names": ("invalid_cable", "invalid_robot_or_action")},
    )
    stretch = RewTerm(
        func=mdp.cable_stretch,
        weight=-0.25,
        params={"cable_cfg": SceneEntityCfg("cable"), "rest_length": CABLE_SEGMENT_LENGTH},
    )
    action_rate = RewTerm(
        func=mdp.finite_action_rate_l2,
        weight=-0.002,
        params={"binary_action_names": ("left_gripper", "right_gripper")},
    )
    left_joint_velocity = RewTerm(
        func=mdp.finite_joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("yam_left", joint_names=["joint[1-6]"])},
    )
    right_joint_velocity = RewTerm(
        func=mdp.finite_joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("yam_right", joint_names=["joint[1-6]"])},
    )


@configclass
class TerminationsCfg:
    """Success, numerical failure, workspace bounds, and timeout."""

    success = DoneTerm(func=mdp.route_complete, params={"command_name": "route"})
    invalid_cable = DoneTerm(
        func=mdp.cable_invalid_or_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("cable")},
    )
    invalid_robot_or_action = DoneTerm(
        func=mdp.robot_or_action_invalid,
        params={
            "robot_cfgs": [
                SceneEntityCfg("yam_left"),
                SceneEntityCfg("yam_right"),
            ]
        },
    )
    time_out = DoneTerm(func=env_mdp.time_out, time_out=True)


@configclass
class CableRoutingEnvCfg(ManagerBasedRLEnvCfg):
    """Goal-conditioned bimanual cable-routing environment using Newton only."""

    scene: CableRoutingSceneCfg = CableRoutingSceneCfg(
        num_envs=256,
        env_spacing=1.5,
        replicate_physics=True,
    )
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        use_newton_actuators=True,
        physics=NewtonCfg(
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(
                        name="rigid",
                        solver_cfg=MJWarpSolverCfg(
                            # Reset-bank settling exercises dense, stochastic
                            # bimanual contact. Match the proven Lift/Stack
                            # headroom so an unlucky reset cannot overflow the
                            # MJWarp constraint arena and contaminate poses.
                            njmax=300,
                            nconmax=200,
                            cone="elliptic",
                            ls_iterations=20,
                            integrator="implicitfast",
                            ccd_iterations=100,
                        ),
                        bodies=[
                            r"/World/envs/env_.*/YamLeft",
                            r"/World/envs/env_.*/YamRight",
                            r"/World/envs/env_.*/Table",
                            r"/World/envs/env_.*/Board",
                            r"/World/envs/env_.*/Peg0",
                            r"/World/envs/env_.*/Peg1",
                        ],
                    ),
                    CouplerEntryCfg(
                        name="cable",
                        # Dense cable contact can exceed Newton's default 64 body-body
                        # contacts per body during multigoal training.
                        # The Newton cable-pile reference uses 256; this retains all
                        # observed contacts with ample headroom at modest memory cost.
                        solver_cfg=VBDSolverCfg(iterations=10, rigid_body_contact_buffer_size=256),
                        bodies=[r"/World/envs/env_.*/Cable"],
                        include_static_shapes=True,
                    ),
                ],
                proxies=[
                    CouplerProxyMappingCfg(
                        source="rigid",
                        destination="cable",
                        bodies=[
                            r"/World/envs/env_.*/Yam(Left|Right)",
                            r"/World/envs/env_.*/(Table|Board)",
                            r"/World/envs/env_.*/Peg(0|1)",
                        ],
                        mode="lagged",
                        mass_scale=1.0,
                        # Refresh every other 1/1200 s substep. The 10-substep cadence is
                        # divisible by two, so CUDA-graph replay preserves its collision phase.
                        collide_interval=2,
                    )
                ],
                iterations=1,
            ),
            # CableCfg generates rigid capsules directly from the builder defaults; imported
            # robots and fixtures use the explicit materials above. Newton sums both shapes'
            # gaps, so 1 mm preserves early contact without the default 20 mm pair envelope.
            default_shape_cfg=NewtonShapeCfg(
                gap=0.001,
                ke=CONTACT_STIFFNESS,
                kd=CONTACT_DAMPING,
                mu=CABLE_CONTACT_FRICTION,
            ),
            num_substeps=10,
            use_cuda_graph=True,
        ),
    )

    def __post_init__(self) -> None:
        """Set control frequency, episode duration, and the overview camera."""
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.render_interval = self.decimation
        self.sim.default_visualizer_cfg = VisualizerCfg(
            eye=(1.25, -1.10, 1.55),
            lookat=(0.0, 0.0, BOARD_TOP_Z),
            focal_length=28.0,
        )

    def play_mode(self) -> None:
        """Use a small scene for interactive inspection."""
        super().play_mode()
        self.scene.num_envs = 4
        self.commands.route.reset_replay.buffer_size = 8
        self.commands.route.debug_vis = True


@configclass
class CableRoutingPeg0CCWEnvCfg(CableRoutingEnvCfg):
    """Stage 1: counterclockwise wrapping around the first peg."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.route.allowed_route_ids = (0,)


@configclass
class CableRoutingPeg1CWEnvCfg(CableRoutingEnvCfg):
    """Stage 2: clockwise wrapping around the second peg."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.route.allowed_route_ids = (1,)


@configclass
class CableRoutingTier1PegsEnvCfg(CableRoutingEnvCfg):
    """Stage 3: ordered counterclockwise then clockwise Tier-1 peg route."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.route.allowed_route_ids = (2,)


@configclass
class CableRoutingSevenGoalsEnvCfg(CableRoutingEnvCfg):
    """Joint training over all seven peg, direction, and ordering goals."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.route.allowed_route_ids = tuple(range(7))
