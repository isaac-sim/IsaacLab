# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based control configuration for the physical GB300 cable example."""

import math

from isaaclab_newton.controllers.ik.newton_ik_objectives_cfg import (
    NewtonIKJointLimitObjectiveCfg,
    NewtonIKPoseObjectiveCfg,
)
from isaaclab_newton.controllers.ik.newton_ik_solver_cfg import NewtonIKSolverCfg
from isaaclab_newton.envs.mdp.actions.newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg
from isaaclab_newton.physics import (
    HydroelasticSDFCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
    VBDSolverCfg,
)
from isaaclab_newton.physics.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
from isaaclab_newton.sim.schemas import NewtonCollisionCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, CableObjectCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_assets.robots.rizon_sharpa import (
    ARM_JOINT_NAMES,
    HAND_JOINT_LIMITS_RAD,
    HAND_JOINT_NAMES,
    PALM_BODY_NAME,
    RIZON4S_SHARPA_CFG,
)

from . import mdp
from .cable import CONNECTOR_SPAN_M, PATCH_CABLE_COLOR, PatchCableCfg, hanging_cable_positions
from .hydroelastic import cable_collision_pipeline
from .workcell import SOCKET_POSITION_E, UPPER_SOCKET_POSITION_E

ROBOT_BASE_POSITION_E = (0.40706384, -0.44981234, 1.78624670)
ROBOT_BASE_ROTATION_XYZW = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)
PREGRASP_ARM_RAD = (-0.28426774, -0.68145812, 0.31863682, 2.04760345, 0.40848495, 2.17470681, -2.85753436)
# Thumb on the housing roof, index and middle underneath; no clip compression.
CLOSE_HAND_RAD = (
    0.84772129,
    -0.10509452,
    0.67820725,
    -0.10205317,
    0.00001000,
    0.83700016,
    -0.08471779,
    0.86494442,
    0.00001000,
    1.24722315,
    -0.19387218,
    0.40609412,
    0.49765344,
    1.1,
    0.0,
    1.4,
    1.0,
    0.0,
    1.1,
    0.0,
    1.4,
    1.0,
)


_ARM_ACTUATORS = RIZON4S_SHARPA_CFG.actuators.copy()
# Use the authored per-joint effort limits and regularize tiny finger inertias.
_HAND_ACTUATOR = _ARM_ACTUATORS.pop("hand").replace(
    joint_effort_limit=None, stiffness=96.0, damping=2.4, armature=2.0e-4
)

_RJ45_ROBOT_CFG = RIZON4S_SHARPA_CFG.replace(
    spawn=RIZON4S_SHARPA_CFG.spawn.replace(
        func="isaaclab_tasks.contrib.gb300_sharpa_insertion.sharpa_contacts:spawn_sharpa_contact_meshes",
        collision_props={
            "/.*": [NewtonCollisionCfg(contact_gap=0.001)],
            # Tucked digits are visual-only for this three-finger pinch.
            "/right_(?:ring|pinky)_.*/.*": [sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False)],
        },
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=ROBOT_BASE_POSITION_E,
        rot=ROBOT_BASE_ROTATION_XYZW,
        joint_pos={
            **dict(zip(ARM_JOINT_NAMES, PREGRASP_ARM_RAD, strict=True)),
            **dict(zip(HAND_JOINT_NAMES, CLOSE_HAND_RAD, strict=True)),
        },
    ),
    actuators={
        **_ARM_ACTUATORS,
        "right_thumb": _HAND_ACTUATOR.replace(joint_names_expr=["right_thumb_.*"]),
        "right_fingers": _HAND_ACTUATOR.replace(joint_names_expr=["right_(?:index|middle|ring|pinky)_.*"]),
    },
)

_RIGID_BODIES = [
    r"{ENV_REGEX_NS}/Robot",
    r"{ENV_REGEX_NS}/Fixtures",
    r"{ENV_REGEX_NS}/Cable/head",
    r"{ENV_REGEX_NS}/Cable/latch",
]


@configclass
class _RJ45CollisionPipelineCfg(NewtonCollisionPipelineCfg):
    """Use frame-to-frame matching for the rigid plug and hand contacts."""

    contact_matching: str = "latest"


@configclass
class GB300SharpaSceneCfg(InteractiveSceneCfg):
    """One pregrasped short cable, two physical sockets and the fixed-base robot."""

    robot = _RJ45_ROBOT_CFG
    visual_scene: AssetBaseCfg | None = None
    visual_racks: AssetBaseCfg | None = None
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1500))
    task_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TaskLight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, -0.2, 2.55)),
        spawn=sim_utils.SphereLightCfg(intensity=1200, radius=0.25, color=(0.85, 0.93, 1.0)),
    )
    back_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BackLight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.1, -0.45, 2.4)),
        spawn=sim_utils.SphereLightCfg(intensity=3500, radius=0.35, color=(0.95, 0.97, 1.0)),
    )
    pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pedestal",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(*ROBOT_BASE_POSITION_E[:2], ROBOT_BASE_POSITION_E[2] / 2)),
        spawn=sim_utils.CuboidCfg(
            size=(0.42, 0.42, ROBOT_BASE_POSITION_E[2]),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.14, 0.17, 0.21)),
        ),
    )
    cable = CableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cable",
        init_state=CableObjectCfg.InitialStateCfg(pos=(0.52, 0.26, 2.04), rot=(math.sqrt(0.5), 0, 0, math.sqrt(0.5))),
        spawn=PatchCableCfg(
            positions=hanging_cable_positions((0.52, 0.26, 2.04), UPPER_SOCKET_POSITION_E),
            latch_damping_scale=3.0,
            physics_material=sim_utils.CableMaterialCfg(
                thickness=0.0065,
                density=1100,
                stretch_stiffness=1.0e8,
                bend_stiffness=1.0e8,
                twist_stiffness=6.0e7,
            ),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=PATCH_CABLE_COLOR),
        ),
    )
    connector = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cable",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.52, 0.26 - CONNECTOR_SPAN_M / 2, 2.04),
            rot=(math.sqrt(0.5), 0, 0, math.sqrt(0.5)),
            joint_pos={"spring_hinge": 0.0},
            joint_vel={"spring_hinge": 0.0},
        ),
        actuators={"clip": ImplicitActuatorCfg(joint_names_expr=["spring_hinge"], stiffness=None, damping=None)},
    )


@configclass
class ActionsCfg:
    """Canonical-palm pose [m, XYZW] followed by 22 absolute finger targets [rad]."""

    right_palm = NewtonInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        controller=NewtonIKSolverCfg(
            class_type="isaaclab_tasks.contrib.gb300_sharpa_insertion.ik:CapturedNewtonIKSolver",
            iterations=4,
        ),
        objectives=[
            NewtonIKPoseObjectiveCfg(
                name="right_palm",
                body_name=PALM_BODY_NAME,
                use_relative_mode=False,
                scale=1.0,
            ),
            NewtonIKJointLimitObjectiveCfg(weight=0.1),
        ],
        clip=None,
    )
    right_hand = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(HAND_JOINT_NAMES),
        preserve_order=True,
        use_default_offset=False,
        scale=1.0,
        clip=dict(zip(HAND_JOINT_NAMES, HAND_JOINT_LIMITS_RAD, strict=True)),
    )


@configclass
class ObservationsCfg:
    @configclass
    class StateCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=base_mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=base_mdp.joint_vel_rel)
        head = ObservationTermCfg(func=mdp.head_pose)
        latch = ObservationTermCfg(func=mdp.latch_state)
        cable = ObservationTermCfg(func=mdp.cable_state)
        enable_corruption = False

    state = StateCfg()


@configclass
class EventsCfg:
    reset_scene = EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})


@configclass
class GB300SharpaInsertionEnvCfg(ManagerBasedEnvCfg):
    """Absolute control of a native VBD cable with hydroelastic hand/plug contact."""

    class_type: type | str = "{DIR}.env:GB300SharpaInsertionEnv"
    scene = GB300SharpaSceneCfg(num_envs=1, env_spacing=8.0)
    actions = ActionsCfg()
    observations = ObservationsCfg()
    events = EventsCfg()
    decimation = 4
    hydroelastic_pad_pressure_gradient: float = 5.0e8
    """Compliant finger-cover pressure gradient [N/m^3]."""
    rigid_connector_contact_stiffness: float = 1.0e5
    """Connector-housing ordinary-contact stiffness [N/m]."""
    tcp_body_name: str = PALM_BODY_NAME
    target_socket_position_e: tuple[float, float, float] = SOCKET_POSITION_E
    """Target socket origin in environment coordinates [m]."""
    grasp_body_patterns: tuple[str, ...] = (
        "right_hand_C_MC",
        "right_thumb_",
        "right_index_",
        "right_middle_",
    )

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        visualizer_cfgs=VisualizerCfg(eye=(0.92, -0.02, 2.22), lookat=(0.52, 0.30, 2.075), focal_length=22.0),
        physics=NewtonCfg(
            num_substeps=4,
            use_cuda_graph=True,
            default_shape_cfg=NewtonShapeCfg(mu=1.0, ke=1.0e4, kd=0.1),
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(
                        name="rigid",
                        bodies=_RIGID_BODIES,
                        solver_cfg=MJWarpSolverCfg(
                            cone="elliptic",
                            ls_iterations=50,
                            integrator="implicitfast",
                            use_mujoco_contacts=False,
                            update_data_interval=4,
                            nconmax=2048,
                            njmax=2048,
                        ),
                    ),
                    CouplerEntryCfg(
                        name="cable",
                        bodies=[r"{ENV_REGEX_NS}/Cable/(?!(?:head|latch)(?:/|$)).*"],
                        shape_label_patterns=[r"/World/GroundPlane/.*"],
                        solver_cfg=VBDSolverCfg(
                            class_type="isaaclab_tasks.contrib.gb300_sharpa_insertion.endpoint:EndpointVBDSolver",
                            iterations=4,
                            rigid_joint_angular_ke=10.0,
                            rigid_body_contact_buffer_size=1024,
                        ),
                    ),
                ],
                # MJWarp owns the robot and plug contacts. VBD follows their poses in its own collision pass.
                outer_contact_entry="rigid",
                proxies=[
                    CouplerProxyMappingCfg(
                        source="rigid",
                        destination="cable",
                        bodies=_RIGID_BODIES.copy(),
                        mode="staggered",
                        collision_pipeline=cable_collision_pipeline,
                    )
                ],
            ),
            collision_decimation=1,
            collision_cfg=_RJ45CollisionPipelineCfg(
                sdf_hydroelastic_config=HydroelasticSDFCfg(output_contact_surface=True, margin_contact_area=1.0e-6),
            ),
        ),
    )

    def __post_init__(self) -> None:
        self.sim.render_interval = self.decimation
