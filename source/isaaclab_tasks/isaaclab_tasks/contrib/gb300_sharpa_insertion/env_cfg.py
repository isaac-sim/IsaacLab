# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based GB300 cable insertion with normalized PPO and absolute demo controls."""

import math

from isaaclab_newton.controllers.ik.newton_ik_objectives_cfg import (
    NewtonIKJointLimitObjectiveCfg,
    NewtonIKPoseObjectiveCfg,
)
from isaaclab_newton.controllers.ik.newton_ik_solver_cfg import NewtonIKSolverCfg
from isaaclab_newton.envs.mdp.actions.newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg, VBDSolverCfg
from isaaclab_newton.sim.schemas import NewtonCollisionCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, CableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab_tasks.core.lift.mdp.observations import cable_segment_positions_in_robot_root_frame
from isaaclab_tasks.utils import PresetCfg

from . import mdp
from .assets import (
    RIZON_SHARPA_ARM_JOINT_NAMES,
    RIZON_SHARPA_END_EFFECTOR_BODY_NAME,
    RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES,
    RIZON_SHARPA_RIGHT_HAND_LIMITS_RAD,
)
from .cable import CONNECTOR_SPAN_M, PATCH_CABLE_COLOR, PatchCableCfg, hanging_cable_positions
from .robot_cfg import ROBOT_BASE_POSITION_E, rizon_sharpa_articulation_cfg
from .workcell import SOCKET_POSITION_E, UPPER_SOCKET_POSITION_E


def _sticky_collision_pipeline(model):
    """Retain contact friction anchors while the gripped connector rotates."""
    import newton

    return newton.CollisionPipeline(model, contact_matching="sticky")


@configclass
class _ContactHistoryVBDSolverCfg(VBDSolverCfg):
    rigid_contact_history: bool = True
    rigid_joint_angular_ke: float = 10.0


@configclass
class GB300SharpaSceneCfg(InteractiveSceneCfg):
    """One pregrasped short cable, two physical sockets and the fixed-base robot."""

    robot = rizon_sharpa_articulation_cfg()
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1500))
    task_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TaskLight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, -0.2, 2.55)),
        spawn=sim_utils.SphereLightCfg(intensity=1200, radius=0.25, color=(0.85, 0.93, 1.0)),
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
            physics_material=sim_utils.CableMaterialCfg(
                thickness=0.0065,
                density=1100,
                stretch_stiffness=1.0e8,
                bend_stiffness=5.0e7,
                twist_stiffness=3.0e7,
            ),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=PATCH_CABLE_COLOR),
        ),
    )
    latch = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cable/latch",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.52, 0.24, 2.04), rot=(math.sqrt(0.5), 0, 0, math.sqrt(0.5))),
    )


@configclass
class ActionsCfg:
    """Six bounded palm increments and 22 reset-relative finger targets.

    Unit actions request 0.005 [m] translation, 0.025 [rad] rotation or a
    1.0 [rad] finger offset. Finger targets are clipped to physical joint limits.
    """

    right_palm = NewtonInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(RIZON_SHARPA_ARM_JOINT_NAMES),
        controller=NewtonIKSolverCfg(iterations=4),
        objectives=[
            NewtonIKPoseObjectiveCfg(
                name="right_palm",
                body_name=RIZON_SHARPA_END_EFFECTOR_BODY_NAME,
                use_relative_mode=True,
                scale=(0.005, 0.005, 0.005, 0.025, 0.025, 0.025),
            ),
            NewtonIKJointLimitObjectiveCfg(weight=0.1),
        ],
        clip={".*": (-1.0, 1.0)},
    )
    right_hand = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES),
        preserve_order=True,
        use_default_offset=True,
        scale=1.0,
        clip=dict(zip(RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES, RIZON_SHARPA_RIGHT_HAND_LIMITS_RAD, strict=True)),
    )


@configclass
class AbsoluteActionsCfg(ActionsCfg):
    """Canonical-palm pose [m, XYZW] followed by 22 absolute finger targets [rad]."""

    def __post_init__(self) -> None:
        objective = self.right_palm.objectives[0]
        objective.use_relative_mode = False
        objective.scale = 1.0
        self.right_palm.clip = None
        self.right_hand.use_default_offset = False
        self.right_hand.scale = 1.0


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=base_mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=base_mdp.joint_vel_rel)
        head = ObservationTermCfg(func=mdp.head_state)
        latch = ObservationTermCfg(func=mdp.latch_state)
        goal = ObservationTermCfg(func=mdp.goal_state)
        cable = ObservationTermCfg(func=cable_segment_positions_in_robot_root_frame)
        actions = ObservationTermCfg(func=base_mdp.last_action)
        enable_corruption = False

    policy = PolicyCfg()


@configclass
class EventsCfg:
    reset_scene = EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})


@configclass
class RewardsCfg:
    reach = RewardTermCfg(func=mdp.reaching, weight=1.0)
    head_position = RewardTermCfg(func=mdp.head_goal_position, weight=5.0, params={"std": 0.05})
    head_orientation = RewardTermCfg(func=mdp.head_goal_orientation, weight=1.0, params={"std": 0.25})
    seated = RewardTermCfg(func=mdp.insertion_success, weight=10.0)
    action_rate = RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.001)


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=base_mdp.time_out, time_out=True)
    cable_invalid = TerminationTermCfg(func=mdp.cable_invalid)


@configclass
class GB300SharpaInsertionEnvCfg(ManagerBasedRLEnvCfg):
    """Native VBD cable with MJWarp proxy coupling and normalized PPO controls."""

    scene = GB300SharpaSceneCfg(num_envs=1, env_spacing=8.0)
    actions = ActionsCfg()
    observations = ObservationsCfg()
    events = EventsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    commands = None
    curriculum = None
    decimation = 2
    episode_length_s = 60.0
    hydroelastic_contacts: bool = False
    """Use hydroelastic hand/plug/clip contacts."""
    hydroelastic_pad_pressure_gradient: float = 5.0e8
    """Compliant finger-cover pressure gradient [N/m^3]."""
    rigid_connector_contact_stiffness: float = 1.0e5
    """Connector-housing ordinary-contact stiffness [N/m]."""
    tcp_body_name: str = RIZON_SHARPA_END_EFFECTOR_BODY_NAME
    target_socket_position_e: tuple[float, float, float] = SOCKET_POSITION_E
    """Target socket origin in environment coordinates [m]."""
    grasp_body_patterns: tuple[str, ...] = (
        "right_hand_C_MC",
        "right_thumb_",
        "right_index_",
        "right_middle_",
        "right_ring_",
        "right_pinky_",
    )

    def __post_init__(self) -> None:
        if self.hydroelastic_contacts:
            self.scene.cable.spawn.rigid_connector = True
            self.scene.latch = None
            self.scene.connector = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Cable",
                spawn=None,
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.52, 0.26 - CONNECTOR_SPAN_M / 2, 2.04),
                    rot=self.scene.cable.init_state.rot,
                    joint_pos={"spring_hinge": 0.0},
                    joint_vel={"spring_hinge": 0.0},
                ),
                actuators={
                    "clip": ImplicitActuatorCfg(joint_names_expr=["spring_hinge"], stiffness=None, damping=None)
                },
            )
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = NewtonCfg(
            num_substeps=8,
            # Proxy VBD contact history cannot yet initialize during graph capture.
            use_cuda_graph=self.hydroelastic_contacts,
            default_shape_cfg=NewtonShapeCfg(mu=1.0, ke=1.0e4, kd=0.1),
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(
                        name="rigid",
                        bodies=[r"/World/envs/env_.*/Robot", r"/World/envs/env_.*/Fixtures"],
                        solver_cfg=MJWarpSolverCfg(
                            cone="elliptic",
                            ls_iterations=50,
                            integrator="implicitfast",
                            use_mujoco_contacts=False,
                            njmax=2048,
                        ),
                    ),
                    CouplerEntryCfg(
                        name="cable",
                        bodies=[r"/World/envs/env_.*/Cable"],
                        include_static_shapes=True,
                        solver_cfg=_ContactHistoryVBDSolverCfg(
                            iterations=12,
                            rigid_compliant_alm=True,
                            rigid_body_contact_buffer_size=1024,
                        ),
                    ),
                ],
                proxies=[
                    CouplerProxyMappingCfg(
                        source="rigid",
                        destination="cable",
                        bodies=[r"/World/envs/env_.*/Robot/.*", r"/World/envs/env_.*/Fixtures"],
                        collide_interval=1,
                        collision_pipeline=_sticky_collision_pipeline,
                    )
                ],
            ),
        )
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(1.65, -1.5, 2.35), lookat=(0.6, 0.35, 1.9))
        if self.hydroelastic_contacts:
            from .hydroelastic import configure_hydroelastic_physics

            configure_hydroelastic_physics(self.sim.physics)


@configclass
class GB300SharpaScriptedEnvCfg(GB300SharpaInsertionEnvCfg):
    """Absolute controls and hydroelastic contacts for the continuous physical example."""

    actions = AbsoluteActionsCfg()
    decimation = 4
    hydroelastic_contacts: bool = True
    grasp_body_patterns: tuple[str, ...] = (
        "right_hand_C_MC",
        "right_thumb_",
        "right_index_",
        "right_middle_",
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # Millimeter-scale grasps do not need the default 10 mm search halo.
        self.scene.robot.spawn.collision_props = {
            "/.*": [NewtonCollisionCfg(contact_gap=0.001)],
            # Tucked digits are visual-only for this three-finger scripted pinch.
            "/right_(?:ring|pinky)_.*/.*": [sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False)],
        }
        rigid, cable = self.sim.physics.solver_cfg.entries
        # Refresh tight socket contacts without increasing rigid integration frequency.
        self.sim.physics.num_substeps = 4
        self.sim.physics.collision_cfg.contact_matching = "latest"
        rigid.substeps = 2
        rigid.solver_cfg.update_data_interval = 4
        cable.substeps = 2
        cable.solver_cfg.iterations = 8
        self.scene.cable.spawn.physics_material.bend_stiffness *= 2.0
        self.scene.cable.spawn.physics_material.twist_stiffness *= 2.0
        self.scene.cable.spawn.latch_damping_scale = 3.0
        for name in ("right_thumb", "right_fingers"):
            actuator = self.scene.robot.actuators[name]
            actuator.stiffness = 96.0
            actuator.damping = 2.4
            # Honor the authored per-joint limits; regularize tiny finger inertias.
            actuator.joint_effort_limit = None
            actuator.armature = 2.0e-4
        rigid.solver_cfg.nconmax = 2048
        rigid.solver_cfg.njmax = 2048


@configclass
class GB300SharpaInsertionPresetsCfg(PresetCfg):
    """Normalized training by default; absolute physical example with ``scripted``."""

    default = GB300SharpaInsertionEnvCfg()
    scripted = GB300SharpaScriptedEnvCfg()
