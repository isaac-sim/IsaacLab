# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import REQUIRED, copy_config, replace_config
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks.core.cabinet.mdp as mdp
from isaaclab_tasks.utils import PresetCfg

FRAME_MARKER_SMALL_CFG = copy_config(FRAME_MARKER_CFG)
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

CABINET_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Cabinet",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.8, 0, 0.4),
        rot=(0.0, 0.0, 1.0, 0.0),
        joint_pos={
            "door_left_joint": 0.0,
            "door_right_joint": 0.0,
            "drawer_bottom_joint": 0.0,
            "drawer_top_joint": 0.0,
        },
    ),
    actuators={
        "drawers": ImplicitActuatorCfg(
            joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
            joint_effort_limit=87.0,
            stiffness=10.0,
            damping=1.0,
        ),
        "doors": ImplicitActuatorCfg(
            joint_names_expr=["door_left_joint", "door_right_joint"],
            joint_effort_limit=87.0,
            stiffness=10.0,
            damping=2.5,
        ),
    },
)
"""Shared cabinet articulation configuration."""

PLANE_CFG = AssetBaseCfg(
    prim_path="/World/GroundPlane",
    init_state=AssetBaseCfg.InitialStateCfg(),
    spawn=sim_utils.GroundPlaneCfg(),
    collision_group=-1,
)
"""Shared ground-plane configuration."""

LIGHT_CFG = AssetBaseCfg(
    prim_path="/World/light",
    spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
)
"""Shared dome-light configuration."""


@dataclass
class CabinetSimCfg(PresetCfg):
    """Simulation configuration presets for the cabinet environment.

    Wraps the full :class:`~isaaclab.sim.SimulationCfg` so that Newton can run at a
    finer physics timestep (1/600 s) while PhysX keeps its default (1/60 s).
    """

    isaacsim_physx: SimulationCfg = field(
        default_factory=lambda: SimulationCfg(
            dt=1 / 60,
            render_interval=1,
            physics=PhysxCfg(bounce_threshold_velocity=0.01, friction_correlation_distance=0.00625),
            default_visualizer_cfg=VisualizerCfg(eye=(-2.0, 2.0, 2.0), lookat=(0.8, 0.0, 0.5)),
        )
    )
    ovphysx: SimulationCfg = field(
        default_factory=lambda: replace_config(
            SimulationCfg(
                dt=1 / 60,
                render_interval=1,
                physics=PhysxCfg(bounce_threshold_velocity=0.01, friction_correlation_distance=0.00625),
                default_visualizer_cfg=VisualizerCfg(eye=(-2.0, 2.0, 2.0), lookat=(0.8, 0.0, 0.5)),
            ),
            physics=OvPhysxCfg(),
        )
    )
    physx: SimulationCfg = field(
        default_factory=lambda: replace_config(
            SimulationCfg(
                dt=1 / 60,
                render_interval=1,
                physics=PhysxCfg(bounce_threshold_velocity=0.01, friction_correlation_distance=0.00625),
                default_visualizer_cfg=VisualizerCfg(eye=(-2.0, 2.0, 2.0), lookat=(0.8, 0.0, 0.5)),
            ),
            physics=PhysxAutoCfg(
                isaacsim_physx=SimulationCfg(
                    dt=1 / 60,
                    render_interval=1,
                    physics=PhysxCfg(bounce_threshold_velocity=0.01, friction_correlation_distance=0.00625),
                    default_visualizer_cfg=VisualizerCfg(eye=(-2.0, 2.0, 2.0), lookat=(0.8, 0.0, 0.5)),
                ).physics,
                ovphysx=replace_config(
                    SimulationCfg(
                        dt=1 / 60,
                        render_interval=1,
                        physics=PhysxCfg(bounce_threshold_velocity=0.01, friction_correlation_distance=0.00625),
                        default_visualizer_cfg=VisualizerCfg(eye=(-2.0, 2.0, 2.0), lookat=(0.8, 0.0, 0.5)),
                    ),
                    physics=OvPhysxCfg(),
                ).physics,
            ),
        )
    )
    newton_mjwarp: SimulationCfg = field(
        default_factory=lambda: SimulationCfg(
            dt=1 / 600,
            render_interval=1,
            default_visualizer_cfg=VisualizerCfg(eye=(-2.0, 2.0, 2.0), lookat=(0.8, 0.0, 0.5)),
            physics=NewtonCfg(
                solver_cfg=MJWarpSolverCfg(
                    njmax=90,
                    nconmax=100,
                    cone="pyramidal",
                    integrator="implicitfast",
                    impratio=1,
                ),
                num_substeps=1,
                debug_mode=False,
            ),
        )
    )
    default: SimulationCfg = field(
        default_factory=lambda: SimulationCfg(
            dt=1 / 600,
            render_interval=1,
            default_visualizer_cfg=VisualizerCfg(eye=(-2.0, 2.0, 2.0), lookat=(0.8, 0.0, 0.5)),
            physics=NewtonCfg(
                solver_cfg=MJWarpSolverCfg(
                    njmax=90,
                    nconmax=100,
                    cone="pyramidal",
                    integrator="implicitfast",
                    impratio=1,
                ),
                num_substeps=1,
                debug_mode=False,
            ),
        )
    )


@dataclass
class CabinetDecimationCfg(PresetCfg):
    """Physics steps per policy action.

    Chosen per backend so that the policy always acts at 60 Hz, since the backends step physics at
    different rates.
    """

    isaacsim_physx: int = 1
    ovphysx: int = 1
    physx: int = 1
    newton_mjwarp: int = 10
    default: int = 10


##
# Scene definition
##


@dataclass
class CabinetSceneCfg(InteractiveSceneCfg):
    """Configuration for the cabinet scene with a robot and a cabinet.

    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames
    """

    # robot and end-effector frames -- set by a robot-specific subclass
    robot: ArticulationCfg = REQUIRED
    ee_frame: FrameTransformerCfg = REQUIRED

    cabinet: Any = field(default_factory=lambda: deepcopy(CABINET_CFG))

    # drawer handle frame, aligned with the end-effector frame
    cabinet_frame: Any = field(
        default_factory=lambda: FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Cabinet/sektion",
            debug_vis=True,
            visualizer_cfg=replace_config(FRAME_MARKER_SMALL_CFG, prim_path="/Visuals/CabinetFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_top",
                    name="drawer_handle_top",
                    offset=OffsetCfg(
                        pos=(0.305, 0.0, 0.01),
                        rot=(0.5, -0.5, -0.5, 0.5),  # align with end-effector frame
                    ),
                ),
            ],
        )
    )

    # plane
    plane: Any = field(default_factory=lambda: deepcopy(PLANE_CFG))

    # lights
    light: Any = field(default_factory=lambda: deepcopy(LIGHT_CFG))


##
# MDP settings
##


@dataclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg = REQUIRED
    gripper_action: mdp.BinaryJointPositionActionCfg = REQUIRED


@dataclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @dataclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos: Any = field(default_factory=lambda: ObsTerm(func=mdp.joint_pos_rel))
        joint_vel: Any = field(default_factory=lambda: ObsTerm(func=mdp.joint_vel_rel))
        cabinet_joint_pos: Any = field(
            default_factory=lambda: ObsTerm(
                func=mdp.joint_pos_rel,
                params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
            )
        )
        cabinet_joint_vel: Any = field(
            default_factory=lambda: ObsTerm(
                func=mdp.joint_vel_rel,
                params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
            )
        )
        rel_ee_drawer_distance: Any = field(default_factory=lambda: ObsTerm(func=mdp.rel_ee_drawer_distance))

        # the raw action is unbounded; feeding it back unclipped lets the critic and the policy
        # inflate each other without limit
        actions: Any = field(default_factory=lambda: ObsTerm(func=mdp.last_action, clip=(-5.0, 5.0)))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = field(default_factory=PolicyCfg)


@dataclass
class EventCfg:
    """Configuration for events."""

    robot_physics_material: Any = field(
        default_factory=lambda: EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.8, 1.25),
                "dynamic_friction_range": (0.8, 1.25),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 16,
            },
        )
    )

    cabinet_physics_material: Any = field(
        default_factory=lambda: EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("cabinet", body_names="drawer_handle_top"),
                "static_friction_range": (1.0, 1.25),
                "dynamic_friction_range": (1.25, 1.5),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 16,
            },
        )
    )

    reset_all: Any = field(default_factory=lambda: EventTerm(func=mdp.reset_scene_to_default, mode="reset"))

    reset_robot_joints: Any = field(
        default_factory=lambda: EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.1, 0.1),
                "velocity_range": (0.0, 0.0),
            },
        )
    )


@dataclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 1. Approach the handle
    approach_ee_handle: Any = field(
        default_factory=lambda: RewTerm(func=mdp.approach_ee_handle, weight=2.0, params={"threshold": 0.2})
    )
    align_ee_handle: Any = field(default_factory=lambda: RewTerm(func=mdp.align_ee_handle, weight=0.5))

    # 2. Grasp the handle
    approach_gripper_handle: Any = field(
        default_factory=lambda: RewTerm(func=mdp.approach_gripper_handle, weight=5.0, params={"offset": REQUIRED})
    )
    align_grasp_around_handle: Any = field(
        default_factory=lambda: RewTerm(func=mdp.align_grasp_around_handle, weight=0.125)
    )
    grasp_handle: Any = field(
        default_factory=lambda: RewTerm(
            func=mdp.grasp_handle,
            weight=0.5,
            params={
                "threshold": 0.03,
                "open_joint_pos": REQUIRED,
                "asset_cfg": SceneEntityCfg("robot", joint_names=REQUIRED),
            },
        )
    )

    # 3. Open the drawer
    open_drawer_bonus: Any = field(
        default_factory=lambda: RewTerm(
            func=mdp.open_drawer_bonus,
            weight=7.5,
            params={
                "asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
                "success_threshold": 0.30,
            },
        )
    )
    multi_stage_open_drawer: Any = field(
        default_factory=lambda: RewTerm(
            func=mdp.multi_stage_open_drawer,
            weight=1.0,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
        )
    )

    # 4. Penalize actions for cosmetic reasons
    action_rate_l2: Any = field(default_factory=lambda: RewTerm(func=mdp.action_rate_l2, weight=-1e-2))
    joint_vel: Any = field(default_factory=lambda: RewTerm(func=mdp.joint_vel_l2, weight=-0.0001))


@dataclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: Any = field(default_factory=lambda: DoneTerm(func=mdp.time_out, time_out=True))


##
# Environment configuration
##


@dataclass
class CabinetEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cabinet environment."""

    sim: CabinetSimCfg = field(default_factory=CabinetSimCfg)
    # Scene settings
    scene: CabinetSceneCfg = field(default_factory=lambda: CabinetSceneCfg(num_envs=4096, env_spacing=2.0))
    # Basic settings
    observations: ObservationsCfg = field(default_factory=ObservationsCfg)
    actions: ActionsCfg = field(default_factory=ActionsCfg)
    # MDP settings
    rewards: RewardsCfg = field(default_factory=RewardsCfg)
    terminations: TerminationsCfg = field(default_factory=TerminationsCfg)
    events: EventCfg = field(default_factory=EventCfg)

    decimation: int = field(default_factory=CabinetDecimationCfg)

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.episode_length_s = 8.0
        # simulation settings are defined in CabinetSimCfg (dt/physics vary per backend)
