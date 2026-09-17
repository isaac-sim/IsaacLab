# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from typing import Any

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import REQUIRED
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


##
# Scene definition
##
@dataclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = REQUIRED
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = REQUIRED

    # Table
    table: Any = field(
        default_factory=lambda: AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0, 0, 0.707, 0.707]),
            spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
        )
    )

    # plane
    plane: Any = field(
        default_factory=lambda: AssetBaseCfg(
            prim_path="/World/GroundPlane",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
            spawn=GroundPlaneCfg(),
        )
    )

    # lights
    light: Any = field(
        default_factory=lambda: AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
    )


##
# MDP settings
##
@dataclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = REQUIRED
    gripper_action: mdp.BinaryJointPositionActionCfg = REQUIRED


@dataclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @dataclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions: Any = field(default_factory=lambda: ObsTerm(func=mdp.last_action))
        joint_pos: Any = field(default_factory=lambda: ObsTerm(func=mdp.joint_pos_rel))
        joint_vel: Any = field(default_factory=lambda: ObsTerm(func=mdp.joint_vel_rel))
        object: Any = field(default_factory=lambda: ObsTerm(func=mdp.instance_randomize_object_obs))
        cube_positions: Any = field(
            default_factory=lambda: ObsTerm(func=mdp.instance_randomize_cube_positions_in_world_frame)
        )
        cube_orientations: Any = field(
            default_factory=lambda: ObsTerm(func=mdp.instance_randomize_cube_orientations_in_world_frame)
        )
        eef_pos: Any = field(default_factory=lambda: ObsTerm(func=mdp.ee_frame_pos))
        eef_quat: Any = field(default_factory=lambda: ObsTerm(func=mdp.ee_frame_quat))
        gripper_pos: Any = field(default_factory=lambda: ObsTerm(func=mdp.gripper_pos))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = field(default_factory=PolicyCfg)


@dataclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: Any = field(default_factory=lambda: DoneTerm(func=mdp.time_out, time_out=True))


@dataclass
class StackInstanceRandomizeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = field(
        default_factory=lambda: ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    )
    # Basic settings
    observations: ObservationsCfg = field(default_factory=ObservationsCfg)
    actions: ActionsCfg = field(default_factory=ActionsCfg)
    # MDP settings
    terminations: TerminationsCfg = field(default_factory=TerminationsCfg)

    # Unused managers
    commands: Any = None
    rewards: Any = None
    events: Any = None
    curriculum: Any = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
        )
