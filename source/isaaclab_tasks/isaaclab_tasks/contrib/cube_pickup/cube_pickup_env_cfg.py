# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka pickup with a binary goal reward and generated reset states."""

import math
from typing import Literal

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.reset import ResetStateBufferCfg
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, replace
from isaaclab.visualizers import VisualizerCfg

from isaaclab_assets.robots import FRANKA_PANDA_HIGH_PD_CFG

from ...core.lift.mdp import ObjectUniformPoseCommandCfg
from ...utils import ResetStateSamplerCfg, preset
from . import mdp as pickup_mdp


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Existing Franka asset with a table and one 40 mm cube."""

    robot: ArticulationCfg = replace(FRANKA_PANDA_HIGH_PD_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, -0.02)),
        spawn=sim_utils.CuboidCfg(size=(0.8, 1.0, 0.04), collision_props=sim_utils.UsdPhysicsCollisionCfg()),
    )
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.021)),
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=0.05),
            physics_material=sim_utils.RigidBodyMaterialBaseCfg(static_friction=0.8, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.45, 0.8)),
        ),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))

    def __post_init__(self) -> None:
        super().__post_init__()
        self.robot.spawn.variants["Physics"] = "mujoco"
        # Match the high-PD asset's gravity-free arm in the native MuJoCo model.
        # The cube retains gravity throughout generation and training.
        self.robot.spawn.rigid_props = [
            PhysxRigidBodyCfg(disable_gravity=True, max_depenetration_velocity=5.0),
            MujocoRigidBodyCfg(gravcomp=1.0),
        ]
        # Keep the ordinary home start above the table with a downward-facing hand.
        self.robot.init_state.joint_pos["panda_joint6"] = 2.241
        self.robot.init_state.joint_pos["panda_joint7"] = math.pi / 4.0
        # The authored mimic constraint drives the second finger.
        self.robot.actuators["panda_hand"].stiffness = 1200.0
        self.robot.actuators["panda_hand"].damping = 60.0
        self.robot.actuators["panda_hand"].joint_effort_limit = 40.0


@configclass
class ActionsCfg:
    """Relative Cartesian pose increments and a binary gripper command."""

    arm = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint[1-7]"],
        body_name="panda_hand",
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=(0.025, 0.025, 0.025, 0.1, 0.1, 0.1),
    )
    gripper = mdp.AbsBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint1"],
        open_command_expr={"panda_finger_joint1": 0.04},
        close_command_expr={"panda_finger_joint1": 0.0},
        threshold=0.5,
    )


@configclass
class CommandsCfg:
    """Fixed pickup target expressed in the robot root frame [m]."""

    object_pose = ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(1.0e9, 1.0e9),
        position_only=True,
        debug_vis=False,
        ranges=ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(0.5, 0.5),
            pos_y=(0.0, 0.0),
            pos_z=(0.15, 0.15),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ObservationsCfg:
    """State observations; reset-row identities and teacher actions are never observed."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, scale=0.1)
        object_position = ObservationTermCfg(func=pickup_mdp.object_position)
        goal = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "object_pose"})
        hand_pose = ObservationTermCfg(
            func=pickup_mdp.hand_pose, params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_hand")}
        )
        object_to_hand = ObservationTermCfg(
            func=pickup_mdp.object_to_hand, params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_hand")}
        )
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy = PolicyCfg()


@configclass
class RewardsCfg:
    """One unit terminal success reward."""

    success = RewardTermCfg(func=pickup_mdp.success_reward, weight=1.0)


@configclass
class TerminationsCfg:
    """Success, table bounds, and a finite attempt duration."""

    success = TerminationTermCfg(
        func=pickup_mdp.goal_reached,
        params={
            "command_name": "object_pose",
            "threshold": 0.04,
            "hand_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
            "gripper_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*"),
        },
    )
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    dropped = TerminationTermCfg(func=pickup_mdp.object_dropped)


@configclass
class CubePickupEnvCfg(ManagerBasedRLEnvCfg):
    """Reference task for sparse learning with generated and validated reset states."""

    class_type: type | str = "{DIR}.cube_pickup_env:CubePickupEnv"

    scene: SceneCfg = SceneCfg(num_envs=1024, env_spacing=2.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: dict = {}
    commands: CommandsCfg = CommandsCfg()
    curriculum = None
    reset_buffer: ResetStateBufferCfg = ResetStateBufferCfg(size=1024, max_batches=512, seed=0)
    reset_sampler: ResetStateSamplerCfg = ResetStateSamplerCfg()
    reset_mode: Literal["adaptive", "uniform", "table"] = "adaptive"
    """Training distribution, or ordinary home/table starts for evaluation."""

    def __post_init__(self) -> None:
        self.decimation = 4
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        newton = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=128,
                nconmax=64,
                iterations=30,
                ls_iterations=15,
                cone="elliptic",
                integrator="implicitfast",
                use_mujoco_contacts=True,
            ),
            num_substeps=2,
        )
        self.sim.physics = preset(default=newton, newton_mjwarp=newton)
        self.episode_length_s = 12.0
        self.is_finite_horizon = True
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(1.3, 1.1, 0.9), lookat=(0.45, 0.0, 0.15))

    def play_mode(self) -> None:
        super().play_mode()
        self.reset_mode = "table"
