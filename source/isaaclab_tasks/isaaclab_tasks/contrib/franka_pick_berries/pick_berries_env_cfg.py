# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A Franka and a Gaussian berry on a studio surface or the EBC demo table."""

from isaaclab_newton.physics import NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.envs import mdp
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    TerminationTermCfg,
)
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from ..stack.config.franka.stack_joint_pos_env_cfg import FrankaCubeStackEnvCfg
from .assets.asset_root import berry_root
from .control.gamepad import BerryGamepadCfg
from .mdp.actions import BerryGraspActionCfg, BerryIKAction
from .physics.solver import BerrySolverCfg
from .scene.tableware import PLATE, TABLE_POSITION, TABLE_ROTATION, spawn_tableware


@configclass
class BerryObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        actions = ObservationTermCfg(func=mdp.last_action)
        concatenate_terms = False
        enable_corruption = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class BerryEventsCfg:
    reset_robot = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )


@configclass
class BerryTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class BerryPickEnvCfg(FrankaCubeStackEnvCfg):
    berry: str = "raspberry"
    target_berry: str = "raspberry"
    pair: bool = False
    berry_count: int = 1
    sorting_demo: bool = False
    randomize_layout: bool = True
    layout_seed: int = 0
    pair_layout: str = "plate"
    berry_friction: float = 0.4
    berry_asset_path: str | None = None
    berry_asset_version: str = "v1"
    physics_profile: str = "handling"
    physics_resolution: str = "full"
    mpm_hz: int | None = None
    asset_root: str = berry_root()
    berry_position: tuple[float, float, float] = (0.48, 0.0, 0.0)
    background: str = "studio"

    def __post_init__(self):
        super().__post_init__()
        lab_table = self.scene.table.copy()
        self.scene.num_envs = 1
        # Inherited as False from the stack task's per-env cube randomization; a no-op with a
        # single environment, but the Newton backend's articulation/body pattern matching fails
        # to register the robot at all when it is False here.
        self.scene.replicate_physics = True
        self.scene.cube_1 = self.scene.cube_2 = self.scene.cube_3 = None
        self.scene.ee_frame = None
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Gravity-compensated arm, as in the standard Franka differential-IK task.
        self.scene.robot.spawn.rigid_props.disable_gravity = True
        ready = (
            0.00785503,
            0.30280935,
            -0.00700201,
            -2.45886405,
            0.00562982,
            2.76166094,
            0.78134144,
        )
        self.scene.robot.init_state.joint_pos = {f"panda_joint{i}": q for i, q in enumerate(ready, 1)}
        self.scene.robot.init_state.joint_pos["panda_finger_joint.*"] = 0.04
        self.scene.table.init_state.pos = (0.45, 0, -0.025)
        self.scene.table.init_state.rot = (0, 0, 0, 1)
        self.scene.table.spawn = sim_utils.CuboidCfg(
            size=(0.8, 0.7, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.26, 0.29), roughness=0.65),
        )
        if self.background == "ebc":
            self.scene.table = lab_table
            self.scene.table.init_state.pos = TABLE_POSITION
            self.scene.table.init_state.rot = TABLE_ROTATION
            self.scene.plane.spawn.visible = False
            self.berry_position = (PLATE[0], PLATE[1], PLATE[4])
            self.scene.tableware = AssetBaseCfg(
                prim_path="/World/TablewareCollisions",
                spawn=sim_utils.SpawnerCfg(func=spawn_tableware),
            )
        elif self.background != "studio":
            raise ValueError(f"Unknown berry background: {self.background}")
        self.scene.robot.actuators["panda_hand"].stiffness = 400.0
        self.scene.robot.actuators["panda_hand"].damping = 20.0
        self.scene.robot.actuators["panda_hand"].joint_effort_limit = 10.0
        self.actions.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            class_type=BerryIKAction,
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=1.0,
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0, 0, 0.107)),
        )
        self.actions.gripper_action = BerryGraspActionCfg(asset_name="robot")
        self.observations = BerryObservationsCfg()
        self.events = BerryEventsCfg()
        self.terminations = BerryTerminationsCfg()
        self.decimation = 4
        self.sim.dt = 1 / 120
        self.sim.render_interval = 4
        self.episode_length_s = 3600
        self.sim.physics = NewtonCfg(
            num_substeps=1,
            use_cuda_graph=False,
            # The task draws with its own viewer, which the default (None) does not detect.
            load_visual_shapes=True,
            solver_cfg=BerrySolverCfg(
                class_type="isaaclab_tasks.contrib.franka_pick_berries.physics.solver:NewtonBerryManager",
                use_mujoco_contacts=True,
                use_mujoco_cpu=True,
                integrator="implicitfast",
                cone="elliptic",
                iterations=40,
                njmax=1024,
                nconmax=256,
                update_data_interval=1,
            ),
        )
        self.sim.physics.default_shape_cfg.gap = 0.0
        self.teleop_devices = DevicesCfg(devices={"gamepad": BerryGamepadCfg(sim_device="cpu")})
