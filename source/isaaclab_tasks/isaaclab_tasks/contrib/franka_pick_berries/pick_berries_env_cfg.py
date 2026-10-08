# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A Franka at a lab table with a punnet of raspberries, a receiving bowl and a reject dish, in a scanned room."""

from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from .assets.asset_paths import raspberry_asset_path
from .mdp.actions import GripperApertureActionCfg, WorkspaceIKAction
from .physics.coupling import coupled_physics_cfg
from .scene.tableware import PUNNET, TABLE_POSITION, TABLE_ROTATION, spawn_kinematic_usd, spawn_tableware

# Franka joint positions [rad] with the hand pointing down above the punnet.
_READY_JOINTS = (0.00785503, 0.30280935, -0.00700201, -2.45886405, 0.00562982, 2.76166094, 0.78134144)


@configclass
class BerrySceneCfg(InteractiveSceneCfg):
    """The Franka, the lab table and the tableware; the environment adds the berries' tissue."""

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={**{f"panda_joint{i}": q for i, q in enumerate(_READY_JOINTS, 1)}, "panda_finger_joint.*": 0.04}
        ),
    )
    # A kinematic body: the arm solver collides with it and the tissue solver sees it as a proxy.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POSITION, rot=TABLE_ROTATION),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            func=spawn_kinematic_usd,
        ),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(visible=False),
    )
    # The punnet, bowl and reject dish as static colliders; the viewer renders their visual meshes.
    tableware = AssetBaseCfg(prim_path="/World/TablewareCollisions", spawn=sim_utils.SpawnerCfg(func=spawn_tableware))


@configclass
class BerryActionsCfg:
    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        class_type=WorkspaceIKAction,
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=1.0,
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0, 0, 0.107)),
    )
    gripper_action = GripperApertureActionCfg(asset_name="robot")


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
class BerryPickEnvCfg(ManagerBasedRLEnvCfg):
    scene: BerrySceneCfg = BerrySceneCfg(num_envs=1, env_spacing=2.5)
    actions: BerryActionsCfg = BerryActionsCfg()
    observations: BerryObservationsCfg = BerryObservationsCfg()
    events: BerryEventsCfg = BerryEventsCfg()
    terminations: BerryTerminationsCfg = BerryTerminationsCfg()
    rewards = None
    commands = None
    curriculum = None

    tissue_solver: str = "explicit"
    """Tissue solver: ``explicit`` (explicit MLS-MPM with frictional finger pads, :mod:`.physics.grasp_explicit_mpm`)
    or ``implicit`` (Newton's implicit MPM with clamped grasping, :mod:`.physics.grasp_implicit_mpm`)."""
    num_berries: int = 1
    """Number of raspberries in the punnet, 1 to 3."""
    randomize_layout: bool = True
    """Scatter several berries in the punnet with random orientations, instead of the fixed layout."""
    layout_seed: int = 0
    """Seed of the random layout; reset replays the same layout."""
    tissue_resolution: str = "full"
    """Tissue particles: ``full``, or ``half`` of them for speed; the Gaussians are unchanged."""
    berry_asset: str = raspberry_asset_path()
    """Raspberry USDZ package: a Nucleus URL or a local path."""
    berry_position: tuple[float, float, float] = (PUNNET[0], PUNNET[1], PUNNET[6])
    """Center [m] of the berries' layout, on the punnet floor."""

    def __post_init__(self):
        super().__post_init__()
        # IK targets need backend-native gravity compensation to hold steady between commands, as in the standard
        # Franka differential-IK reach task.
        self.scene.robot.spawn.rigid_props = [
            PhysxRigidBodyCfg(disable_gravity=True, max_depenetration_velocity=5.0),
            MujocoRigidBodyCfg(gravcomp=1.0),
        ]
        self.scene.robot.actuators["panda_hand"].stiffness = 400.0
        self.scene.robot.actuators["panda_hand"].damping = 20.0
        self.scene.robot.actuators["panda_hand"].joint_effort_limit = 10.0
        # Four 120 Hz physics steps per 30 Hz control step.
        self.decimation = 4
        self.sim.dt = 1 / 120
        self.sim.render_interval = 4
        self.episode_length_s = 3600
        self.sim.physics = coupled_physics_cfg()
        self.sim.physics.default_shape_cfg.gap = 0.0
