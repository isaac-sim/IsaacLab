# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp.actions as act
import isaaclab.envs.mdp.observations as obs
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip


@configclass
class TestingSceneCfg(InteractiveSceneCfg):
    # terrain
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    # robot
    robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class TestingActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = act.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class TestingObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # settings
        enable_corruption = False
        concatenate_terms = False
        # observations
        joint_pos = ObsTerm(func=obs.joint_pos)
        joint_vel = ObsTerm(func=obs.joint_vel)
        joint_effort = ObsTerm(func=obs.joint_effort)

    policy: PolicyCfg = PolicyCfg()


@configclass
class AnymalDBaseLineEnvCfg(ManagerBasedEnvCfg):
    """Environment config using Anymal to test out all publishers."""

    sim = SimulationCfg(device="cpu", dt=0.002, render_interval=16)
    scene = TestingSceneCfg(num_envs=1, env_spacing=2.5)
    actions = TestingActionsCfg()
    observations = TestingObservationsCfg()
    decimation = 4


"""
Additive sensors and observations
"""


@configclass
class AnymalDPlusLinkPoseObsCfg(AnymalDBaseLineEnvCfg):
    def __post_init__(self):
        self.observations.policy.base_link_pose = ObsTerm(func=obs.link_pose)


@configclass
class AnymalDPlusTwistObsCfg(AnymalDBaseLineEnvCfg):
    def __post_init__(self):
        self.observations.policy.base_lin_vel = ObsTerm(func=obs.base_lin_vel)
        self.observations.policy.base_ang_vel = ObsTerm(func=obs.base_ang_vel)


@configclass
class AnymalDPlusProjGravEnvCfg(AnymalDBaseLineEnvCfg):
    def __post_init__(self):
        self.observations.policy.projected_gravity = ObsTerm(func=obs.projected_gravity)


@configclass
class AnymalDPlusImuEnvCfg(AnymalDBaseLineEnvCfg):
    def __post_init__(self):
        # sensors
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=ImuCfg.OffsetCfg(
                pos=(-0.25565, 0.00255, 0.07672),
                rot=(0.0, 0.0, 1.0, 0.0),
            ),
        )
        self.observations.policy.imu_quat = ObsTerm(
            func=obs.imu_orientation,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        self.observations.policy.imu_ang_vel = ObsTerm(
            func=obs.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        self.observations.policy.imu_lin_acc = ObsTerm(
            func=obs.imu_lin_acc,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )


@configclass
class AnymalDPlusContactEnvCfg(AnymalDBaseLineEnvCfg):
    def __post_init__(self):
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
        )


@configclass
class AnymalDPlusHeightScanEnvCfg(AnymalDBaseLineEnvCfg):
    def __post_init__(self):
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.height_scan = ObsTerm(
            func=obs.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )


@configclass
class AnymalDPlusGridMapEnvCfg(AnymalDBaseLineEnvCfg):
    def __post_init__(self):
        self.scene.grid_map = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="world",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[4.0, 4.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )


@configclass
class AnymalDPlusWrenchEnvCfg(AnymalDBaseLineEnvCfg):
    def __post_init__(self):
        self.observations.policy.RF_foot_reaction = ObsTerm(
            func=obs.body_incoming_wrench, params={"asset_cfg": SceneEntityCfg("robot", body_names="RF_FOOT")}
        )


@configclass
class AnymalDPlusJointReactionWrenchEnvCfg(AnymalDBaseLineEnvCfg):
    def __post_init__(self):
        self.observations.policy.joint_reactions = ObsTerm(
            func=obs.body_incoming_wrench, params={"asset_cfg": SceneEntityCfg("robot")}
        )


@configclass
class AnymalDTestAllEnvCfg(AnymalDBaseLineEnvCfg):
    def __post_init__(self):
        # pose
        self.observations.policy.base_link_pose = ObsTerm(func=obs.link_pose)
        # twist
        self.observations.policy.base_lin_vel = ObsTerm(func=obs.base_lin_vel)
        self.observations.policy.base_ang_vel = ObsTerm(func=obs.base_ang_vel)
        # projected gravity
        self.observations.policy.projected_gravity = ObsTerm(func=obs.projected_gravity)
        # imu
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=ImuCfg.OffsetCfg(
                pos=(-0.25565, 0.00255, 0.07672),
                rot=(0.0, 0.0, 1.0, 0.0),
            ),
        )
        self.observations.policy.imu_quat = ObsTerm(
            func=obs.imu_orientation,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        self.observations.policy.imu_ang_vel = ObsTerm(
            func=obs.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        self.observations.policy.imu_lin_acc = ObsTerm(
            func=obs.imu_lin_acc,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        # contact forces
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
        )
        # height scanner
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.height_scan = ObsTerm(
            func=obs.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        # gridmap
        self.scene.grid_map = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="world",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[4.0, 4.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        # wrench
        self.observations.policy.RF_foot_reaction = ObsTerm(
            func=obs.body_incoming_wrench, params={"asset_cfg": SceneEntityCfg("robot", body_names="RF_FOOT")}
        )
        self.observations.policy.joint_reactions = ObsTerm(
            func=obs.body_incoming_wrench, params={"asset_cfg": SceneEntityCfg("robot")}
        )
