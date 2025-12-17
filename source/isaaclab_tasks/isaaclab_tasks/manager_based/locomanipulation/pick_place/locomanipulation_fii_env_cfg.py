# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import Articulation
from isaaclab.controllers.pink_ik.local_frame_task import LocalFrameTask
from isaaclab.controllers.pink_ik.pink_ik_cfg import PinkIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.fii.fii_retargeter import FiiRetargeterCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp

from .swerve_ik import swerve_isosceles_ik

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FII_USD_PATH = os.path.join(CURRENT_DIR, "Fiibot_W_1_V2_251016_Modified.usd")
FII_URDF_PATH = os.path.join(CURRENT_DIR, "Fiibot_W_1_V2_251016_Modified_urdf")  # will be created if it doesn't exit
OBJECT_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd"
FORCE_URDF_BUILD = True


class FiibotSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.85, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.75, 1.035), rot=(1, 0, 0, 0)),
        spawn=UsdFileCfg(
            usd_path=OBJECT_USD_PATH,
            scale=(1.5, 1.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.7071068, 0.0, 0.0, 0.7071068),
            joint_pos={
                "jack_joint": 0.7,
                "left_1_joint": 0.0,
                "left_2_joint": 0.785398,
                "left_3_joint": 0.0,
                "left_4_joint": 1.570796,
                "left_5_joint": 0.0,
                "left_6_joint": -0.785398,
                "left_7_joint": 0.0,
                "right_1_joint": 0.0,
                "right_2_joint": 0.785398,
                "right_3_joint": 0.0,
                "right_4_joint": 1.570796,
                "right_5_joint": 0.0,
                "right_6_joint": -0.785398,
                "right_7_joint": 0.0,
            },
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=FII_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        actuators={
            "actuators": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None),
            "jack_joint": ImplicitActuatorCfg(joint_names_expr=["jack_joint"], damping=5000.0, stiffness=500000.0),
        },
    )


class FiibotLowerBodyAction(ActionTerm):
    """Action term that is based on Agile lower body RL policy."""

    cfg: "FiibotLowerBodyActionCfg"
    """The configuration of the action term."""

    _asset: Articulation
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: "FiibotLowerBodyActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._env = env

        self._joint_names = [
            "walk_mid_top_joint",
            "walk_left_bottom_joint",
            "walk_right_bottom_joint",
            "jack_joint",
            "front_wheel_joint",
            "left_wheel_joint",
            "right_wheel_joint",
        ]

        self._joint_ids = [self._asset.data.joint_names.index(joint_name) for joint_name in self._joint_names]

        self._joint_pos_target = torch.zeros(self.num_envs, 7, device=self.device)
        self._joint_vel_target = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def action_dim(self) -> int:
        """Lower Body Action: [vx, vy, wz, jack_joint_height]"""
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._joint_pos_target

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._joint_pos_target

    def process_actions(self, actions: torch.Tensor):

        ik_out = swerve_isosceles_ik(
            vx=float(actions[0, 0]),
            vy=float(actions[0, 1]),
            wz=float(actions[0, 2]),
            L1=0.30438,
            d=0.17362,
            w=0.25,
            R=0.06,
        )

        self._joint_pos_target[:, 0] = ik_out["wheel1"]["angle_rad"]
        self._joint_pos_target[:, 1] = ik_out["wheel2"]["angle_rad"]
        self._joint_pos_target[:, 2] = ik_out["wheel3"]["angle_rad"]
        self._joint_pos_target[:, 3] = float(actions[0, 3])

        self._joint_vel_target[:, 0] = ik_out["wheel1"]["omega"]
        self._joint_vel_target[:, 1] = ik_out["wheel2"]["omega"]
        self._joint_vel_target[:, 2] = ik_out["wheel3"]["omega"]

    def apply_actions(self):

        self._joint_pos_target[:, 4:] = self._joint_pos_target[:, 4:] + self._env.physics_dt * self._joint_vel_target

        self._asset.set_joint_position_target(target=self._joint_pos_target, joint_ids=self._joint_ids)


@configclass
class FiibotLowerBodyActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = FiibotLowerBodyAction


@configclass
class FiibotActionsCfg:
    """Action specifications for the MDP."""

    upper_body_ik = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            # "waist_joint",
            "left_1_joint",
            "left_2_joint",
            "left_3_joint",
            "left_4_joint",
            "left_5_joint",
            "left_6_joint",
            "left_7_joint",
            "right_1_joint",
            "right_2_joint",
            "right_3_joint",
            "right_4_joint",
            "right_5_joint",
            "right_6_joint",
            "right_7_joint",
        ],
        hand_joint_names=[
            "left_hand_grip1_joint",
            "left_hand_grip2_joint",
            "right_hand_grip1_joint",
            "right_hand_grip2_joint",
        ],
        target_eef_link_names={
            "left_wrist": "Fiibot_W_2_V2_left_7_Link",
            "right_wrist": "Fiibot_W_2_V2_right_7_Link",
        },
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="base_link",
            num_hand_joints=4,
            show_ik_warnings=True,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                LocalFrameTask(
                    "Fiibot_W_2_V2_left_7_Link",
                    base_link_frame_name="Root",
                    position_cost=1.0,
                    orientation_cost=1.0,
                    lm_damping=10,
                    gain=0.1,
                ),
                LocalFrameTask(
                    "Fiibot_W_2_V2_right_7_Link",
                    base_link_frame_name="Root",
                    position_cost=1.0,
                    orientation_cost=1.0,
                    lm_damping=10,
                    gain=0.1,
                ),
            ],
            fixed_input_tasks=[],
        ),
    )

    lower_body_ik = FiibotLowerBodyActionCfg(asset_name="robot")


@configclass
class FiibotObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=manip_mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        robot_links_state = ObsTerm(func=manip_mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "left_7_Link"})
        left_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "left_7_Link"})
        right_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "right_7_Link"})
        right_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "right_7_Link"})

        hand_joint_state = ObsTerm(
            func=manip_mdp.get_robot_joint_state,
            params={
                "joint_names": [
                    "left_hand_grip1_joint",
                    "left_hand_grip2_joint",
                    "right_hand_grip1_joint",
                    "right_hand_grip2_joint",
                ]
            },
        )

        object = ObsTerm(
            func=manip_mdp.object_obs,
            params={"left_eef_link_name": "left_7_Link", "right_eef_link_name": "right_7_Link"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class FiibotTerminationsCfg:

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(
        func=manip_mdp.task_done_pick_place,
        params={
            "task_link_name": "right_7_Link",
            "right_wrist_max_x": 0.26,
            "min_x": 0.40,
            "max_x": 0.85,
            "min_y": 0.35,
            "max_y": 0.8,
            "max_height": 1.10,
            "min_vel": 0.20,
        },
    )


@configclass
class FiibotRewardsCfg:
    pass


@configclass
class FiibotEnvCfg(ManagerBasedRLEnvCfg):

    scene: FiibotSceneCfg = FiibotSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    actions: FiibotActionsCfg = FiibotActionsCfg()
    observations = FiibotObservationsCfg()
    rewards = FiibotRewardsCfg()
    terminations = FiibotTerminationsCfg()

    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.25),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, 3.0, 1.5), lookat=(0.0, 0.0, 0.7), origin_type="asset_body", asset_name="robot", body_name="base_link"
    )

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 200.0
        self.sim.dt = 1 / 120  # 200Hz
        self.sim.render_interval = 4

        urdf_path = FII_URDF_PATH

        # Get the USD path from the robot spawn configuration
        robot_spawn = self.scene.robot.spawn
        # Type checker doesn't recognize UsdFileCfg.usd_path, but it exists
        usd_file_path = getattr(robot_spawn, "usd_path", FII_USD_PATH)

        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            usd_file_path, urdf_path, force_conversion=FORCE_URDF_BUILD
        )

        self.actions.upper_body_ik.controller.urdf_path = temp_urdf_output_path
        self.actions.upper_body_ik.controller.mesh_path = temp_urdf_meshes_output_path

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        FiiRetargeterCfg(
                            sim_device=self.sim.device,
                        )
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )
