# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab import sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp


@configclass
class KukaAllegroRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class KukaAllegroReorientRewardCfg(dexsuite.RewardsCfg):

    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb.
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.5,
        params={"threshold": 1.0},
    )


@configclass
class KukaAllegroMixinCfg:
    rewards: KukaAllegroReorientRewardCfg = KukaAllegroReorientRewardCfg()
    actions: KukaAllegroRelJointPosActionCfg = KukaAllegroRelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "palm_link"
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(os.path.dirname(__file__), "kuka_allegro.usd"),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    retain_accelerations=True,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1000.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=1,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.0005,
                ),
                joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    "iiwa7_joint_(1|2|7)": 0.0,
                    "iiwa7_joint_3": 0.7854,
                    "iiwa7_joint_4": 1.5708,
                    "iiwa7_joint_(5|6)": -1.5708,
                    "(index|middle|ring)_joint_0": 0.0,
                    "(index|middle|ring)_joint_1": 0.3,
                    "(index|middle|ring)_joint_2": 0.3,
                    "(index|middle|ring)_joint_3": 0.3,
                    "thumb_joint_0": 1.5,
                    "thumb_joint_1": 0.60147215,
                    "thumb_joint_2": 0.33795027,
                    "thumb_joint_3": 0.60845138,
                },
            ),
            actuators={
                "kuka_allegro_actuators": ImplicitActuatorCfg(
                    joint_names_expr=[
                        "iiwa7_joint_(1|2|3|4|5|6|7)",
                        "index_joint_(0|1|2|3)",
                        "middle_joint_(0|1|2|3)",
                        "ring_joint_(0|1|2|3)",
                        "thumb_joint_(0|1|2|3)",
                    ],
                    effort_limit_sim={
                        "iiwa7_joint_(1|2|3|4|5|6|7)": 300.0,
                        "index_joint_(0|1|2|3)": 0.5,
                        "middle_joint_(0|1|2|3)": 0.5,
                        "ring_joint_(0|1|2|3)": 0.5,
                        "thumb_joint_(0|1|2|3)": 0.5,
                    },
                    stiffness={
                        "iiwa7_joint_(1|2|3|4)": 300.0,
                        "iiwa7_joint_5": 100.0,
                        "iiwa7_joint_6": 50.0,
                        "iiwa7_joint_7": 25.0,
                        "index_joint_(0|1|2|3)": 3.0,
                        "middle_joint_(0|1|2|3)": 3.0,
                        "ring_joint_(0|1|2|3)": 3.0,
                        "thumb_joint_(0|1|2|3)": 3.0,
                    },
                    damping={
                        "iiwa7_joint_(1|2|3|4)": 45.0,
                        "iiwa7_joint_5": 20.0,
                        "iiwa7_joint_6": 15.0,
                        "iiwa7_joint_7": 15.0,
                        "index_joint_(0|1|2|3)": 0.1,
                        "middle_joint_(0|1|2|3)": 0.1,
                        "ring_joint_(0|1|2|3)": 0.1,
                        "thumb_joint_(0|1|2|3)": 0.1,
                    },
                    friction={
                        "iiwa7_joint_(1|2|3|4|5|6|7)": 1.0,
                        "index_joint_(0|1|2|3)": 0.01,
                        "middle_joint_(0|1|2|3)": 0.01,
                        "ring_joint_(0|1|2|3)": 0.01,
                        "thumb_joint_(0|1|2|3)": 0.01,
                    },
                ),
            },
            soft_joint_pos_limit_factor=1.0,
        )
        finger_tip_body_list = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["palm_link", ".*_tip"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"])


@configclass
class DexsuiteKukaAllegroReorientEnvCfg(KukaAllegroMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroReorientEnvCfg_PLAY(KukaAllegroMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    pass


@configclass
class DexsuiteKukaAllegroLiftEnvCfg(KukaAllegroMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroLiftEnvCfg_PLAY(KukaAllegroMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass
