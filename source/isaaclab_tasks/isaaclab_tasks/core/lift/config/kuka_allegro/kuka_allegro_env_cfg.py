# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import config_field, replace_config

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

from ... import lift_env_cfg as lift
from ... import mdp
from .camera_cfg import StateObservationCfg

FINGERTIP_LIST = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
THUMB_SENSOR = "thumb_link_3_object_s"
FINGER_SENSORS = [f"{name}_object_s" for name in FINGERTIP_LIST if name != "thumb_link_3"]


@dataclass
class KukaAllegroSceneCfg(lift.SceneCfg):
    """KukaAllegro scene for the Lift and Reorient tasks.

    The ``base_camera`` / ``wrist_camera`` slots are left unset (``None``) for the state task; the
    camera env config populates them (see ``kuka_allegro_camera_env_cfg``).
    """

    robot: ArticulationCfg = config_field(replace_config(KUKA_ALLEGRO_CFG, prim_path="{ENV_REGEX_NS}/Robot"))
    base_camera: CameraCfg | None = config_field(None)
    wrist_camera: CameraCfg | None = config_field(None)

    def __post_init__(self):
        if parent_post_init := getattr(super(), "__post_init__", None):
            parent_post_init()
        for link_name in FINGERTIP_LIST:
            setattr(
                self,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )


@dataclass
class KukaAllegroRelJointPosActionCfg:
    action: Any = config_field(mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1))


@dataclass
class KukaAllegroReorientRewardCfg(lift.RewardsCfg):
    good_finger_contact: Any = config_field(
        RewTerm(
            func=mdp.contacts,
            weight=1.0,
            params={"threshold": 0.01, "thumb_name": THUMB_SENSOR, "finger_names": FINGER_SENSORS},
        )
    )

    contact_count: Any = config_field(
        RewTerm(
            func=mdp.contact_count,
            weight=0.1,
            params={"threshold": 0.01, "sensor_names": FINGER_SENSORS + [THUMB_SENSOR]},
        )
    )

    def __post_init__(self):
        if parent_post_init := getattr(super(), "__post_init__", None):
            parent_post_init()
        self.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"])
        self.fingers_to_object.params["thumb_name"] = THUMB_SENSOR
        self.fingers_to_object.params["finger_names"] = FINGER_SENSORS
        self.position_tracking.params["thumb_name"] = THUMB_SENSOR
        self.position_tracking.params["finger_names"] = FINGER_SENSORS
        if self.orientation_tracking:
            self.orientation_tracking.params["thumb_name"] = THUMB_SENSOR
            self.orientation_tracking.params["finger_names"] = FINGER_SENSORS
        self.success.params["thumb_name"] = THUMB_SENSOR
        self.success.params["finger_names"] = FINGER_SENSORS


@dataclass
class KukaAllegroMixinCfg:
    scene: KukaAllegroSceneCfg = config_field(KukaAllegroSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True))
    rewards: KukaAllegroReorientRewardCfg = config_field(KukaAllegroReorientRewardCfg())
    observations: StateObservationCfg = config_field(StateObservationCfg())
    actions: KukaAllegroRelJointPosActionCfg = config_field(KukaAllegroRelJointPosActionCfg())

    def __post_init__(self: lift.ReorientEnvCfg):
        if parent_post_init := getattr(super(), "__post_init__", None):
            parent_post_init()
        self.commands.object_pose.body_name = "palm_link"
        events = self.events.conditional_reset.params["terms"]
        events["reset_robot_wrist_joint"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names="iiwa7_joint_7")
        events["reset_object_to_target"].params["target_cfg"] = SceneEntityCfg("robot", body_names="palm_link")
        events["reset_object_to_target"].params["pose_range"] = {
            "x": [0.03, 0.07],
            "y": [-0.04, 0.04],
            "z": [0.02, 0.08],
        }
        # table/ground clearance: everything but the ground-mounted arm base (allegro finger links
        # are also named *_link_N, so exclude exactly iiwa7_link_0)
        self.events.conditional_reset.params["valid_criteria"][
            "robot_table_clearance"
        ].body_names = "(?!iiwa7_link_0$).*"
        # spread the reset bank over the grasp geometry, same bodies as fingers_to_object
        diversity_feature = self.events.conditional_reset.params.get("diversity_feature")
        if diversity_feature is not None:
            diversity_feature.body_names = ["palm_link", ".*_tip"]
        # finger closing-speed DR: armature sets tau/M.
        self.events.finger_closing_speed = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names="(index|middle|ring|thumb)_joint_(0|1|2|3)"),
                "armature_distribution_params": (0.1, 1.0),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )


@dataclass
class KukaAllegroReorientEnvCfg(KukaAllegroMixinCfg, lift.ReorientEnvCfg):
    pass


@dataclass
class KukaAllegroLiftEnvCfg(KukaAllegroMixinCfg, lift.LiftEnvCfg):
    pass
