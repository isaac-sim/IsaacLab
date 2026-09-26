# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Kuka-Allegro lift and reorient environments."""

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

from ... import lift_env_cfg as lift
from ... import mdp
from .camera_cfg import FINGERTIP_LIST, StateObservationCfg

THUMB_SENSOR = "thumb_link_3_object_s"
"""Contact sensor of the thumb."""

FINGER_SENSORS = [f"{name}_object_s" for name in FINGERTIP_LIST if name != "thumb_link_3"]
"""Contact sensors of the remaining fingers."""


##
# Scene definition
##


@configclass
class KukaAllegroSceneCfg(lift.SceneCfg):
    """Kuka-Allegro scene for the lift and reorient tasks.

    The ``base_camera`` and ``wrist_camera`` slots are left unset for the state task; the camera
    environment configuration populates them.
    """

    robot: ArticulationCfg = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    base_camera: CameraCfg | None = None
    wrist_camera: CameraCfg | None = None

    def __post_init__(self):
        super().__post_init__()
        for link_name in FINGERTIP_LIST:
            setattr(
                self,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )


##
# MDP settings
##


@configclass
class KukaAllegroRelJointPosActionCfg:
    """Relative joint position targets for all joints."""

    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class KukaAllegroReorientRewardCfg(lift.RewardsCfg):
    """Reward terms for the MDP, with the Allegro finger contact sensors filled in."""

    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=1.0,
        params={"threshold": 0.01, "thumb_name": THUMB_SENSOR, "finger_names": FINGER_SENSORS},
    )

    contact_count = RewTerm(
        func=mdp.contact_count,
        weight=0.1,
        params={"threshold": 0.01, "sensor_names": FINGER_SENSORS + [THUMB_SENSOR]},
    )

    def __post_init__(self):
        super().__post_init__()
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


##
# Environment configuration
##


@configclass
class KukaAllegroMixinCfg:
    """Kuka-Allegro specific scene, observation, action and reward terms, mixed into the task configurations."""

    scene: KukaAllegroSceneCfg = KukaAllegroSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True)
    rewards: KukaAllegroReorientRewardCfg = KukaAllegroReorientRewardCfg()
    observations: StateObservationCfg = StateObservationCfg()
    actions: KukaAllegroRelJointPosActionCfg = KukaAllegroRelJointPosActionCfg()

    def __post_init__(self):
        super().__post_init__()
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
        # finger closing-speed randomization: the armature sets the effort-to-inertia ratio
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

    def validate_config(self) -> None:
        """Select image terms from the resolved camera data types before managers initialize."""
        for sensor_name, group_name, term_name in (
            ("base_camera", "base_image", "object_observation_b"),
            ("wrist_camera", "wrist_image", "wrist_observation"),
        ):
            camera = getattr(self.scene, sensor_name)
            if camera is None:
                continue
            term = getattr(getattr(self.observations, group_name), term_name)
            data_type = camera.data_types[0]
            if data_type in ("depth", "distance_to_image_plane", "distance_to_camera"):
                term.func = base_mdp.image_depth
            elif "segmentation" in data_type:
                term.func = base_mdp.image_segmentation
            else:
                term.func = base_mdp.image_rgb
            term.params["data_type"] = data_type


@configclass
class KukaAllegroReorientEnvCfg(KukaAllegroMixinCfg, lift.ReorientEnvCfg):
    """Kuka-Allegro object reorientation environment."""


@configclass
class KukaAllegroLiftEnvCfg(KukaAllegroMixinCfg, lift.LiftEnvCfg):
    """Kuka-Allegro object lifting environment."""
