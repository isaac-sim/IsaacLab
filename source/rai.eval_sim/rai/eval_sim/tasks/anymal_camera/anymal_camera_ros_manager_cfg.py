# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from rai.eval_sim.ros_manager import (
    OmniGraphCameraTermCfg,
    StaticTFBroadcasterCfg,
    TFBroadcasterCfg,
)

ENTITY = SceneEntityCfg("robot")
CAMERA = SceneEntityCfg("camera")

from ..anymal.anymal_ros_manager_cfg import AnymalDRosManagerCfg


@configclass
class AnymalDCameraRosManagerCfg(AnymalDRosManagerCfg):
    # inherit from AnymalDRosManager
    # add camera
    camera1 = OmniGraphCameraTermCfg(
        asset_cfg=CAMERA,
        namespace="camera1",
        enable_rgb=True,
        enable_depth=True,
        enable_info=True,
        rgb_topic="/rgb",
        depth_topic="/depth",
        info_topic="/camera_info",
    )
    # Add TF broadcasters
    tf_broadcaster = TFBroadcasterCfg(asset_cfg=[ENTITY], substep=50)
    static_tf_broadcaster = StaticTFBroadcasterCfg(asset_cfg=[CAMERA], additional_prim_paths=["/Robot/base/face_front"])
