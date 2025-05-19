# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from rai.eval_sim.ros_manager import OmniGraphCameraTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

ENTITY = SceneEntityCfg("robot")
CAMERA = SceneEntityCfg("camera")
CAMERA2 = SceneEntityCfg("camera2")
CAMERA3 = SceneEntityCfg("camera3")


from ..anymal.anymal_ros_manager_cfg import AnymalDRosManagerCfg


@configclass
class AnymalDMultiCameraRosManagerCfg(AnymalDRosManagerCfg):
    # inherit from AnymalDRosManager
    # add cameras
    camera1 = OmniGraphCameraTermCfg(
        asset_cfg=CAMERA,
        namespace="camera1",
        enable_rgb=True,
        enable_depth=True,
        enable_info=True,
        rgb_topic="/rgb/image",
        depth_topic="/depth",
        info_topic="/camera_info",
    )
    camera2 = OmniGraphCameraTermCfg(
        asset_cfg=CAMERA2,
        namespace="camera2",
        enable_rgb=True,
        enable_depth=True,
        enable_info=True,
        rgb_topic="/rgb/image",
        depth_topic="/depth",
        info_topic="/camera_info",
    )
    camera3 = OmniGraphCameraTermCfg(
        asset_cfg=CAMERA3,
        namespace="camera3",
        enable_rgb=True,
        enable_depth=True,
        enable_info=True,
        rgb_topic="/rgb/image",
        depth_topic="/depth",
        info_topic="/camera_info",
    )
