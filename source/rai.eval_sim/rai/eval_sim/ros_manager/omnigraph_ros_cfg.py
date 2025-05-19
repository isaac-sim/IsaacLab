# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import omnigraph_ros


@configclass
class OmniGraphTermCfg:
    class_type: object = MISSING
    """OmniGraphTerm object linked to this config"""
    graph_name: str = "ActionGraph"
    """Name of omnigraph to create"""
    asset_cfg: SceneEntityCfg | None = None
    """Defines which asset is used for connecting to omnigraph nodes"""
    namespace: str = ""
    """ROS namespace to append to topic names"""


@configclass
class OmniGraphCameraTermCfg(OmniGraphTermCfg):
    class_type = omnigraph_ros.OmniGraphCameraTerm
    graph_name: str | None = None
    """Name of omnigraph to create, if None will auto name to ActionGraph_{camera_name}"""
    asset_cfg: SceneEntityCfg = MISSING
    """Defines which camera asset to use in camera omnigraph nodes"""

    namespace: str = ""
    """ROS namespace to append to topic names"""
    enable_rgb: bool = True
    """Enabling RGB publisher of message type sensor_msgs/Image"""
    enable_depth: bool = False
    """Enabling Pseudo-Depth with 32FC1 output publisher of message type sensor_msgs/Image"""
    enable_info: bool = False
    """Enabling camera info publisher of message type sensor_msgs/CameraInfo"""
    rgb_topic: str = "camera/rgb/raw"
    """Topic name for RGB publisher"""
    depth_topic: str = "camera/depth"
    """Topic name for Pseudo-Depth publisher"""
    info_topic: str = "camera/camera_info"
    """Topic name for Camera Info publisher"""
