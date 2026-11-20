# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
public camera configuration
include the basic configuration for different types of cameras, support scene-specific parameter customization
"""

from collections.abc import Sequence

from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils.presets import MultiBackendRendererCfg


@configclass
class CameraBaseCfg:
    """camera base configuration class

    provide the default configuration for different types of cameras, support scene-specific parameter customization
    """

    @classmethod
    def get_camera_config(
        cls,
        prim_path: str = "{ENV_REGEX_NS}/Robot/d435_link/front_cam",
        update_period: float = 0.02,
        height: int = 480,
        width: int = 640,
        focal_length: float = 7.6,
        focus_distance: float = 400.0,
        horizontal_aperture: float = 20.0,
        clipping_range: tuple[float, float] = (0.1, 1.0e5),
        pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rot_offset: tuple[float, float, float, float] = (0.5, -0.5, 0.5, -0.5),
        data_types: Sequence[str] | None = None,
    ) -> CameraCfg:
        """Get a pinhole camera configuration.

        Args:
            prim_path: the path of the camera in the scene
            update_period: update period (seconds)
            height: image height (pixels)
            width: image width (pixels)
            focal_length: focal length
            focus_distance: focus distance
            horizontal_aperture: horizontal aperture
            clipping_range: clipping range (near clipping plane, far clipping plane)
            pos_offset: position offset (x, y, z)
            rot_offset: rotation offset quaternion
            data_types: data type list

        Returns:
            CameraCfg: camera configuration
        """
        if data_types is None:
            data_types = ("rgb",)

        return CameraCfg(
            prim_path=prim_path,
            renderer_cfg=MultiBackendRendererCfg(default=IsaacRtxRendererCfg()),
            update_period=update_period,
            height=height,
            width=width,
            data_types=list(data_types),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                focus_distance=focus_distance,
                horizontal_aperture=horizontal_aperture,
                clipping_range=clipping_range,
            ),
            offset=CameraCfg.OffsetCfg(pos=pos_offset, rot=rot_offset, convention="ros"),
        )


@configclass
class CameraPresets:
    """camera preset configuration collection

    include the common camera configuration preset for different scenes
    """

    @classmethod
    def g1_front_camera(cls, **overrides) -> CameraCfg:
        params = {
            "height": 240,
            "width": 320,
            "focal_length": 8.7,
            "horizontal_aperture": 9.0,
        }
        params.update(overrides)
        return CameraBaseCfg.get_camera_config(**params)

    @classmethod
    def left_dex3_wrist_camera(cls, **overrides) -> CameraCfg:
        """left wrist camera configuration"""
        params = {
            "prim_path": "{ENV_REGEX_NS}/Robot/left_hand_camera_base_link/left_wrist_camera",
            "height": 240,
            "width": 320,
            "update_period": 0.02,
            "data_types": ["rgb"],
            "focal_length": 11.4,
            "focus_distance": 400.0,
            "horizontal_aperture": 19.2,
            "clipping_range": (0.05, 1.0e5),
            "pos_offset": (-0.01, -0.025, 0.1),
            "rot_offset": (0.93721, -0.15211, 0.31385, 0.00166),
        }
        params.update(overrides)
        return CameraBaseCfg.get_camera_config(**params)

    @classmethod
    def right_dex3_wrist_camera(cls, **overrides) -> CameraCfg:
        """right wrist camera configuration"""
        params = {
            "prim_path": "{ENV_REGEX_NS}/Robot/right_hand_camera_base_link/right_wrist_camera",
            "height": 240,
            "width": 320,
            "update_period": 0.02,
            "data_types": ["rgb"],
            "focal_length": 11.4,
            "focus_distance": 400.0,
            "horizontal_aperture": 19.2,
            "clipping_range": (0.05, 1.0e5),
            "pos_offset": (-0.01, 0.025, 0.1),
            "rot_offset": (0.93536, 0.13055, 0.32871, 0.00283),
        }
        params.update(overrides)
        return CameraBaseCfg.get_camera_config(**params)
