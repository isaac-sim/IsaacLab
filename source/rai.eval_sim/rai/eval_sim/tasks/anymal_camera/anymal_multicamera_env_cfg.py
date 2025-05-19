# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from rai.eval_sim.tasks.anymal import AnymalDEnvCfg


@configclass
class AnymalDMultiCameraEnvCfg(AnymalDEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
            height=480,
            width=640,
            data_types=[],  # leave data empty when utilizing omnigraph based publishers
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        )
        self.scene.camera2 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/right_cam",
            height=480,
            width=640,
            data_types=[],  # leave data empty when utilizing omnigraph based publishers
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.510, -0.2, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        )
        self.scene.camera3 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/left_cam",
            height=480,
            width=640,
            data_types=[],  # leave data empty when utilizing omnigraph based publishers
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.510, 0.2, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        )
