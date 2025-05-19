# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from rai.eval_sim.tasks.anymal import AnymalDEnvCfg


@configclass
class AnymalDCameraEnvCfg(AnymalDEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/face_front/front_cam",
            update_period=0.1,  # capture rate of the camera in the Synthetic data pipeline
            height=480,
            width=640,
            data_types=["distance_to_image_plane", "rgb"],  # use distance_to_image_plane and rgb for RGBD cameras
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        )
        self.sim.render_interval = 50  # render step every 50 sim steps: 0.002*50 = 0.1 -> 10 Hz, rendering rate controls omnigraph camera publisher
