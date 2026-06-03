# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg
from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg


@configclass
class CartpoleTiledCameraCfg(PresetCfg):
    @configclass
    class BaseCartpoleTiledCameraCfg(CameraCfg):
        prim_path: str = "/World/envs/env_.*/Camera"
        offset: CameraCfg.OffsetCfg = CameraCfg.OffsetCfg(
            pos=(-5.0, 0.0, 2.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"
        )
        data_types: list[str] = []
        spawn: sim_utils.PinholeCameraCfg = sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        )
        width: int = 100
        height: int = 100
        renderer_cfg: MultiBackendRendererCfg = MultiBackendRendererCfg()

    default = BaseCartpoleTiledCameraCfg(data_types=["rgb"])
    depth = BaseCartpoleTiledCameraCfg(data_types=["depth"])
    albedo = BaseCartpoleTiledCameraCfg(data_types=["albedo"])
    semantic_segmentation = BaseCartpoleTiledCameraCfg(data_types=["semantic_segmentation"])
    simple_shading_constant_diffuse = BaseCartpoleTiledCameraCfg(data_types=["simple_shading_constant_diffuse"])
    simple_shading_diffuse_mdl = BaseCartpoleTiledCameraCfg(data_types=["simple_shading_diffuse_mdl"])
    simple_shading_full_mdl = BaseCartpoleTiledCameraCfg(data_types=["simple_shading_full_mdl"])
    rgb = default


@configclass
class CartpoleCameraEnvCfg(PresetCfg):
    @configclass
    class BaseCartpoleCameraEnvCfg(CartpoleEnvCfg):
        """Camera variant of :class:`CartpoleEnvCfg` — only the fields that differ are overridden."""

        # camera
        tiled_camera: CartpoleTiledCameraCfg = CartpoleTiledCameraCfg()
        write_image_to_file = False

        frame_stack: int = -1
        """Number of frames to stack along the channel dim.

        ``-1`` (default) auto-resolves to ``2`` for the Newton + Warp combo and ``1`` otherwise.
        Set to ``1`` to force single-frame; set to ``N > 1`` to force an explicit stack size.
        """

        # spaces: an image instead of the 4-dim joint-state vector
        observation_space = [3, 100, 100]

        # change viewer settings
        viewer = ViewerCfg(eye=(20.0, 20.0, 20.0))

        # scene: fewer, more-spaced envs and no fabric cloning so the camera renders cleanly
        scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=20.0, replicate_physics=True)

        # reset: smaller initial pole angle than the proprioceptive task
        initial_pole_angle_range = [-0.125, 0.125]

    default = BaseCartpoleCameraEnvCfg()
    depth = BaseCartpoleCameraEnvCfg(observation_space=[1, 100, 100])
    albedo = BaseCartpoleCameraEnvCfg()
    semantic_segmentation = BaseCartpoleCameraEnvCfg(observation_space=[4, 100, 100])
    simple_shading_constant_diffuse = BaseCartpoleCameraEnvCfg()
    simple_shading_diffuse_mdl = BaseCartpoleCameraEnvCfg()
    simple_shading_full_mdl = BaseCartpoleCameraEnvCfg()
