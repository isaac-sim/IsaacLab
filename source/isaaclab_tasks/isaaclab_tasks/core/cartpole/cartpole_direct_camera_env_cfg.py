# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import config_field
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg
from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg


@dataclass
class CartpoleTiledCameraCfg(PresetCfg):
    @dataclass
    class BaseCartpoleTiledCameraCfg(CameraCfg):
        prim_path: str = config_field("{ENV_REGEX_NS}/Camera")
        offset: CameraCfg.OffsetCfg = config_field(
            CameraCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world")
        )
        data_types: list[str] = config_field([])
        spawn: sim_utils.PinholeCameraCfg = config_field(
            sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
            )
        )
        width: int = config_field(96)
        height: int = config_field(96)
        renderer_cfg: MultiBackendRendererCfg = config_field(MultiBackendRendererCfg())

    default: Any = config_field(BaseCartpoleTiledCameraCfg(data_types=["rgb"]))
    depth: Any = config_field(BaseCartpoleTiledCameraCfg(data_types=["depth"]))
    albedo: Any = config_field(BaseCartpoleTiledCameraCfg(data_types=["albedo"]))
    semantic_segmentation: Any = config_field(BaseCartpoleTiledCameraCfg(data_types=["semantic_segmentation"]))
    simple_shading_constant_diffuse: Any = config_field(
        BaseCartpoleTiledCameraCfg(data_types=["simple_shading_constant_diffuse"])
    )
    simple_shading_diffuse_mdl: Any = config_field(
        BaseCartpoleTiledCameraCfg(data_types=["simple_shading_diffuse_mdl"])
    )
    simple_shading_full_mdl: Any = config_field(BaseCartpoleTiledCameraCfg(data_types=["simple_shading_full_mdl"]))
    rgb: Any = config_field(default)


@dataclass
class CartpoleCameraEnvCfg(PresetCfg):
    @dataclass
    class BaseCartpoleCameraEnvCfg(CartpoleEnvCfg):
        """Camera variant of :class:`CartpoleEnvCfg` — only the fields that differ are overridden."""

        # camera
        tiled_camera: CartpoleTiledCameraCfg = config_field(CartpoleTiledCameraCfg())
        write_image_to_file: Any = config_field(False)

        frame_stack: int = config_field(2)
        """Number of frames to stack along the channel dimension.

        Values less than two disable stacking.
        """

        # spaces: single-frame channels + default spatial size. At env init, height/width
        # are replaced with the tiled camera size and channels are expanded by frame_stack.
        # Only the channel count must stay in sync with the camera data type (presets set this).
        observation_space: Any = config_field([3, 96, 96])
        state_space: Any = config_field(4)

        # scene: fewer, more-spaced envs and no fabric cloning so the camera renders cleanly
        scene: InteractiveSceneCfg = config_field(
            InteractiveSceneCfg(num_envs=512, env_spacing=20.0, replicate_physics=True)
        )

        # reset: smaller initial pole angle than the proprioceptive task
        initial_pole_angle_range: Any = config_field((-0.125 * math.pi, 0.125 * math.pi))  # [rad]

        def __post_init__(self):
            self.sim.default_visualizer_cfg = VisualizerCfg(eye=(20.0, 20.0, 20.0))

    default: Any = config_field(BaseCartpoleCameraEnvCfg())
    depth: Any = config_field(BaseCartpoleCameraEnvCfg(observation_space=[1, 96, 96]))
    albedo: Any = config_field(BaseCartpoleCameraEnvCfg())
    semantic_segmentation: Any = config_field(BaseCartpoleCameraEnvCfg(observation_space=[4, 96, 96]))
    simple_shading_constant_diffuse: Any = config_field(BaseCartpoleCameraEnvCfg())
    simple_shading_diffuse_mdl: Any = config_field(BaseCartpoleCameraEnvCfg())
    simple_shading_full_mdl: Any = config_field(BaseCartpoleCameraEnvCfg())
