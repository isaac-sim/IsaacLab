# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the direct-workflow cartpole camera environment."""

from __future__ import annotations

import math

from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.core.cartpole.cartpole_common import CartpoleTiledCameraCfg
from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg, CartpoleSceneCfg
from isaaclab_tasks.utils import PresetCfg


@configclass
class CartpoleCameraSceneCfg(CartpoleSceneCfg):
    """Cartpole scene with a selectable camera and no view-obstructing ground plane."""

    ground = None
    tiled_camera: CartpoleTiledCameraCfg = CartpoleTiledCameraCfg()


@configclass
class CartpoleCameraEnvCfg(PresetCfg):
    """Cartpole camera environment with a selectable camera data type.

    The selector also picks the matching tiled camera preset and its rendering backend through
    :attr:`CartpoleCameraSceneCfg.tiled_camera`.
    """

    @configclass
    class BaseCartpoleCameraEnvCfg(CartpoleEnvCfg):
        """Camera variant of :class:`CartpoleEnvCfg`; only the fields that differ are overridden."""

        write_image_to_file = False

        frame_stack: int = 2
        """Number of frames to stack along the channel dimension.

        Values less than two disable stacking.
        """

        # spaces: single-frame channels + default spatial size. At env init, height/width
        # are replaced with the tiled camera size and channels are expanded by frame_stack.
        # Only the channel count must stay in sync with the camera data type (presets set this).
        observation_space = [3, 96, 96]
        state_space = 4

        # scene: fewer, more-spaced envs and no fabric cloning so the camera renders cleanly
        scene: CartpoleCameraSceneCfg = CartpoleCameraSceneCfg(num_envs=512, env_spacing=20.0, replicate_physics=True)

        # reset: smaller initial pole angle than the proprioceptive task
        initial_pole_angle_range = (-0.125 * math.pi, 0.125 * math.pi)  # [rad]

        def __post_init__(self):
            super().__post_init__()
            self.sim.default_visualizer_cfg = VisualizerCfg(eye=(20.0, 20.0, 20.0))

    default = BaseCartpoleCameraEnvCfg()
    depth = BaseCartpoleCameraEnvCfg(observation_space=[1, 96, 96])
    albedo = BaseCartpoleCameraEnvCfg()
    semantic_segmentation = BaseCartpoleCameraEnvCfg(observation_space=[4, 96, 96])
    simple_shading_constant_diffuse = BaseCartpoleCameraEnvCfg()
    simple_shading_diffuse_mdl = BaseCartpoleCameraEnvCfg()
    simple_shading_full_mdl = BaseCartpoleCameraEnvCfg()
