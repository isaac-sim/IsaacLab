# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering-throughput benchmark variant of the Shadow Hand camera task."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.reorient.config.shadow_hand.feature_extractor import FeatureExtractorCfg
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_camera_env_cfg import ShadowHandCameraEnvCfg


@configclass
class ShadowHandCameraBenchmarkEnvCfg(ShadowHandCameraEnvCfg):
    """Benchmark configuration with the feature extractor CNN disabled.

    The tiled camera renders frames each step as normal, but the CNN forward pass is
    bypassed — zero embeddings are returned instead. This isolates rendering throughput
    from CNN inference overhead when profiling.

    The renderer backend and camera data types can still be selected via ``presets``::

        presets = newton_renderer  # benchmark with Newton renderer
        presets = ovrtx  # benchmark with OVRTX renderer
        presets = rgb  # benchmark RGB rendering only
        presets = depth, newton_renderer  # benchmark depth rendering with Newton
    """

    feature_extractor: FeatureExtractorCfg = FeatureExtractorCfg(enabled=False)
