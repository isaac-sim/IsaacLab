# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

from .feature_extractor import FeatureExtractorCfg
from .shadow_hand_env_cfg import ShadowHandEnvCfg


@configclass
class ShadowHandVisionRendererCfg(PresetCfg):
    """Renderer backend presets for the shadow hand vision environment.

    Select a renderer via the ``presets`` CLI argument, e.g.::

        presets = warp  # Newton Warp software renderer
        presets = ovrtx  # OVRTX high-fidelity renderer
        presets = isaacsim_rtx  # Isaac Sim RTX (same as default)
    """

    default: IsaacRtxRendererCfg = IsaacRtxRendererCfg()
    """Isaac RTX renderer (default). Used when no renderer preset is specified."""

    warp: NewtonWarpRendererCfg = NewtonWarpRendererCfg()
    """Newton Warp software renderer."""

    ovrtx: OVRTXRendererCfg = OVRTXRendererCfg()
    """OVRTX high-fidelity RTX renderer."""

    isaacsim_rtx: IsaacRtxRendererCfg = IsaacRtxRendererCfg()
    """Isaac Sim RTX renderer (alias for default)."""


@configclass
class _ShadowHandBaseTiledCameraCfg(TiledCameraCfg):
    """Base tiled camera configuration for the shadow hand vision environment.

    This is an internal config used by :class:`ShadowHandVisionTiledCameraCfg` presets and
    by derived env configs that hard-code a specific data type. It embeds
    :class:`ShadowHandVisionRendererCfg` so the renderer backend can still be selected via
    the ``presets`` CLI argument.
    """

    prim_path: str = "/World/envs/env_.*/Camera"
    offset: TiledCameraCfg.OffsetCfg = TiledCameraCfg.OffsetCfg(
        pos=(0, -0.35, 1.0), rot=(0.0, 0.7071, 0.0, 0.7071), convention="world"
    )
    data_types: list[str] = []
    spawn: sim_utils.PinholeCameraCfg = sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    )
    width: int = 120
    height: int = 120
    renderer_cfg: ShadowHandVisionRendererCfg = ShadowHandVisionRendererCfg()


@configclass
class ShadowHandVisionTiledCameraCfg(PresetCfg):
    """Camera data-type presets for the shadow hand vision environment.

    Each preset selects which image modalities are captured. The selected data types must
    match :attr:`FeatureExtractorCfg.data_types` so the CNN receives the expected channels.

    Select a data-type preset via the ``presets`` CLI argument, e.g.::

        presets = rgb  # RGB only (3 channels)
        presets = albedo  # albedo (3 channels)
        presets = simple_shading_constant_diffuse  # simple shading, constant diffuse (3 channels)

    Renderer and data-type presets can be combined::

        presets = warp, rgb
    """

    default: _ShadowHandBaseTiledCameraCfg = _ShadowHandBaseTiledCameraCfg(
        data_types=["rgb", "depth", "semantic_segmentation"]
    )
    """Default: RGB + depth + semantic segmentation (7 CNN input channels)."""

    full: _ShadowHandBaseTiledCameraCfg = _ShadowHandBaseTiledCameraCfg(
        data_types=["rgb", "depth", "semantic_segmentation"]
    )
    """Full modalities: RGB + depth + semantic segmentation (7 channels). Alias for default."""

    rgb: _ShadowHandBaseTiledCameraCfg = _ShadowHandBaseTiledCameraCfg(data_types=["rgb"])
    """RGB only (3 CNN input channels)."""

    albedo: _ShadowHandBaseTiledCameraCfg = _ShadowHandBaseTiledCameraCfg(data_types=["albedo"])
    """Albedo (3 CNN input channels)."""

    simple_shading_constant_diffuse: _ShadowHandBaseTiledCameraCfg = _ShadowHandBaseTiledCameraCfg(
        data_types=["simple_shading_constant_diffuse"]
    )
    """Simple shading with constant diffuse (3 CNN input channels)."""

    simple_shading_diffuse_mdl: _ShadowHandBaseTiledCameraCfg = _ShadowHandBaseTiledCameraCfg(
        data_types=["simple_shading_diffuse_mdl"]
    )
    """Simple shading with diffuse MDL (3 CNN input channels)."""

    simple_shading_full_mdl: _ShadowHandBaseTiledCameraCfg = _ShadowHandBaseTiledCameraCfg(
        data_types=["simple_shading_full_mdl"]
    )
    """Simple shading with full MDL (3 CNN input channels)."""

    depth: _ShadowHandBaseTiledCameraCfg = _ShadowHandBaseTiledCameraCfg(data_types=["depth"])
    """Depth only (1 channel).

    .. warning::
        This preset is intended for **benchmarking only**. The keypoint-regression CNN
        cannot be meaningfully trained from depth alone. Use it with
        :class:`ShadowHandVisionBenchmarkEnvCfg` (``feature_extractor.enabled=False``)
        to measure pure depth-rendering throughput, e.g.::

            presets=depth          # depth rendering, default renderer
            presets=depth,warp     # depth rendering with Warp renderer
            presets=depth,ovrtx    # depth rendering with OVRTX renderer
    """


@configclass
class ShadowHandVisionEnvCfg(ShadowHandEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1225, env_spacing=2.0, replicate_physics=True)

    # camera — data-type and renderer backend selectable via CLI presets
    tiled_camera: ShadowHandVisionTiledCameraCfg = ShadowHandVisionTiledCameraCfg()
    feature_extractor: FeatureExtractorCfg = FeatureExtractorCfg()

    # env
    observation_space = 164 + 27  # state observation + vision CNN embedding
    state_space = 187 + 27  # asymmetric states + vision CNN embedding


@configclass
class ShadowHandVisionEnvPlayCfg(ShadowHandVisionEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.0, replicate_physics=True)
    # inference for CNN
    feature_extractor: FeatureExtractorCfg = FeatureExtractorCfg(train=False, load_checkpoint=True)


@configclass
class ShadowHandVisionBenchmarkEnvCfg(ShadowHandVisionEnvCfg):
    """Benchmark configuration with the feature extractor CNN disabled.

    The tiled camera renders frames each step as normal, but the CNN forward pass is
    bypassed — zero embeddings are returned instead. This isolates rendering throughput
    from CNN inference overhead when profiling.

    The renderer backend and camera data types can still be selected via ``presets``::

        presets = warp  # benchmark with Warp renderer
        presets = ovrtx  # benchmark with OVRTX renderer
        presets = rgb  # benchmark RGB rendering only
        presets = depth, warp  # benchmark depth rendering with Warp
    """

    feature_extractor: FeatureExtractorCfg = FeatureExtractorCfg(enabled=False)
