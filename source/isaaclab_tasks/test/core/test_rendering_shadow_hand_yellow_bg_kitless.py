# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less golden render test: Shadow Hand with yellow camera background via OVRTX (RGB only)."""

import os

# OVRTX uses Vulkan, which requires a display socket even in headless mode.
os.environ.setdefault("DISPLAY", ":0")

import pytest  # noqa: E402

from isaaclab.utils.configclass import configclass  # noqa: E402

from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_camera_env import ShadowHandCameraEnv  # noqa: E402
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_camera_env_cfg import (  # noqa: E402
    ShadowHandCameraEnvCfg,
    ShadowHandTiledCameraCfg,
    _ShadowHandBaseTiledCameraCfg,
)
from rendering_test_utils import (  # noqa: E402
    _apply_overrides_to_env_cfg,
    _physics_preset_name,
    _redirect_ovrtx_renderer_log_to_stdout,
    make_require_ovlibs_install_fixture,
    validate_camera_outputs,
)

pytestmark = [pytest.mark.isaacsim_ci]

_YELLOW = (1.0, 1.0, 0.0)
_TEST_NAME = "shadow_hand_yellow_bg"

_require_ovlibs_install_fixture = make_require_ovlibs_install_fixture()


@configclass
class _YellowBgCameraCfg(_ShadowHandBaseTiledCameraCfg):
    data_types: list[str] = ["rgb"]
    background_color: tuple[float, float, float] | None = _YELLOW


@configclass
class _YellowBgTiledCameraCfg(ShadowHandTiledCameraCfg):
    default: _YellowBgCameraCfg = _YellowBgCameraCfg()
    rgb: _YellowBgCameraCfg = _YellowBgCameraCfg()


@configclass
class _YellowBgEnvCfg(ShadowHandCameraEnvCfg):
    tiled_camera: _YellowBgTiledCameraCfg = _YellowBgTiledCameraCfg()


@pytest.mark.parametrize("physics_backend,renderer", [("newton", "ovrtx")])
def test_rgb_yellow_background_ovrtx(ovstage_variant, physics_backend, renderer):
    """Kit-less golden render test: RGB output with yellow background via OVRTX."""
    env_cfg = _YellowBgEnvCfg()
    env_cfg.feature_extractor.enabled = False
    env_cfg.scene.num_envs = 4
    env_cfg = _apply_overrides_to_env_cfg(env_cfg, [f"presets={_physics_preset_name(physics_backend)},{renderer},rgb"])
    _redirect_ovrtx_renderer_log_to_stdout(env_cfg)

    env = None
    try:
        env = ShadowHandCameraEnv(env_cfg)
        validate_camera_outputs(
            _TEST_NAME,
            physics_backend,
            renderer,
            env._tiled_camera.data.output,
            max_different_pixels_percentage=5.0,
            comparison_scores=[],
        )
    finally:
        if env is not None:
            env.close()
            env = None
