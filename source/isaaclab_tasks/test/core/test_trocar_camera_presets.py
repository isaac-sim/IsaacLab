# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration-only regression for the Trocar task's explicit RTX preset."""

from isaaclab_tasks.contrib.assemble_trocar.g129_dex3_env_cfg import G1AssembleTrocarEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets


def test_explicit_rtx_preserves_trocar_render_settings():
    """Selecting RTX explicitly must retain the task's material and image settings on all cameras."""
    cfg = resolve_presets(G1AssembleTrocarEnvCfg(), selected=("rtx",))
    for camera in (cfg.scene.front_camera, cfg.scene.left_wrist_camera, cfg.scene.right_wrist_camera):
        settings = camera.renderer_cfg.global_settings
        assert settings.enable_translucency is True
        assert settings.antialiasing_mode == "DLAA"
        assert settings.carb_settings == {
            "rtx.raytracing.fractionalCutoutOpacity": True,
            "renderer.scenePartitioning.enabled": False,
        }
