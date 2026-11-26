# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trocar camera rendering settings and task-owned policy augmentation."""

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


def test_policy_camera_crop_preserves_pixels_and_storage():
    import torch

    from isaaclab_tasks.contrib.assemble_trocar.mdp.observations import crop_policy_camera

    image = torch.arange(16, dtype=torch.uint8).reshape(1, 4, 4, 1)
    assert crop_policy_camera(image, scale=0) is image
    cropped = crop_policy_camera(image, scale=0.5)
    assert cropped.flatten().tolist() == [5, 6, 9, 10]
    cropped[0, 0, 0, 0] = 99
    assert image[0, 1, 1, 0] == 99
    original = image.clone()
    torch.manual_seed(7)
    random = crop_policy_camera(image, scale=0.5, random_crop=True)
    torch.manual_seed(7)
    torch.testing.assert_close(random, crop_policy_camera(image, scale=0.5, random_crop=True))
    random.zero_()
    torch.testing.assert_close(image, original)


def test_policy_crop_uses_final_task_configuration():
    from isaaclab_tasks.contrib.assemble_trocar.mdp.observations import crop_policy_camera

    cfg = G1AssembleTrocarEnvCfg()
    cfg.policy_crop_scale = 0.5
    cfg.policy_crop_random = True
    cfg.validate_config()
    cfg.validate_config()  # Repeated validation must not stack augmentation twice.
    for name in ("front_camera", "left_wrist_camera", "right_wrist_camera"):
        processors = getattr(cfg.observations.camera_images, name).tensor_processors
        assert len(processors) == 1 and processors[0].func is crop_policy_camera
        assert processors[0].params == {"scale": 0.5, "random_crop": True}
    cfg.policy_crop_scale = 0
    cfg.validate_config()
    assert cfg.observations.camera_images.front_camera.tensor_processors is None
