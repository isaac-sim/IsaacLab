# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering correctness tests for camera-based registered tasks."""

# Launch Isaac Sim Simulator first for kit-based combinations.
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import gymnasium as gym  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from rendering_test_utils import (  # noqa: E402
    MAX_DIFFERENT_PIXELS_PERCENTAGE_BY_ENV_NAME,
    make_attach_comparison_properties_fixture,
    make_determinism_fixture,
    make_generate_html_report_fixture,
    maybe_save_stage,
    validate_camera_outputs,
)

pytestmark = pytest.mark.isaacsim_ci

_COMPARISON_SCORES: list[dict] = []

_determinism_fixture = make_determinism_fixture()
_generate_html_report_fixture = make_generate_html_report_fixture(_COMPARISON_SCORES, Path(__file__).stem + ".html")
_attach_comparison_properties_fixture = make_attach_comparison_properties_fixture(_COMPARISON_SCORES)


def _collect_camera_outputs(env: object) -> dict[str, dict[str, torch.Tensor]]:
    """Collect camera outputs from env.scene.sensors."""
    base = getattr(env, "unwrapped", env)
    outputs = {}

    scene = getattr(base, "scene", None)
    if scene is not None:
        sensors = getattr(scene, "sensors", None)
        if sensors is not None:
            for name, sensor in sensors.items():
                data = getattr(sensor, "data", None)
                output = getattr(data, "output", None) if data is not None else None
                if not isinstance(output, dict):
                    continue

                tensor_output = {}
                for data_type, value in output.items():
                    tensor = value if isinstance(value, torch.Tensor) else value.torch
                    if tensor.numel() > 0:
                        tensor_output[data_type] = tensor
                if tensor_output:
                    outputs[name] = tensor_output

    return outputs


# Task IDs that expose camera/tiled_camera image observations; each is validated for non-blank rendering.
# The max different pixels percentage is set based on the screen space taken up by the env.
# The ``presets`` column selects a data-type variant on the consolidated cartpole camera task;
# ``None`` uses the default.
_RENDER_CORRECTNESS_TASK_IDS = [
    ("Isaac-Cartpole-Camera-Direct", None, "cartpole"),
    ("Isaac-Cartpole-Camera-Direct", "albedo", "cartpole"),
    ("Isaac-Cartpole-Camera-Direct", "depth", "cartpole"),
    ("Isaac-Cartpole-Camera-Direct", "rgb", "cartpole"),
    ("Isaac-Cartpole-Camera-Direct", "simple_shading_constant_diffuse", "cartpole"),
    ("Isaac-Cartpole-Camera-Direct", "simple_shading_diffuse_mdl", "cartpole"),
    ("Isaac-Cartpole-Camera-Direct", "simple_shading_full_mdl", "cartpole"),
    pytest.param(
        "Isaac-Repose-Cube-Shadow-Vision-Direct-v0",
        None,
        "shadow_hand",
        # The Shadow-Vision render is right at the SSIM/diff-pixel tolerance and intermittently
        # exceeds the 3% diff threshold by a fraction of a percent. Allow up to 3 attempts and
        # require at least one pass while we tighten the validation tolerances for this scene.
        marks=pytest.mark.flaky(max_runs=3, min_passes=1),
    ),
]


@pytest.mark.parametrize("task_id, presets, env_name", _RENDER_CORRECTNESS_TASK_IDS)
def test_rendering_registered_tasks(task_id: str, presets: str | None, env_name: str):
    """Test registered tasks rendering correctness."""
    env = None

    try:
        from isaaclab_tasks.utils.hydra import resolve_presets
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
        env_cfg = resolve_presets(env_cfg, {presets} if presets else frozenset())
        env_cfg.sim.device = "cuda:0"
        env_cfg.scene.num_envs = 4

        env = gym.make(task_id, cfg=env_cfg)
        unwrapped: Any = env.unwrapped
        sim = getattr(unwrapped, "sim", None)
        if sim is not None:
            sim._app_control_on_stop_handle = None

        maybe_save_stage(f"registered_tasks_{task_id}", "default_physics", "default_renderer", "stage")

        camera_outputs_nested_dict = _collect_camera_outputs(env)
        num_camera_outputs = len(camera_outputs_nested_dict)
        assert num_camera_outputs == 1, f"[{task_id}] Expected 1 camera output, got {num_camera_outputs}."

        camera_outputs = next(iter(camera_outputs_nested_dict.values()))

        validate_camera_outputs(
            f"registered_tasks/{task_id}",
            "default_physics",
            "default_renderer",
            camera_outputs,
            max_different_pixels_percentage=MAX_DIFFERENT_PIXELS_PERCENTAGE_BY_ENV_NAME[env_name],
            comparison_scores=_COMPARISON_SCORES,
        )
    finally:
        if env is not None:
            env.close()

            # This invokes camera sensor and renderer cleanup explicitly before pytest teardown, otherwise OV
            # native code could probably complain about leaks and trigger segmentation fault.
            env = None
