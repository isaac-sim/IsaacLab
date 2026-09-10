# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the command-line arguments passed to Kit at startup."""

import sys

import pytest

from isaaclab.app.app_launcher import AppLauncher, _sanitize_sys_argv_for_kit
from isaaclab.utils import renderers
from isaaclab.utils.renderers import ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING


@pytest.fixture(autouse=True)
def _isolate_scene_partition_default(monkeypatch):
    """Keep the process-wide scene-partitioning default from leaking between tests."""
    monkeypatch.setattr(renderers, "_scene_partition_default", True)
    monkeypatch.delenv("ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION", raising=False)


def test_sanitize_sys_argv_removes_trailing_pytest_verbosity(monkeypatch):
    """Remove a pytest verbosity flag even when it is the final argument."""
    monkeypatch.setitem(sys.modules, "pytest", object())

    result = _sanitize_sys_argv_for_kit(["test_script.py", "--capture=no", "-vv"])

    assert result == ["test_script.py"]


def test_sanitize_sys_argv_preserves_user_verbosity_outside_pytest(monkeypatch):
    """Preserve application verbosity flags when pytest is not running."""
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    argv = ["script.py", "-v"]

    result = _sanitize_sys_argv_for_kit(argv)

    assert result is argv


def test_sanitize_sys_argv_removes_pytest_marker_pair(monkeypatch):
    """Remove a pytest marker option together with its expression."""
    monkeypatch.setitem(sys.modules, "pytest", object())

    result = _sanitize_sys_argv_for_kit(["test_script.py", "-m", "not isaacsim_ci", "--keep"])

    assert result == ["test_script.py", "--keep"]


def _resolve_devices_and_kit_args(launcher_args: dict, monkeypatch, *, xr: bool = False) -> tuple[dict, list[str]]:
    """Resolve device settings and Kit arguments without constructing an ``AppLauncher``.

    ``_resolve_kit_args`` extends ``sys.argv``, so the caller's argv is isolated.
    """
    monkeypatch.setattr(sys, "argv", ["script.py"])
    launcher = AppLauncher.__new__(AppLauncher)
    launcher.device_id = 0
    launcher._deferred_cuda_device_id = None
    launcher._xr = xr
    AppLauncher._resolve_device_settings(launcher, launcher_args)
    AppLauncher._resolve_kit_args(launcher, launcher_args)
    return launcher_args, launcher._kit_args


@pytest.mark.parametrize(
    ("launcher_args", "expected_renderer_args"),
    [
        pytest.param(
            {"device": "cuda:1", "multi_gpu": False},
            ["--/renderer/multiGpu/activeCudaGpus=1,"],
            id="single-gpu",
        ),
        pytest.param(
            {"device": "cuda:1", "multi_gpu": False, "kit_args": "--/renderer/multiGpu/activeCudaGpus=3,"},
            ["--/renderer/multiGpu/activeCudaGpus=3,"],
            id="explicit-kit-arg",
        ),
        pytest.param({"device": "cuda:1"}, [], id="multi-gpu"),
    ],
)
def test_devices_selected_by_cuda_index(launcher_args, expected_renderer_args, monkeypatch):
    """Select physics and single-GPU rendering devices by CUDA index."""
    args, kit_args = _resolve_devices_and_kit_args(launcher_args, monkeypatch)

    renderer_args = [arg for arg in kit_args if arg.startswith("--/renderer/multiGpu/activeCudaGpus=")]
    assert renderer_args == expected_renderer_args
    assert args["physics_gpu"] == 1
    assert "active_gpu" not in args


@pytest.mark.parametrize(
    ("launcher_args", "expected_renderer_args", "expected_physics_gpu"),
    [
        pytest.param(
            {"device": "cuda:1", "device_explicit": True},
            ["--/renderer/multiGpu/activeCudaGpus=1,"],
            1,
            id="xr-explicit-cuda-device",
        ),
        pytest.param({}, [], 0, id="xr-default-cpu-device"),
    ],
)
def test_xr_pins_the_renderer_only_for_a_cuda_device(
    launcher_args, expected_renderer_args, expected_physics_gpu, monkeypatch
):
    """Pin the renderer under XR when a CUDA device is selected, and only then.

    XR streams one stereo swapchain that the CloudXR compositor imports, so the renderer and
    the compositor have to agree on a device. A bare ``--xr`` resolves to ``cpu``, where there
    is no simulation GPU to align to, so Kit keeps its own choice -- forcing CUDA 0 there would
    break hosts whose display is not attached to GPU 0.
    """
    args, kit_args = _resolve_devices_and_kit_args(launcher_args, monkeypatch, xr=True)

    renderer_args = [arg for arg in kit_args if arg.startswith("--/renderer/multiGpu/activeCudaGpus=")]
    assert renderer_args == expected_renderer_args
    assert args["physics_gpu"] == expected_physics_gpu


@pytest.mark.parametrize(
    ("launcher_state", "expected_enabled"),
    [
        pytest.param({}, False, id="headless-training"),
        pytest.param({"_cfg_has_kit_visualizer": True}, True, id="config-kit-visualizer"),
        pytest.param(
            {"_cli_visualizer_explicit": True, "_cli_visualizer_types": ["kit"]},
            True,
            id="cli-kit-visualizer",
        ),
        pytest.param({"_render_viewport": True}, True, id="viewport"),
        pytest.param(
            {
                "_cfg_has_kit_visualizer": True,
                "_cli_visualizer_explicit": True,
                "_cli_visualizer_types": ["rerun"],
            },
            False,
            id="cli-non-kit-overrides-config",
        ),
        pytest.param({"_video_enabled": True}, True, id="video"),
        pytest.param({"_livestream": 1}, True, id="livestream"),
        pytest.param({"_xr": True}, True, id="xr"),
    ],
)
def test_spectator_view_follows_visual_output_intent(launcher_state, expected_enabled, monkeypatch):
    """Enable all-partitions spectator mode only for visual output paths."""
    monkeypatch.setattr(sys, "argv", ["script.py"])
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._cli_visualizer_explicit = False
    launcher._cli_visualizer_types = []
    launcher._cfg_has_kit_visualizer = False
    launcher._render_viewport = False
    launcher._video_enabled = False
    launcher._livestream = 0
    launcher._xr = False
    launcher.device = "cpu"
    for name, value in launcher_state.items():
        setattr(launcher, name, value)

    launcher._resolve_kit_args({})

    spectator_arg = f"--{ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING}=true"
    assert (spectator_arg in launcher._kit_args) is expected_enabled
    # The pinned Kit ignores the spectator setting, so a spectator launch must also stop authoring
    # partitions; otherwise the untokened viewport and XR views lose the partitioned environments.
    assert renderers.isaac_rtx_per_env_scene_partition_enabled() is not expected_enabled


def test_explicit_spectator_setting_overrides_visualizer_default(monkeypatch):
    """Preserve an explicit Kit setting when a Kit visualizer is requested."""
    monkeypatch.setattr(sys, "argv", ["script.py"])
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._cli_visualizer_explicit = True
    launcher._cli_visualizer_types = ["kit"]
    launcher._xr = False
    launcher.device = "cpu"
    explicit_arg = f"--{ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING}=false"

    launcher._resolve_kit_args({"kit_args": explicit_arg})

    assert explicit_arg in launcher._kit_args
    assert f"--{ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING}=true" not in launcher._kit_args


def test_explicit_scene_partition_env_var_survives_spectator_launch(monkeypatch):
    """An explicit partitioning opt-in should survive a spectator launch."""
    monkeypatch.setattr(sys, "argv", ["script.py"])
    monkeypatch.setenv("ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION", "1")
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._cli_visualizer_explicit = True
    launcher._cli_visualizer_types = ["kit"]

    launcher._resolve_kit_args({})

    assert renderers.isaac_rtx_per_env_scene_partition_enabled() is True
