# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for shared reinforcement learning script utilities."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import gymnasium as gym
import pytest
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_rl_common_module() -> ModuleType:
    module_path = _repo_root() / "scripts" / "reinforcement_learning" / "common.py"
    spec = importlib.util.spec_from_file_location("isaaclab_test_reinforcement_learning_common", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load reinforcement learning common module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_rl_common = _load_rl_common_module()
CaptureEnvSensors: Any = getattr(_rl_common, "CaptureEnvSensors")
add_common_train_args: Any = getattr(_rl_common, "add_common_train_args")
enable_cameras_for_video: Any = getattr(_rl_common, "enable_cameras_for_video")
dispatch_library_entrypoint: Any = getattr(_rl_common, "dispatch_library_entrypoint")
wrap_sensor_capture: Any = getattr(_rl_common, "wrap_sensor_capture")


class _FakeEnv(gym.Env):
    """Minimal Gymnasium env exposing an IsaacLab-style scene sensor mapping."""

    def __init__(self, sensors: dict[str, Any] | None = None) -> None:
        self.scene = SimpleNamespace(sensors=sensors or {})
        self.closed = False

    def reset(self, **kwargs: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return {"obs": torch.zeros(1)}, {}

    def step(self, action: Any) -> tuple[dict[str, torch.Tensor], float, bool, bool, dict[str, Any]]:
        return {"obs": torch.ones(1)}, 0.0, False, False, {}

    def close(self) -> None:
        self.closed = True


def _make_sensor(output: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(data=SimpleNamespace(output=output))


def _make_capture_wrapper(tmp_path: Path, **kwargs: Any) -> Any:
    defaults = {
        "env": _FakeEnv(),
        "output_dir": str(tmp_path),
        "frame_count": 1,
        "capture_num_envs": 1,
        "interval": 1,
        "output_format": "file",
    }
    defaults.update(kwargs)
    return CaptureEnvSensors(**defaults)


def test_capture_env_sensors_saves_file_outputs_on_scheduled_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """File capture writes image grids during the active capture window."""
    rgb = torch.tensor(
        [
            [[[0, 127, 255, 9], [255, 0, 127, 9]]],
            [[[42, 42, 42, 9], [43, 43, 43, 9]]],
        ],
        dtype=torch.uint8,
    )
    env = _FakeEnv({"front/camera": _make_sensor({"rgb": rgb})})
    saved_images: list[Any] = []
    saved_paths: list[Path] = []

    class _FakeImage:
        def __init__(self, image: Any) -> None:
            self.image = image

        def save(self, path: str) -> None:
            saved_images.append(self.image.copy())
            saved_paths.append(Path(path))

    monkeypatch.setattr(_rl_common.Image, "fromarray", _FakeImage)
    wrapper = _make_capture_wrapper(
        tmp_path,
        env=env,
        frame_count=2,
        capture_num_envs=1,
        interval=3,
    )

    wrapper.reset()
    wrapper.step(None)
    wrapper.step(None)
    wrapper.step(None)

    relative_paths = [path.relative_to(tmp_path).as_posix() for path in saved_paths]
    assert relative_paths == [
        "front_camera/rgb/episode_00001_step_00000000.png",
        "front_camera/rgb/episode_00001_step_00000001.png",
        "front_camera/rgb/episode_00001_step_00000003.png",
    ]
    assert all(image.shape == (1, 2, 4) for image in saved_images)
    assert all((image == rgb[0].numpy()).all() for image in saved_images)


def test_capture_env_sensors_accepts_proxyarray_torch_buffers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ProxyArray-style buffers are read through their ``.torch`` accessor."""
    image_buffer = SimpleNamespace(torch=torch.ones((3, 2, 2, 4), dtype=torch.float32))
    env = _FakeEnv({"camera": _make_sensor({"rgb": image_buffer})})
    captured_tensors: list[torch.Tensor] = []

    def fake_normalize(tensor: torch.Tensor, data_type: str) -> torch.Tensor:
        captured_tensors.append(tensor.clone())
        return tensor

    monkeypatch.setattr(_rl_common, "normalize_camera_output_for_display", fake_normalize)
    monkeypatch.setattr(_rl_common, "make_camera_output_grid", lambda images: torch.zeros((4, 1, 1)))
    wrapper = _make_capture_wrapper(tmp_path, env=env, capture_num_envs=2)

    wrapper.reset()

    assert len(captured_tensors) == 1
    assert captured_tensors[0].shape == (2, 2, 2, 4)


def test_capture_env_sensors_skips_none_outputs(tmp_path: Path) -> None:
    """Missing sensor outputs are skipped instead of being written."""
    env = _FakeEnv({"camera": _make_sensor({"rgb": None})})
    wrapper = _make_capture_wrapper(tmp_path, env=env)

    wrapper.reset()

    assert not any(tmp_path.rglob("*.png"))


def test_capture_env_sensors_rejects_unknown_output_format(tmp_path: Path) -> None:
    """Only tensorboard and file output formats are supported."""
    with pytest.raises(ValueError, match="Unsupported sensor capture output format"):
        _make_capture_wrapper(tmp_path, output_format="invalid")


def test_wrap_sensor_capture_uses_training_sensor_frame_directory(tmp_path: Path) -> None:
    """The train helper wraps the env with the configured sensor capture output directory."""
    env = _FakeEnv()
    args_cli = argparse.Namespace(
        capture_env_sensors=2,
        capture_env_sensors_length=5,
        capture_env_sensors_interval=7,
        capture_env_sensors_format="file",
    )

    wrapped_env = wrap_sensor_capture(env, str(tmp_path), args_cli)

    assert isinstance(wrapped_env, CaptureEnvSensors)
    assert Path(wrapped_env.output_dir) == tmp_path / "sensor_frames" / "train"
    assert wrapped_env.frame_count == 5
    assert wrapped_env.capture_num_envs == 2
    assert wrapped_env.interval == 7
    assert wrapped_env.env is env


def test_wrap_sensor_capture_returns_env_when_disabled(tmp_path: Path) -> None:
    """The train helper leaves the env unwrapped when sensor capture is disabled."""
    env = _FakeEnv()
    args_cli = argparse.Namespace(capture_env_sensors=0)

    assert wrap_sensor_capture(env, str(tmp_path), args_cli) is env


def test_common_train_args_include_sensor_capture_options() -> None:
    """Common train parsers expose sensor capture CLI arguments."""
    parser = argparse.ArgumentParser()
    add_common_train_args(parser, agent_default=None, agent_help="", include_agent=False)

    args_cli = parser.parse_args(
        [
            "--capture_env_sensors",
            "3",
            "--capture_env_sensors_length",
            "4",
            "--capture_env_sensors_interval",
            "5",
            "--capture_env_sensors_format",
            "file",
        ]
    )

    assert args_cli.capture_env_sensors == 3
    assert args_cli.capture_env_sensors_length == 4
    assert args_cli.capture_env_sensors_interval == 5
    assert args_cli.capture_env_sensors_format == "file"


def test_enable_cameras_for_video_enables_cameras_for_sensor_capture() -> None:
    """Sensor capture requires camera rendering even when normal video capture is disabled."""
    args_cli = argparse.Namespace(video=False, capture_env_sensors=1, enable_cameras=False)

    enable_cameras_for_video(args_cli)

    assert args_cli.enable_cameras


def test_dispatch_library_entrypoint_shows_help_without_library(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The unified dispatcher shows its help before requiring a library selection."""
    result = dispatch_library_entrypoint(
        ["--help"],
        {"rsl_rl": tmp_path / "bench_rsl_rl.py"},
        action="bench",
        description="Benchmark training.",
        library_help="Training library to benchmark.",
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "--rl_library {rsl_rl}" in output
