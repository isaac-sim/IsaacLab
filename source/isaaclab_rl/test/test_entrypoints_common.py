# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for shared reinforcement learning script utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
from isaaclab_newton.physics import NewtonCfg
from PIL import Image

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.sim import SimulationCfg

from isaaclab_rl.entrypoints import common as rl_common
from isaaclab_rl.entrypoints.common import (
    CaptureEnvSensors,
    add_common_train_args,
    create_isaaclab_env,
    enable_cameras_for_video,
    normalize_task_name,
    wrap_sensor_capture,
)


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


def test_capture_env_sensors_saves_file_outputs_on_scheduled_steps(tmp_path: Path) -> None:
    """File capture writes image grids during the active capture window."""
    rgb = torch.tensor(
        [
            [[[0, 127, 255, 9], [255, 0, 127, 9]]],
            [[[42, 42, 42, 9], [43, 43, 43, 9]]],
        ],
        dtype=torch.uint8,
    )
    env = _FakeEnv({"front/camera": _make_sensor({"rgb": rgb})})
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

    saved_paths = sorted(tmp_path.rglob("*.png"))
    relative_paths = [path.relative_to(tmp_path).as_posix() for path in saved_paths]
    assert relative_paths == [
        "front_camera/rgb/episode_00001_step_00000000.png",
        "front_camera/rgb/episode_00001_step_00000001.png",
        "front_camera/rgb/episode_00001_step_00000003.png",
    ]
    for path in saved_paths:
        with Image.open(path) as image:
            np.testing.assert_array_equal(np.asarray(image), rgb[0].numpy())


def test_capture_env_sensors_accepts_proxyarray_and_skips_missing_outputs(tmp_path: Path) -> None:
    """File capture limits ProxyArray batches and ignores missing outputs."""
    import warp as wp

    from isaaclab.utils.warp import ProxyArray

    pixels = torch.full((3, 2, 2, 4), 127, dtype=torch.uint8)
    image_buffer = ProxyArray(wp.from_torch(pixels))
    env = _FakeEnv({"camera": _make_sensor({"rgb": image_buffer, "depth": None})})
    wrapper = _make_capture_wrapper(tmp_path, env=env, capture_num_envs=1)
    wrapper.reset()

    paths = sorted(tmp_path.rglob("*.png"))
    assert len(paths) == 1
    assert paths[0].parent.name == "rgb"
    with Image.open(paths[0]) as image:
        np.testing.assert_array_equal(np.asarray(image), pixels[0].numpy())


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


def test_common_train_args_register_frontend_with_torch_default() -> None:
    """Every RL library CLI exposes ``--frontend`` and defaults to the torch runtime."""
    parser = argparse.ArgumentParser()
    add_common_train_args(parser, agent_default=None, agent_help="", include_agent=False)

    assert parser.parse_args([]).frontend == "torch"
    assert parser.parse_args(["--frontend", "warp"]).frontend == "warp"
    with pytest.raises(SystemExit):
        parser.parse_args(["--frontend", "tensorflow"])


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("Isaac-Task", "Isaac-Task"),
        ("my_module:Isaac-Task-v12", "Isaac-Task-v12"),
        ("my_module:Isaac-Task-Play-v12", "Isaac-Task-Play-v12"),
        ("Isaac-Playground-v0", "Isaac-Playground-v0"),
    ],
)
def test_normalize_task_name_removes_namespace(task: str, expected: str) -> None:
    """Checkpoint lookup removes only the optional namespace."""
    assert normalize_task_name(task) == expected


def test_create_isaaclab_env_uses_registered_torch_env_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared factory preserves the existing Gym path when no frontend is selected."""
    expected_env = object()
    env_cfg = object()
    calls: list[tuple[Any, ...]] = []

    def fake_make(task: str, **kwargs: Any) -> Any:
        calls.append((task, kwargs))
        return expected_env

    monkeypatch.setattr(rl_common.gym, "make", fake_make)
    args_cli = argparse.Namespace(video=False, frontend="torch")

    env = create_isaaclab_env("Isaac-Test", env_cfg, args_cli, convert_marl_to_single_agent=False)

    assert env is expected_env
    assert len(calls) == 1
    assert calls[0][0] == "Isaac-Test"
    assert calls[0][1]["cfg"] is env_cfg
    assert "render_mode" not in calls[0][1]


def test_create_isaaclab_env_uses_selected_warp_frontend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared factory delegates Warp selection to the experimental frontend."""
    import isaaclab_experimental.envs.frontend as frontend_module

    expected_env = object()
    env_cfg = object()
    calls: list[tuple[Any, ...]] = []

    def fake_build_env(cfg: Any, task: str, **kwargs: Any) -> Any:
        calls.append((cfg, task, kwargs))
        return expected_env

    monkeypatch.setattr(frontend_module.WarpFrontend, "build_env", fake_build_env)
    args_cli = argparse.Namespace(video=True, frontend="warp")

    env = create_isaaclab_env("Isaac-Test", env_cfg, args_cli, convert_marl_to_single_agent=False)

    assert env is expected_env
    assert calls == [(env_cfg, "Isaac-Test", {})]


class _RecordingScreen:
    """Record summary fields without drawing a loading screen."""

    def __init__(self) -> None:
        self.fields: dict[str, str] = {}

    def summary(self, title: str, fields: dict[str, str]) -> None:
        self.fields = fields


@pytest.mark.parametrize(
    "selectors, expected_physics, expected_renderer",
    [
        (["physics=ovphysx", "renderer=rtx"], "ovphysx", "rtx (ovrtx)"),
        (["physics=isaacsim_physx", "renderer=rtx"], "isaacsim_physx", "rtx (isaacsim_rtx)"),
        (["physics=physx", "renderer=rtx"], "physx (ovphysx)", "rtx (ovrtx)"),
        ([], "newton_mjwarp", "newton_renderer"),
        (["physics=physx", "presets=depth"], "physx (ovphysx)", "newton_renderer"),
    ],
)
def test_run_summary_reports_concrete_backends(
    selectors: list[str],
    expected_physics: str,
    expected_renderer: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The summary reports concrete backends and launcher-owned automatic choices."""
    import isaaclab_tasks.registry  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    task = "Isaac-Cartpole-Camera-Direct"
    monkeypatch.setattr(rl_common.sys, "argv", ["train.py", *selectors])
    env_cfg, _ = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    screen = _RecordingScreen()
    args_cli = argparse.Namespace(task=task, device=None, num_envs=None, visualizer=None)

    rl_common.show_run_summary(screen, args_cli, env_cfg, library="rsl_rl", action="train")

    assert screen.fields["Physics"] == expected_physics
    assert screen.fields["Renderer"] == expected_renderer
    assert "Presets" not in screen.fields


def test_apply_env_overrides_records_the_deterministic_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """The deterministic option is recorded in the physics configuration."""
    import isaaclab_tasks.registry  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    monkeypatch.setattr(rl_common.sys, "argv", ["train.py"])
    env_cfg, _ = resolve_task_config("Isaac-Cartpole-Camera", "rl_games_cfg_entry_point")
    assert env_cfg.sim.physics.deterministic is False

    args_cli = argparse.Namespace(num_envs=None, device=None, deterministic=True)
    rl_common.apply_env_overrides(args_cli, env_cfg, apply_device=False)

    assert env_cfg.sim.physics.deterministic is True
    assert env_cfg.sim.physics.deterministic_mode == "not_guaranteed"
    assert env_cfg.sim.physics.solver_cfg.disable_sensors is False


def test_apply_env_overrides_leaves_physics_alone_without_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``--deterministic`` the physics config is untouched."""
    import isaaclab_tasks.registry  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    monkeypatch.setattr(rl_common.sys, "argv", ["train.py"])
    env_cfg, _ = resolve_task_config("Isaac-Cartpole-Camera", "rl_games_cfg_entry_point")

    args_cli = argparse.Namespace(num_envs=None, device=None, deterministic=False)
    rl_common.apply_env_overrides(args_cli, env_cfg, apply_device=False)

    assert env_cfg.sim.physics.deterministic is False


@pytest.mark.parametrize(
    ("already_set", "configured_mode", "expected"),
    [
        ("NOT_GUARANTEED", None, "RUN_TO_RUN"),
        ("NOT_GUARANTEED", "not_guaranteed", "RUN_TO_RUN"),
        ("NOT_GUARANTEED", "run_to_run", "RUN_TO_RUN"),
        ("NOT_GUARANTEED", "gpu_to_gpu", "GPU_TO_GPU"),
        ("RUN_TO_RUN", "gpu_to_gpu", "GPU_TO_GPU"),
        ("GPU_TO_GPU", None, "GPU_TO_GPU"),
        ("GPU_TO_GPU", "run_to_run", "GPU_TO_GPU"),
    ],
)
def test_apply_env_overrides_raises_warp_determinism_to_the_configured_mode(
    already_set: str, configured_mode: str | None, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warp determinism is raised to the strongest requested mode."""
    import warp as wp

    monkeypatch.setattr(wp.config, "deterministic", getattr(wp.DeterministicMode, already_set))
    physics = PhysicsCfg() if configured_mode is None else NewtonCfg(deterministic_mode=configured_mode)
    env_cfg = ManagerBasedRLEnvCfg(sim=SimulationCfg(physics=physics))

    args_cli = argparse.Namespace(num_envs=None, device=None, deterministic=True)
    rl_common.apply_env_overrides(args_cli, env_cfg, apply_device=False)

    assert wp.config.deterministic == getattr(wp.DeterministicMode, expected)


def test_apply_env_overrides_leaves_warp_alone_without_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warp determinism is unchanged when the option is disabled."""
    import warp as wp

    monkeypatch.setattr(wp.config, "deterministic", wp.DeterministicMode.NOT_GUARANTEED)
    env_cfg = ManagerBasedRLEnvCfg(sim=SimulationCfg(physics=NewtonCfg()))

    args_cli = argparse.Namespace(num_envs=None, device=None, deterministic=False)
    rl_common.apply_env_overrides(args_cli, env_cfg, apply_device=False)

    assert wp.config.deterministic == wp.DeterministicMode.NOT_GUARANTEED


def test_apply_env_overrides_records_the_request_for_unknown_backend() -> None:
    """Unknown physics backends receive the backend-agnostic request."""
    physics = SimpleNamespace(deterministic=False)
    env_cfg = SimpleNamespace(sim=SimpleNamespace(physics=physics))

    args_cli = argparse.Namespace(num_envs=None, device=None, deterministic=True)
    rl_common.apply_env_overrides(args_cli, env_cfg, apply_device=False)

    assert physics.deterministic is True


def test_apply_env_overrides_tolerates_a_config_without_physics() -> None:
    """A configuration without a physics backend is accepted."""
    env_cfg = SimpleNamespace(sim=SimpleNamespace(physics=None))

    args_cli = argparse.Namespace(num_envs=None, device=None, deterministic=True)
    rl_common.apply_env_overrides(args_cli, env_cfg, apply_device=False)

    assert env_cfg.sim.physics is None
