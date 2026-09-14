# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ``Isaac-RenderBenchmark-Franka-Cabinet`` task's registration and presets.

No Kit/GPU required: these only load and resolve the config through the registry, the same way
:mod:`scripts.benchmarks.benchmark_renderer` selects a preset before launching a run.
"""

from functools import partial
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from isaaclab_newton.physics import NewtonCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.physics import PhysxAutoCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.benchmark.render_benchmark import render_benchmark_env_cfg
from isaaclab_tasks.benchmark.render_benchmark.render_benchmark_env import RenderBenchmarkEnv
from isaaclab_tasks.utils.hydra import collect_presets, resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_TASK = "Isaac-RenderBenchmark-Franka-Cabinet"


def _load_cfg():
    return load_cfg_from_registry(_TASK, "env_cfg_entry_point")


def test_task_registered_with_direct_entry_point():
    spec = gym.spec(_TASK)

    assert spec.kwargs["env_cfg_entry_point"] == (
        "isaaclab_tasks.benchmark.render_benchmark.render_benchmark_env_cfg:RenderBenchmarkFrankaCabinetEnvCfg"
    )
    assert spec.disable_env_checker is True


def test_default_scene_and_articulations():
    cfg = _load_cfg()

    assert cfg.decimation == 2
    assert cfg.episode_length_s == pytest.approx(60.0)
    assert cfg.scene.num_envs == 4
    assert set(cfg.articulations) == {"robot", "cabinet"}
    assert cfg.joint_animation_amplitude == pytest.approx(0.4)
    assert cfg.benchmark_mode == "render"


def test_benchmark_mode_read_from_environment(monkeypatch):
    """``BENCHMARK_MODE`` is how ``benchmark_renderer.py`` selects what a sweep measures."""
    monkeypatch.setenv("BENCHMARK_MODE", "physics_render")

    assert render_benchmark_env_cfg._read_benchmark_mode() == "physics_render"


def test_benchmark_mode_rejects_unknown_value(monkeypatch):
    """A typo must fail at launch rather than silently benchmark the default for a whole sweep."""
    monkeypatch.setenv("BENCHMARK_MODE", "physics-only")

    with pytest.raises(ValueError, match="physics-only"):
        render_benchmark_env_cfg._read_benchmark_mode()


class _RecordingArticulation:
    """Articulation stub that appends the drive calls it receives to a shared event log."""

    def __init__(self, events: list[str]):
        self._events = events
        self.position_writes: list[torch.Tensor] = []
        self.velocity_writes: list[torch.Tensor] = []
        self.actuator_targets: list[torch.Tensor] = []
        default_joint_pos = SimpleNamespace(torch=torch.zeros(1, 2))
        soft_limits = SimpleNamespace(torch=torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]]))
        self.data = SimpleNamespace(default_joint_pos=default_joint_pos, soft_joint_pos_limits=soft_limits)
        self.actuators = SimpleNamespace(target_command=SimpleNamespace(set_position_index=self._set_target))

    def _set_target(self, value):
        self._events.append("target")
        self.actuator_targets.append(value)

    def write_joint_position_to_sim_index(self, position):
        self._events.append("write_position")
        self.position_writes.append(position)

    def write_joint_velocity_to_sim_index(self, velocity):
        self._events.append("write_velocity")
        self.velocity_writes.append(velocity)


class _RecordingCameraData:
    """Camera data stub that records the lazy read which drives the render."""

    def __init__(self, events: list[str]):
        self._events = events

    @property
    def output(self) -> dict:
        self._events.append("render")
        return {}


def _fake_env(mode: str, articulation, events: list[str]) -> SimpleNamespace:
    """Build the minimum ``RenderBenchmarkEnv`` surface the animation and observation paths touch."""
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            benchmark_mode=mode,
            joint_animation_amplitude=0.4,
            joint_animation_freq_hz=0.35,
            decimation=2,
            write_image_to_file=False,
            sim=SimpleNamespace(dt=1.0 / 120.0),
        ),
        sim=SimpleNamespace(forward=lambda: events.append("forward")),
        num_envs=1,
        device="cpu",
        _anim_time=0.0,
        _anim_phases={"robot": torch.zeros(1, 2)},
        _articulations={"robot": articulation},
        _tiled_camera=SimpleNamespace(data=_RecordingCameraData(events)),
    )
    # The hooks under test call these on ``self``; bind the real implementations to the stub so
    # the test exercises them rather than a mock of them.
    for name in ("_animation_targets", "_request_joint_targets", "_pose_joints_directly"):
        setattr(env, name, partial(getattr(RenderBenchmarkEnv, name), env))
    return env


def _run_one_step(mode: str) -> tuple[list[str], _RecordingArticulation]:
    """Drive one environment step's animation hooks, with a marker where physics would run."""
    events: list[str] = []
    articulation = _RecordingArticulation(events)
    env = _fake_env(mode, articulation, events)

    RenderBenchmarkEnv._pre_physics_step(env, actions=None)
    events.append("physics")
    RenderBenchmarkEnv._get_observations(env)

    return events, articulation


def test_render_mode_poses_joints_after_physics_and_before_the_render():
    """The pose must outlive the physics step, or the renderer is timed on the solver's output.

    Writing it before physics leaves the still-active drives, gravity and joint limits free to
    move the joints off it during the step that follows.
    """
    events, articulation = _run_one_step("render")

    assert events == ["physics", "write_position", "write_velocity", "forward", "render"]
    # Nothing is asked to track a target, so the solver does no actuation work for this pose.
    assert articulation.actuator_targets == []
    assert torch.equal(articulation.velocity_writes[0], torch.zeros(1, 2))


def test_physics_render_mode_requests_targets_before_physics():
    """Actuated runs must hand the solver its target before the step, and not overwrite the result."""
    events, articulation = _run_one_step("physics_render")

    assert events == ["target", "physics", "render"]
    assert articulation.position_writes == []
    assert articulation.velocity_writes == []


@pytest.mark.parametrize(
    ("presets", "expected_type"),
    [
        ((), NewtonCfg),
        (("newton_mjwarp",), NewtonCfg),
        (("physx",), PhysxAutoCfg),
        (("isaacsim_physx",), PhysxCfg),
    ],
)
def test_physics_presets_resolve_to_expected_backend(presets, expected_type):
    cfg = resolve_presets(_load_cfg(), selected=presets)

    assert isinstance(cfg.sim.physics, expected_type)


_CAMERA_DATA_TYPE_PRESETS = [
    ("default", ["rgb"]),
    ("rgb", ["rgb"]),
    ("albedo", ["albedo"]),
    ("depth", ["depth"]),
    ("simple_shading_constant_diffuse", ["simple_shading_constant_diffuse"]),
    ("simple_shading_diffuse_mdl", ["simple_shading_diffuse_mdl"]),
    ("simple_shading_full_mdl", ["simple_shading_full_mdl"]),
]


@pytest.fixture(scope="module")
def render_benchmark_presets():
    """Collect every preset once, before any preset selection mutates the config in place."""
    return collect_presets(_load_cfg())


@pytest.mark.parametrize("preset_name,expected_data_types", _CAMERA_DATA_TYPE_PRESETS)
def test_camera_presets_resolve_to_expected_data_types(render_benchmark_presets, preset_name, expected_data_types):
    camera_presets = render_benchmark_presets["tiled_camera"]
    assert preset_name in camera_presets, f"Preset '{preset_name}' not found in tiled_camera presets"
    resolved = camera_presets[preset_name]

    assert resolved.data_types == expected_data_types
    assert resolved.width > 0
    assert resolved.height > 0
    assert isinstance(resolved.renderer_cfg.newton_renderer, NewtonWarpRendererCfg)


def test_camera_resolution_defaults_to_256():
    """``BENCHMARK_RENDER_RESOLUTION`` is read once at import, so its effect is process-scoped;
    this documents the default a fresh process gets when the env var is unset."""
    cfg = resolve_presets(_load_cfg(), selected=())

    assert cfg.tiled_camera.width == 256
    assert cfg.tiled_camera.height == 256


def _fake_env_for_ground_cfg(ground_size: tuple[float, float], env_spacing: float) -> SimpleNamespace:
    return SimpleNamespace(
        cfg=SimpleNamespace(
            ground_size=ground_size, ground_thickness=0.1, ground_color=(0.5, 0.5, 0.5), ground_top_z=0.0
        ),
        scene=SimpleNamespace(cfg=SimpleNamespace(env_spacing=env_spacing)),
    )


def test_ground_cfg_clamps_size_to_env_spacing():
    """An oversized ground tile must be clamped, or cloned environments' grounds overlap."""
    fake_env = _fake_env_for_ground_cfg(ground_size=(50.0, 50.0), env_spacing=3.0)

    ground_cfg = RenderBenchmarkEnv._ground_cfg(fake_env)

    assert ground_cfg.spawn.size[:2] == (3.0, 3.0)


def test_ground_cfg_keeps_size_smaller_than_env_spacing():
    """A tile already smaller than the env spacing must not be inflated."""
    fake_env = _fake_env_for_ground_cfg(ground_size=(2.0, 1.5), env_spacing=3.0)

    ground_cfg = RenderBenchmarkEnv._ground_cfg(fake_env)

    assert ground_cfg.spawn.size[:2] == (2.0, 1.5)
