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
from unittest.mock import Mock, PropertyMock

import gymnasium as gym
import pytest
import torch
from isaaclab_newton.physics import NewtonCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from isaaclab.envs import DirectRLEnv
from isaaclab.physics import PhysxAutoCfg
from isaaclab.sim.schemas import MassCfg, UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg

import isaaclab_tasks  # noqa: F401
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
    assert cfg.scene.robot.prim_path == "{ENV_REGEX_NS}/Robot"
    assert cfg.scene.cabinet.prim_path == "{ENV_REGEX_NS}/Cabinet"
    assert cfg.joint_animation_amplitude == pytest.approx(0.4)
    assert cfg.benchmark_mode == "render"


@pytest.mark.parametrize(
    ("mode", "lazy_sensor_update", "renderer_type", "pumps_app_update", "error"),
    [
        ("render", False, "newton_warp", False, "lazy_sensor_update=True"),
        ("physics_render", False, "newton_warp", False, None),
        ("render", True, "isaac_rtx", True, "--visualizer none"),
        ("physics_render", True, "isaac_rtx", True, None),
        ("render", True, "ovrtx", True, None),
        ("render", True, "isaac_rtx", False, None),
    ],
)
def test_render_mode_rejects_rendering_before_direct_pose(
    monkeypatch, mode, lazy_sensor_update, renderer_type, pumps_app_update, error
):
    """Direct posing requires camera rendering to follow the joint writes."""
    cfg = _load_cfg().replace(benchmark_mode=mode)
    cfg.scene.lazy_sensor_update = lazy_sensor_update

    def initialize(self, *args, **kwargs):
        self.scene = {"tiled_camera": None}
        self.sim = SimpleNamespace(
            render_context=SimpleNamespace(renderer_types=(renderer_type,)),
            visualizers=[SimpleNamespace(pumps_app_update=lambda: pumps_app_update)],
        )

    close = Mock()
    monkeypatch.setattr(DirectRLEnv, "__init__", initialize)
    monkeypatch.setattr(DirectRLEnv, "close", close)

    if error:
        with pytest.raises(ValueError, match=error):
            RenderBenchmarkEnv(cfg)
    else:
        RenderBenchmarkEnv(cfg)
    if error == "--visualizer none":
        close.assert_called_once()
    else:
        close.assert_not_called()


@pytest.mark.parametrize("mode", ["render", "physics_render"])
def test_benchmark_mode_orders_joint_updates_and_rendering(mode):
    """Direct poses follow physics; actuator targets precede it, and both render last."""
    events = Mock()
    articulation = events.articulation
    articulation.data.default_joint_pos.torch = torch.zeros(1, 2)
    articulation.data.soft_joint_pos_limits.torch = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])
    camera_data = Mock()
    type(camera_data).output = PropertyMock(side_effect=lambda: events.render())
    env = SimpleNamespace(
        cfg=_load_cfg().replace(benchmark_mode=mode, write_image_to_file=False),
        sim=events.sim,
        num_envs=1,
        device="cpu",
        _anim_time=0.0,
        _anim_phases={"robot": torch.zeros(1, 2)},
        scene=SimpleNamespace(articulations={"robot": articulation}),
        _tiled_camera=SimpleNamespace(data=camera_data),
    )
    for name in ("_animation_targets", "_request_joint_targets", "_pose_joints_directly"):
        setattr(env, name, partial(getattr(RenderBenchmarkEnv, name), env))

    RenderBenchmarkEnv._pre_physics_step(env, actions=None)
    events.physics()
    RenderBenchmarkEnv._get_observations(env)

    if mode == "render":
        assert [entry[0] for entry in events.mock_calls] == [
            "physics",
            "articulation.write_joint_position_to_sim_index",
            "articulation.write_joint_velocity_to_sim_index",
            "sim.forward",
            "sim.render_context.reset_scene_state_cadence",
            "render",
        ]
        velocity = articulation.write_joint_velocity_to_sim_index.call_args.kwargs["velocity"]
        assert torch.equal(velocity, torch.zeros_like(velocity))
    else:
        assert [entry[0] for entry in events.mock_calls] == [
            "articulation.actuators.target_command.set_position_index",
            "physics",
            "render",
        ]


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
    camera_presets = render_benchmark_presets["scene.tiled_camera"]
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

    assert cfg.scene.tiled_camera.width == 256
    assert cfg.scene.tiled_camera.height == 256


def test_ground_tile_matches_default_env_spacing():
    cfg = _load_cfg()

    assert cfg.scene.ground.spawn.size[:2] == (cfg.scene.env_spacing, cfg.scene.env_spacing)
    rigid_props = cfg.scene.ground.spawn.rigid_props
    assert next(f for f in rigid_props if isinstance(f, UsdPhysicsRigidBodyCfg)).kinematic_enabled is True
    assert next(f for f in rigid_props if isinstance(f, PhysxRigidBodyCfg)).disable_gravity is True
    assert isinstance(cfg.scene.ground.spawn.mass_props, MassCfg)
    assert cfg.scene.ground.spawn.mass_props.mass == pytest.approx(1.0)
    assert isinstance(cfg.scene.ground.spawn.collision_props, UsdPhysicsCollisionCfg)
