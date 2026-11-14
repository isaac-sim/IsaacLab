# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared scene selection and held-tool behavior of the rigid coffee benchmark."""

from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock

import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.renderers import OVRTXRendererCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.benchmark.coffee_kitchen_camera.coffee_kitchen_camera_env import CoffeeKitchenCameraEnv
from isaaclab_tasks.benchmark.render_benchmark.render_benchmark_env import RenderBenchmarkEnv
from isaaclab_tasks.contrib.rizon_sharpa_coffee import demo_env_cfg
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

pytestmark = [pytest.mark.unit, pytest.mark.kitless, pytest.mark.windows_ci]

_TASK = "Isaac-RizonSharpa-Coffee-Kitchen-Camera-Benchmark"


def test_benchmark_builds_rigid_scene_without_material_sampling(monkeypatch) -> None:
    """Resolving the real task needs no particle seed grid or coupled solver."""

    def unexpected_material(*args, **kwargs):
        raise AssertionError("A rigid camera benchmark must not sample coffee or water.")

    monkeypatch.setattr(demo_env_cfg, "_particle_cfg", unexpected_material)
    cfg = resolve_presets(load_cfg_from_registry(_TASK, "env_cfg_entry_point"), selected=())
    assert isinstance(cfg.sim.physics, NewtonCfg)
    assert isinstance(cfg.sim.physics.solver_cfg, MJWarpSolverCfg)
    assert cfg.scene.coffee is None and cfg.scene.water is None
    assert cfg.scene.robot is not None and cfg.scene.spoon is not None
    assert cfg.scene.kitchen.prim_path == "{ENV_REGEX_NS}/Kitchen"
    assert cfg.scene.tiled_camera.renderer_cfg.enable_textures is False
    assert cfg.benchmark_mode == "render"
    rtx = resolve_presets(load_cfg_from_registry(_TASK, "env_cfg_entry_point"), selected=("ovrtx", "depth"))
    assert isinstance(rtx.scene.tiled_camera.renderer_cfg, OVRTXRendererCfg)
    assert rtx.scene.tiled_camera.data_types == ["depth"]


def test_animation_keeps_the_grip_and_poses_the_spoon_before_rendering() -> None:
    """The inherited arm animation is deterministic, while camera reads follow the new mounted pose."""
    events = Mock()
    robot = events.robot
    default = torch.tensor([[0.0, 0.2, 0.7], [0.0, 0.2, 0.7]])
    robot.data.default_joint_pos.torch = default
    robot.data.soft_joint_pos_limits.torch = torch.tensor([[[-1.0, 1.0]] * 3] * 2)
    scene = SimpleNamespace(articulations={"robot": robot}, num_envs=2)
    camera_data = Mock()
    type(camera_data).output = PropertyMock(side_effect=lambda: events.render())
    env = object.__new__(CoffeeKitchenCameraEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(
        benchmark_mode="render",
        joint_animation_amplitude=0.1,
        joint_animation_freq_hz=0.35,
        write_image_to_file=False,
    )
    env.scene = scene
    env.sim = events.sim
    env.sim.device = "cpu"
    env._attachment = events.attachment
    env._hand_ids = [1, 2]
    env._anim_phases = {"robot": torch.zeros(2, 3)}
    env._anim_time = 0.5
    env._tiled_camera = SimpleNamespace(data=camera_data)

    RenderBenchmarkEnv._get_observations(env)
    target = robot.write_joint_position_to_sim_index.call_args.kwargs["position"]
    assert (target[:, 0] != default[:, 0]).all()
    torch.testing.assert_close(target[:, 1:], default[:, 1:])
    assert [call[0] for call in events.mock_calls] == [
        "robot.write_joint_position_to_sim_index",
        "robot.write_joint_velocity_to_sim_index",
        "sim.forward",
        "attachment.sync",
        "sim.forward",
        "render",
    ]
    first = target.clone()
    RenderBenchmarkEnv._get_observations(env)
    torch.testing.assert_close(robot.write_joint_position_to_sim_index.call_args.kwargs["position"], first)
