# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for simulation-owned renderers and their rendering orchestration."""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

from isaaclab.renderers import render_context
from isaaclab.renderers.base_renderer import BaseRenderer
from isaaclab.renderers.render_context import RenderContext
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.sensors.camera.camera_data import CameraData
from isaaclab.sim import BackendCfg, SimulationContext

pytest.importorskip("isaaclab_physx")
pytest.importorskip("isaaclab_newton")

from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

pytestmark = [pytest.mark.integration, pytest.mark.rendering]


def _renderer(cfg):
    renderer = Mock(spec=BaseRenderer)
    renderer.visual_material_writer = None
    return renderer


@pytest.fixture
def sim():
    sim = object.__new__(SimulationContext)
    sim._backend_registry = []
    sim._render_context = RenderContext(sim._backend_registry)
    return sim


def test_renderer_registry_sharing_and_early_clone_requirements(sim):
    constructor = Mock(side_effect=_renderer)
    cfg = IsaacRtxRendererCfg(class_type=constructor, cloning_contexts=("example:CloneContext",))
    renderer = sim.get_or_create_backend(cfg)

    assert sim.get_or_create_backend(cfg) is renderer
    assert sim.get_or_create_backend(cfg.replace()) is renderer
    constructor.assert_called_once_with(cfg)
    assert constructor.call_args.args[0] is cfg
    assert sim.render_context.clone_contexts == set(cfg.cloning_contexts)
    renderer.initialize.assert_not_called()

    assert sim.get_or_create_backend(cfg.replace(semantic_filter="class:robot")) is not renderer
    assert sim.get_or_create_backend(NewtonWarpRendererCfg(class_type=_renderer)) is not renderer
    sim.get_or_create_backend(BackendCfg(class_type=lambda cfg: object()))
    assert sim.render_context.renderer_types == ("isaac_rtx", "isaac_rtx", "newton_warp")


def test_renderer_initializes_once_before_or_after_physics_ready(sim):
    cfg = RendererCfg(class_type=_renderer)
    first = sim.get_or_create_backend(cfg)
    sim.get_or_create_backend(BackendCfg(class_type=lambda cfg: object()))
    sim.render_context.ensure_initialize()
    sim.render_context.ensure_initialize()
    first.initialize.assert_called_once_with()

    second_cfg = cfg.replace(renderer_type="second")
    second = sim.get_or_create_backend(second_cfg)
    second.initialize.assert_called_once_with()
    assert sim.get_or_create_backend(second_cfg) is second
    sim.render_context.ensure_initialize()
    first.initialize.assert_called_once_with()
    second.initialize.assert_called_once_with()


def test_conflicting_global_settings_are_rejected_before_construction(sim):
    constructor = Mock(side_effect=_renderer)
    cfg = IsaacRtxRendererCfg(class_type=constructor)
    renderer = sim.get_or_create_backend(cfg)
    conflicting = cfg.replace(global_settings=cfg.global_settings.replace(enable_shadows=False))

    with pytest.raises(ValueError, match="global settings differ"):
        sim.get_or_create_backend(conflicting)

    constructor.assert_called_once_with(cfg)
    assert sim.get_or_create_backend(cfg) is renderer
    assert sim.render_context.renderer_types == ("isaac_rtx",)


@pytest.mark.parametrize("has_materials", [False, True])
def test_finalized_consumers_allow_cache_hits_but_reject_new_renderers_with_materials(sim, has_materials):
    constructor = Mock(side_effect=_renderer)
    cfg = RendererCfg(class_type=constructor)
    renderer = sim.get_or_create_backend(cfg)
    if has_materials:
        sim.render_context.register_visual_material(
            SimpleNamespace(
                channels=("roughness",),
                _material_paths=("/World/Material",),
                _shader_paths=("/World/Material/Shader",),
                _input_names={"roughness": "roughness"},
                _values={"roughness": torch.zeros(1)},
                _offsets={},
            )
        )
    sim.render_context.finalize_consumers([])

    assert sim.get_or_create_backend(cfg.replace()) is renderer
    late_cfg = cfg.replace(renderer_type="late")
    if has_materials:
        with pytest.raises(RuntimeError, match="before rendering consumers are finalized"):
            sim.get_or_create_backend(late_cfg)
        constructor.assert_called_once_with(cfg)
    else:
        assert sim.get_or_create_backend(late_cfg) is not renderer
        assert constructor.call_count == 2


def test_close_backend_removes_renderer_from_orchestration(sim):
    cfg = RendererCfg(class_type=_renderer)
    renderer = sim.get_or_create_backend(cfg)
    sim.render_context.ensure_prepare_stage(None, 4)
    sim.render_context.update_scene_state(1)

    sim.close_backend(renderer)
    renderer.close.assert_called_once_with()
    assert not sim.render_context.renderer_types
    sim.render_context.update_scene_state(2)
    renderer.update_transforms.assert_called_once_with()
    renderer.update_geometries.assert_called_once_with()
    with pytest.raises(RuntimeError, match="renderer must be registered"):
        sim.render_context.ensure_prepare_stage(None, 4)

    replacement = sim.get_or_create_backend(cfg)
    assert replacement is not renderer
    sim.render_context.ensure_prepare_stage(None, 4)
    sim.render_context.update_scene_state(2)
    replacement.prepare_stage.assert_called_once_with(None, 4)
    replacement.update_transforms.assert_called_once_with()
    sim.render_context.close()
    renderer.close.assert_called_once_with()
    replacement.close.assert_not_called()


def test_prepare_stage_is_idempotent_and_checks_env_count_until_reset(sim):
    renderer = sim.get_or_create_backend(RendererCfg(class_type=_renderer))
    sim.render_context.ensure_prepare_stage(None, 4)
    sim.render_context.ensure_prepare_stage(None, 4)
    renderer.prepare_stage.assert_called_once_with(None, 4)
    with pytest.raises(RuntimeError, match="different num_envs"):
        sim.render_context.ensure_prepare_stage(None, 8)

    sim.render_context.reset_stage_prepare_flag()
    sim.render_context.ensure_prepare_stage(None, 8)
    assert renderer.prepare_stage.call_args_list == [call(None, 4), call(None, 8)]


def test_scene_state_updates_once_per_step_until_cadence_reset(sim):
    renderer = sim.get_or_create_backend(RendererCfg(class_type=_renderer))
    for step in (1, 1, 2):
        sim.render_context.update_scene_state(step)
    assert renderer.update_transforms.call_count == renderer.update_geometries.call_count == 2

    sim.render_context.reset_scene_state_cadence()
    sim.render_context.update_scene_state(2)
    assert renderer.update_transforms.call_count == renderer.update_geometries.call_count == 3


@pytest.mark.parametrize("profile", [False, True])
def test_render_into_camera_call_order_and_profile_output(sim, monkeypatch, capsys, profile):
    """Profiling preserves call order and prints the renderer benchmark's timing format."""
    monkeypatch.setattr(render_context, "_RENDER_PROFILE_ENABLED", profile)
    renderer = sim.get_or_create_backend(RendererCfg(class_type=_renderer))
    data, camera = object(), CameraData()

    sim.render_context.render_into_camera(renderer, data, camera, physics_step_count=1)
    sim.render_context.render_into_camera(renderer, data, camera, physics_step_count=1)

    assert renderer.mock_calls == [
        call.update_transforms(),
        call.update_geometries(),
        call.render(data),
        call.read_output(data, camera),
        call.render(data),
        call.read_output(data, camera),
    ]
    timing = rf"{re.escape(render_context.RENDER_PROFILE_SCOPE)} took [\d.]+ ms"
    assert len(re.findall(timing, capsys.readouterr().out)) == (2 if profile else 0)


@pytest.mark.parametrize("fail_writer", [False, True])
def test_context_close_only_releases_writers_and_resets_bookkeeping(sim, fail_writer):
    cfg = RendererCfg(class_type=_renderer, cloning_contexts=("example:CloneContext",))
    renderer = sim.get_or_create_backend(cfg)
    context = sim.render_context
    context.ensure_initialize()
    context.ensure_prepare_stage(None, 4)
    context.update_scene_state(1)
    writers = (Mock(), Mock())
    if fail_writer:
        writers[0].close.side_effect = RuntimeError("writer failed")
    context._visual_material_writers = writers

    if fail_writer:
        with pytest.raises(RuntimeError, match=r"1 material writer\(s\) failed to close"):
            context.close()
    else:
        context.close()
    context.close()
    for writer in writers:
        writer.close.assert_called_once_with()
    renderer.close.assert_not_called()
    assert sim.get_or_create_backend(cfg) is renderer
    assert not context.clone_contexts

    context.ensure_initialize()
    context.ensure_prepare_stage(None, 8)
    context.update_scene_state(1)
    assert renderer.initialize.call_count == renderer.prepare_stage.call_count == 2
    assert renderer.update_transforms.call_count == renderer.update_geometries.call_count == 2
