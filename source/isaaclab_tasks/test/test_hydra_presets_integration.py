# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests: preset string -> resolve_task_config yields expected backends.

Tests that resolving a task with presets=... (as in the kit decision tests) produces
the expected simulation backend, renderer backend, and camera type. No Kit/GPU required.
"""

import sys

import pytest

# Register cartpole presets task so load_cfg_from_registry can find it.
pytest.importorskip("gymnasium")
pytest.importorskip("isaaclab_physx")
pytest.importorskip("isaaclab_newton")
import isaaclab_tasks.direct.cartpole  # noqa: F401  # register Isaac-Cartpole-Camera-Presets-Direct-v0

from isaaclab_newton.physics import NewtonCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_tasks.utils.hydra import parse_overrides, register_task


# Task name from isaaclab_tasks.direct.cartpole
TASK_NAME = "Isaac-Cartpole-Camera-Presets-Direct-v0"
AGENT_ENTRY = "rl_games_cfg_entry_point"

# re-use of _resolve_with_presets from source/isaaclab_tasks/test/test_preset_kit_decision.py
# resolve_task_config calls register_task and parse_overrides to resolve env and agent configs
def _resolve_with_presets(presets: str):
    """Resolve env_cfg with given presets. Modifies sys.argv temporarily."""
    old_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0], f"presets={presets}"]
        env_cfg, agent_cfg = resolve_task_config(TASK_NAME, AGENT_ENTRY)
        return env_cfg
    finally:
        sys.argv = old_argv


def _get_registered_env_and_presets():
    """Register task and return (env_cfg, agent_cfg, presets) for use in parse_overrides test."""
    env_cfg, agent_cfg, presets = register_task(TASK_NAME, AGENT_ENTRY)
    return env_cfg, agent_cfg, presets


class TestHydraPresetsIntegration:
    """Test that preset string -> resolve_task_config yields expected backends (same pattern as test_preset_kit_decision)."""

    def test_parse_overrides_global_presets(self):
        """parse_overrides extracts global presets from a presets=... override (e.g. presets=physx,isaacsim_rtx_renderer,rgb)."""
        _, _, presets = _get_registered_env_and_presets()
        global_presets, preset_sel, _, _ = parse_overrides(
            ["presets=physx,isaacsim_rtx_renderer,rgb"], presets
        )
        assert set(global_presets) == {"physx", "isaacsim_rtx_renderer", "rgb"}
        assert preset_sel == []

    def test_resolve_simulation_backend_physx(self):
        """presets=physx,... -> resolved env_cfg.sim.physics is PhysxCfg."""
        env_cfg = _resolve_with_presets("physx,isaacsim_rtx_renderer,rgb")
        physics = env_cfg.sim.physics
        assert physics is not None
        assert isinstance(physics, PhysxCfg)

    def test_resolve_simulation_backend_newton(self):
        """presets=newton,... -> resolved env_cfg.sim.physics is NewtonCfg."""
        env_cfg = _resolve_with_presets("newton,newton_renderer,rgb")
        physics = env_cfg.sim.physics
        assert physics is not None
        assert isinstance(physics, NewtonCfg)

    def test_resolve_renderer_backend_isaac_rtx(self):
        """presets=...,isaacsim_rtx_renderer,... -> resolved tiled_camera.renderer_cfg is IsaacRtxRendererCfg."""
        env_cfg = _resolve_with_presets("physx,isaacsim_rtx_renderer,rgb")
        renderer_cfg = env_cfg.tiled_camera.renderer_cfg
        assert renderer_cfg is not None
        assert isinstance(renderer_cfg, IsaacRtxRendererCfg)

    def test_resolve_renderer_backend_newton_warp(self):
        """presets=...,newton_renderer,... -> resolved tiled_camera.renderer_cfg is NewtonWarpRendererCfg."""
        env_cfg = _resolve_with_presets("newton,newton_renderer,rgb")
        renderer_cfg = env_cfg.tiled_camera.renderer_cfg
        assert renderer_cfg is not None
        assert isinstance(renderer_cfg, NewtonWarpRendererCfg)

    def test_resolve_renderer_backend_ovrtx(self):
        """presets=...,ovrtx_renderer,... -> resolve_task_config yields OVRTX renderer.
        Skips if isaaclab_ov/ovrtx not installed or ovrtx_renderer preset not registered."""
        try:
            from isaaclab_ov.renderers import OVRTXRendererCfg
        except (ModuleNotFoundError, ImportError):
            pytest.skip("isaaclab_ov / ovrtx not installed")

        _, _, presets = _get_registered_env_and_presets()
        renderer_presets = presets.get("env", {}).get("tiled_camera.renderer_cfg", set())
        if "ovrtx_renderer" not in renderer_presets:
            pytest.skip("ovrtx_renderer preset not registered (add to MultiBackendRendererCfg)")
        env_cfg = _resolve_with_presets("physx,ovrtx_renderer,rgb")
        renderer_cfg = env_cfg.tiled_camera.renderer_cfg
        assert renderer_cfg is not None
        assert isinstance(renderer_cfg, OVRTXRendererCfg)

    def test_resolve_camera_type_rgb(self):
        """presets=...,rgb -> resolved tiled_camera.data_types includes rgb."""
        env_cfg = _resolve_with_presets("physx,isaacsim_rtx_renderer,rgb")
        data_types = getattr(env_cfg.tiled_camera, "data_types", [])
        assert "rgb" in data_types

    def test_resolve_camera_type_depth(self):
        """presets=...,depth -> resolved tiled_camera.data_types includes depth."""
        env_cfg = _resolve_with_presets("physx,isaacsim_rtx_renderer,depth")
        data_types = getattr(env_cfg.tiled_camera, "data_types", [])
        assert "depth" in data_types

    def test_resolve_camera_type_albedo(self):
        """presets=...,albedo -> resolved tiled_camera.data_types includes albedo."""
        env_cfg = _resolve_with_presets("physx,isaacsim_rtx_renderer,albedo")
        data_types = getattr(env_cfg.tiled_camera, "data_types", [])
        assert "albedo" in data_types
