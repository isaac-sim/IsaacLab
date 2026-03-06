# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests: CLI preset string -> register_task / parse_overrides / apply_overrides -> resolved backends.

Tests the three flows (simulation backend, renderer backend, camera type) with 
initialization and no SimulationContext or rendering; test only config resolution.
"""

import pytest

# Register cartpole presets task so load_cfg_from_registry can find it.
pytest.importorskip("gymnasium")
import isaaclab_tasks.direct.cartpole  # noqa: F401  # register Isaac-Cartpole-Camera-Presets-Direct-v0

from isaaclab_tasks.utils.hydra import apply_overrides, parse_overrides, register_task


# Task name from isaaclab_tasks.direct.cartpole
TASK_NAME = "Isaac-Cartpole-Camera-Presets-Direct-v0"
AGENT_ENTRY = ""


def _get_registered_env_and_presets():
    """Register task and return (env_cfg, agent_cfg, presets) for use in apply_overrides."""
    env_cfg, agent_cfg, presets = register_task(TASK_NAME, AGENT_ENTRY)
    return env_cfg, agent_cfg, presets


class TestHydraPresetsIntegration:
    """Test that preset string -> parse_overrides -> apply_overrides yields expected backends."""

    def test_parse_overrides_global_presets(self):
        """parse_overrides extracts global presets from presets=..."""
        _, _, presets = _get_registered_env_and_presets()
        global_presets, preset_sel, _, _ = parse_overrides(
            ["presets=physx,isaacsim_rtx_renderer,rgb"], presets
        )
        assert set(global_presets) == {"physx", "isaacsim_rtx_renderer", "rgb"}
        assert preset_sel == []

    def test_apply_overrides_simulation_backend_physx(self):
        """presets=physx,... -> env_cfg.sim.physics is PhysxCfg."""
        env_cfg, agent_cfg, presets = _get_registered_env_and_presets()
        hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict() if agent_cfg else {}}
        global_presets, preset_sel, preset_scalar, _ = parse_overrides(
            ["presets=physx,isaacsim_rtx_renderer,rgb"], presets
        )
        env_cfg, _ = apply_overrides(
            env_cfg, agent_cfg, hydra_cfg, global_presets, preset_sel, preset_scalar, presets
        )
        physics = env_cfg.sim.physics
        assert physics is not None
        assert type(physics).__name__ == "PhysxCfg"

    def test_apply_overrides_simulation_backend_newton(self):
        """presets=newton,... -> env_cfg.sim.physics is NewtonCfg."""
        env_cfg, agent_cfg, presets = _get_registered_env_and_presets()
        hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict() if agent_cfg else {}}
        global_presets, preset_sel, preset_scalar, _ = parse_overrides(
            ["presets=newton,newton_renderer,rgb"], presets
        )
        env_cfg, _ = apply_overrides(
            env_cfg, agent_cfg, hydra_cfg, global_presets, preset_sel, preset_scalar, presets
        )
        physics = env_cfg.sim.physics
        assert physics is not None
        assert type(physics).__name__ == "NewtonCfg"

    def test_apply_overrides_renderer_backend_isaac_rtx(self):
        """presets=...,isaacsim_rtx_renderer,... -> tiled_camera.renderer_cfg has renderer_type isaac_rtx."""
        env_cfg, agent_cfg, presets = _get_registered_env_and_presets()
        hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict() if agent_cfg else {}}
        global_presets, preset_sel, preset_scalar, _ = parse_overrides(
            ["presets=physx,isaacsim_rtx_renderer,rgb"], presets
        )
        env_cfg, _ = apply_overrides(
            env_cfg, agent_cfg, hydra_cfg, global_presets, preset_sel, preset_scalar, presets
        )
        renderer_cfg = env_cfg.tiled_camera.renderer_cfg
        assert renderer_cfg is not None
        assert type(renderer_cfg).__name__ == "IsaacRtxRendererCfg"

    def test_apply_overrides_renderer_backend_newton_warp(self):
        """presets=...,newton_renderer,... -> tiled_camera.renderer_cfg is NewtonWarpRendererCfg."""
        env_cfg, agent_cfg, presets = _get_registered_env_and_presets()
        hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict() if agent_cfg else {}}
        global_presets, preset_sel, preset_scalar, _ = parse_overrides(
            ["presets=newton,newton_renderer,rgb"], presets
        )
        env_cfg, _ = apply_overrides(
            env_cfg, agent_cfg, hydra_cfg, global_presets, preset_sel, preset_scalar, presets
        )
        renderer_cfg = env_cfg.tiled_camera.renderer_cfg
        assert renderer_cfg is not None
        name = type(renderer_cfg).__name__
        assert "Newton" in name and "Warp" in name

    def test_apply_overrides_camera_type_rgb(self):
        """presets=...,rgb -> tiled_camera.data_types includes rgb."""
        env_cfg, agent_cfg, presets = _get_registered_env_and_presets()
        hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict() if agent_cfg else {}}
        global_presets, preset_sel, preset_scalar, _ = parse_overrides(
            ["presets=physx,isaacsim_rtx_renderer,rgb"], presets
        )
        env_cfg, _ = apply_overrides(
            env_cfg, agent_cfg, hydra_cfg, global_presets, preset_sel, preset_scalar, presets
        )
        data_types = getattr(env_cfg.tiled_camera, "data_types", [])
        assert "rgb" in data_types

    def test_apply_overrides_camera_type_depth(self):
        """presets=...,depth -> tiled_camera.data_types includes depth."""
        env_cfg, agent_cfg, presets = _get_registered_env_and_presets()
        hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict() if agent_cfg else {}}
        global_presets, preset_sel, preset_scalar, _ = parse_overrides(
            ["presets=physx,isaacsim_rtx_renderer,depth"], presets
        )
        env_cfg, _ = apply_overrides(
            env_cfg, agent_cfg, hydra_cfg, global_presets, preset_sel, preset_scalar, presets
        )
        data_types = getattr(env_cfg.tiled_camera, "data_types", [])
        assert "depth" in data_types

    def test_apply_overrides_camera_type_albedo(self):
        """presets=...,albedo -> tiled_camera.data_types includes albedo."""
        env_cfg, agent_cfg, presets = _get_registered_env_and_presets()
        hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict() if agent_cfg else {}}
        global_presets, preset_sel, preset_scalar, _ = parse_overrides(
            ["presets=physx,isaacsim_rtx_renderer,albedo"], presets
        )
        env_cfg, _ = apply_overrides(
            env_cfg, agent_cfg, hydra_cfg, global_presets, preset_sel, preset_scalar, presets
        )
        data_types = getattr(env_cfg.tiled_camera, "data_types", [])
        assert "albedo" in data_types
