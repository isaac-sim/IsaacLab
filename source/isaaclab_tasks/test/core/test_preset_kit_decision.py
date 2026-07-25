# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for preset resolution and Kit decision logic.

These tests verify that given presets (e.g. ``presets=newton_mjwarp,ovrtx``),
the config-based logic correctly decides whether Isaac Sim Kit is needed.
No Kit/GPU required — safe for CI and beginners.
"""

import sys
from argparse import Namespace

import pytest
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

from isaaclab.app import scan

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_tasks.utils.preset_cli import enumerate_task_presets
from isaaclab_tasks.utils.preset_target import PresetTarget

_CAMERA_PRESETS_TASK = "Isaac-Cartpole-Camera-Direct"


def _resolve_with_presets(presets: str):
    """Resolve env_cfg with given presets. Modifies sys.argv temporarily."""
    return _resolve_with_args(f"presets={presets}")


def _resolve_with_args(*args: str):
    """Resolve env_cfg with the given Hydra-style args. Modifies sys.argv temporarily."""
    old_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0], *args]
        env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, "rl_games_cfg_entry_point")
        return env_cfg
    finally:
        sys.argv = old_argv


def _resolve_runtime_renderer(env_cfg, launcher_args=None):
    """Apply launch-time auto RTX renderer resolution and return the updated scan."""
    return scan(env_cfg, launcher_args)


def test_resolve_task_config_applies_plain_scalar_override():
    """Plain ``env.*=value`` overrides should resolve without requiring Hydra composition."""
    old_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0], "env.scene.num_envs=123"]
        env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, "rl_games_cfg_entry_point")
    finally:
        sys.argv = old_argv

    assert env_cfg.scene.num_envs == 123


def test_rtx_is_renderer_selector():
    """The automatic RTX selector is exposed as ``renderer=rtx``."""
    preset_map = enumerate_task_presets(_CAMERA_PRESETS_TASK)

    assert preset_map is not None
    assert "rtx" in preset_map[PresetTarget.RENDERER]


def test_isaacsim_physx_is_physics_selector():
    """The concrete Isaac Sim PhysX selector is exposed as ``physics=isaacsim_physx``."""
    preset_map = enumerate_task_presets(_CAMERA_PRESETS_TASK)

    assert preset_map is not None
    assert "isaacsim_physx" in preset_map[PresetTarget.PHYSICS]


def test_cartpole_physx_preset_is_plain_physx_cfg():
    """Task configs expose ``physx`` as concrete PhysX, not a public auto wrapper."""
    from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpolePhysicsCfg

    physics_cfg = CartpolePhysicsCfg()

    assert isinstance(physics_cfg.physx, PhysxCfg)
    assert isinstance(physics_cfg.default, PhysxCfg)


def test_physx_and_isaacsim_physx_presets_conflict():
    """``physx`` and ``isaacsim_physx`` are distinct choices even when both use PhysxCfg."""
    with pytest.raises(ValueError, match="Conflicting global presets"):
        _resolve_with_presets("physx,isaacsim_physx")


def test_preset_mjwarp_ovrtx_does_not_need_kit():
    """Newton + OVRTX renderer is kitless — no AppLauncher required."""
    env_cfg = _resolve_with_presets("newton_mjwarp,ovrtx")
    needs_kit = scan(env_cfg).needs_kit
    assert needs_kit is False


def test_preset_rtx_resolves_to_ovphysx_and_ovrtx_without_kit():
    """The RTX preset uses kitless PhysX-family backends when no Kit runtime is requested."""
    env_cfg = _resolve_with_presets("rtx")
    config_scan = _resolve_runtime_renderer(env_cfg)

    assert isinstance(env_cfg.sim.physics, OvPhysxCfg)
    assert isinstance(env_cfg.tiled_camera.renderer_cfg, OVRTXRendererCfg)
    assert config_scan.needs_kit is False


def test_renderer_selector_rtx_resolves_to_ovphysx_and_ovrtx_without_kit():
    """The RTX renderer selector uses kitless PhysX-family backends without Kit signals."""
    env_cfg = _resolve_with_args("renderer=rtx")
    config_scan = _resolve_runtime_renderer(env_cfg)

    assert isinstance(env_cfg.sim.physics, OvPhysxCfg)
    assert isinstance(env_cfg.tiled_camera.renderer_cfg, OVRTXRendererCfg)
    assert config_scan.needs_kit is False


def test_renderer_selector_physx_rtx_resolves_to_ovphysx_without_kit():
    """Automatic PhysX and RTX selectors use kitless backends when no Kit runtime is requested."""
    env_cfg = _resolve_with_args("physics=physx", "renderer=rtx")

    assert isinstance(env_cfg.sim.physics, PhysxCfg)

    config_scan = _resolve_runtime_renderer(env_cfg)

    assert isinstance(env_cfg.sim.physics, OvPhysxCfg)
    assert isinstance(env_cfg.tiled_camera.renderer_cfg, OVRTXRendererCfg)
    assert config_scan.needs_kit is False


def test_renderer_selector_physx_rtx_resolves_to_physx_with_kit_visualizer():
    """Automatic PhysX and RTX selectors use Isaac Sim backends when the Kit visualizer is requested."""
    env_cfg = _resolve_with_args("physics=physx", "renderer=rtx")

    assert isinstance(env_cfg.sim.physics, PhysxCfg)

    config_scan = _resolve_runtime_renderer(env_cfg, Namespace(visualizer="kit"))

    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    assert isinstance(env_cfg.tiled_camera.renderer_cfg, IsaacRtxRendererCfg)
    assert config_scan.needs_kit is True


def test_renderer_selector_isaacsim_physx_rtx_resolves_to_physx_and_needs_kit():
    """The explicit Isaac Sim PhysX selector keeps RTX on the Kit-compatible backend."""
    env_cfg = _resolve_with_args("physics=isaacsim_physx", "renderer=rtx")
    config_scan = _resolve_runtime_renderer(env_cfg)

    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    assert isinstance(env_cfg.tiled_camera.renderer_cfg, IsaacRtxRendererCfg)
    assert config_scan.needs_kit is True


def test_preset_mjwarp_rtx_resolves_to_ovrtx_without_kit():
    """The RTX preset uses OVRTX for a kitless Newton run."""
    env_cfg = _resolve_with_presets("newton_mjwarp,rtx")
    config_scan = _resolve_runtime_renderer(env_cfg)

    assert isinstance(env_cfg.tiled_camera.renderer_cfg, OVRTXRendererCfg)
    assert config_scan.needs_kit is False


def test_preset_mjwarp_rtx_resolves_to_isaac_rtx_with_kit_visualizer():
    """The RTX preset uses Isaac RTX when the Kit visualizer is requested."""
    env_cfg = _resolve_with_presets("newton_mjwarp,rtx")
    config_scan = _resolve_runtime_renderer(env_cfg, Namespace(visualizer="kit"))

    assert isinstance(env_cfg.tiled_camera.renderer_cfg, IsaacRtxRendererCfg)
    assert config_scan.needs_kit is True


def test_renderer_selector_mjwarp_rtx_resolves_to_ovrtx_without_kit():
    """The RTX renderer selector uses OVRTX for a kitless Newton run."""
    env_cfg = _resolve_with_args("physics=newton_mjwarp", "renderer=rtx")
    config_scan = _resolve_runtime_renderer(env_cfg)

    assert isinstance(env_cfg.tiled_camera.renderer_cfg, OVRTXRendererCfg)
    assert config_scan.needs_kit is False


def test_preset_mjwarp_newton_renderer_does_not_need_kit():
    """Newton + Newton Warp renderer is kitless."""
    env_cfg = _resolve_with_presets("newton_mjwarp,newton_renderer")
    needs_kit = scan(env_cfg).needs_kit
    assert needs_kit is False


def test_preset_physx_with_default_kit_camera_resolves_to_physx():
    """Automatic PhysX resolves to Isaac Sim PhysX when the default camera requires Kit."""
    env_cfg = _resolve_with_presets("physx")
    config_scan = scan(env_cfg)

    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    assert config_scan.needs_kit is True


def test_preset_default_needs_kit():
    """Default automatic PhysX plus Isaac RTX requires Kit."""
    env_cfg = _resolve_with_presets("default")
    needs_kit = scan(env_cfg).needs_kit
    assert needs_kit is True


def test_preset_mjwarp_isaac_rtx_needs_kit():
    """Newton + Isaac RTX renderer requires Kit (RTX runs in Kit)."""
    env_cfg = _resolve_with_presets("newton_mjwarp,isaacsim_rtx")
    needs_kit = scan(env_cfg).needs_kit
    assert needs_kit is True
