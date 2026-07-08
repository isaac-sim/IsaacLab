# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for runtime-compatibility validation in ``isaaclab.app.sim_launcher``.

The OVRTX renderer is kitless and cannot run together with Isaac Sim / Kit
runtimes (``PhysxCfg`` physics or the Kit visualizer). These tests verify that
invalid combinations selected via ``presets=...`` (or ``--visualizer kit``) raise
a clear error pointing the user at the correct ``isaacsim_rtx_renderer`` preset.
No Kit/GPU required — safe for CI and beginners.
"""

import argparse
import sys

import pytest
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.app.sim_launcher as sim_launcher_module
import isaaclab.utils as isaaclab_utils
from isaaclab.app import scan
from isaaclab.app.sim_launcher import _validate_runtime, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

_CAMERA_PRESETS_TASK = "Isaac-Cartpole-Camera-Direct"


def validate_runtime_compatibility(env_cfg, launcher_args=None):
    """Run the single-scan runtime validation for *env_cfg* (test adapter)."""
    config_scan = scan(env_cfg, launcher_args)
    _validate_runtime(config_scan, launcher_args)
    return config_scan


def _resolve_with_presets(presets: str):
    """Resolve env_cfg with given presets. Modifies sys.argv temporarily."""
    old_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0], f"presets={presets}"]
        env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, "rl_games_cfg_entry_point")
        return env_cfg
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# Invalid: OVRTX renderer + Isaac Sim / Kit
# ---------------------------------------------------------------------------


def test_default_physx_plus_ovrtx_raises():
    """Default physics is PhysxCfg; pairing it with the OVRTX renderer must raise."""
    env_cfg = _resolve_with_presets("ovrtx_renderer")
    with pytest.raises(ValueError, match=r"OVRTX renderer.*Isaac Sim / Kit"):
        validate_runtime_compatibility(env_cfg)


def test_explicit_physx_plus_ovrtx_raises():
    """Explicit physx preset + ovrtx_renderer is the canonical invalid combination."""
    env_cfg = _resolve_with_presets("physx,ovrtx_renderer")
    with pytest.raises(ValueError) as excinfo:
        validate_runtime_compatibility(env_cfg)
    msg = str(excinfo.value)
    assert "PhysxCfg" in msg
    assert "isaacsim_rtx_renderer" in msg


def test_kit_visualizer_plus_ovrtx_raises():
    """``--visualizer kit`` combined with OVRTX renderer must raise.

    Use Newton physics so the only Kit-side runtime is the visualizer; this
    isolates the visualizer-vs-renderer check from the physics-vs-renderer one.
    """
    env_cfg = _resolve_with_presets("newton,ovrtx_renderer")
    launcher_args = argparse.Namespace(visualizer="kit")
    with pytest.raises(ValueError) as excinfo:
        validate_runtime_compatibility(env_cfg, launcher_args)
    msg = str(excinfo.value)
    assert "Kit visualizer" in msg
    assert "isaacsim_rtx_renderer" in msg


def test_kit_visualizer_dict_args_plus_ovrtx_raises():
    """The dict form of launcher args (used by Hydra) must also be inspected."""
    env_cfg = _resolve_with_presets("newton,ovrtx_renderer")
    with pytest.raises(ValueError, match=r"Kit visualizer"):
        validate_runtime_compatibility(env_cfg, {"visualizer": "kit,newton"})


# ---------------------------------------------------------------------------
# Invalid: OvPhysX physics + Kit visualizer
# ---------------------------------------------------------------------------


def test_ovphysx_plus_kit_visualizer_raises():
    """OvPhysX cannot share a process with the Kit visualizer."""
    env_cfg = _resolve_with_presets("ovphysx,isaacsim_rtx_renderer")
    launcher_args = argparse.Namespace(visualizer="kit")
    with pytest.raises(ValueError) as excinfo:
        validate_runtime_compatibility(env_cfg, launcher_args)
    msg = str(excinfo.value)
    assert "OvPhysX" in msg
    assert "Kit visualizer" in msg


def test_ovphysx_dict_args_plus_kit_visualizer_raises():
    """The dict launcher-args form must also reject OvPhysX with Kit visualization."""
    env_cfg = _resolve_with_presets("ovphysx,isaacsim_rtx_renderer")
    with pytest.raises(ValueError, match=r"OvPhysX.*Kit visualizer"):
        validate_runtime_compatibility(env_cfg, {"visualizer": "kit,newton"})


# ---------------------------------------------------------------------------
# Valid combinations: must NOT raise
# ---------------------------------------------------------------------------


def test_newton_plus_ovrtx_is_valid():
    """Newton physics + OVRTX renderer is the supported kitless combination."""
    env_cfg = _resolve_with_presets("newton,ovrtx_renderer")
    validate_runtime_compatibility(env_cfg)


def test_physx_plus_isaacsim_rtx_is_valid():
    """PhysX physics + Isaac RTX renderer is the supported Kit combination."""
    env_cfg = _resolve_with_presets("physx,isaacsim_rtx_renderer")
    validate_runtime_compatibility(env_cfg)


def test_default_preset_is_valid():
    """The default preset (PhysX + Isaac RTX) is supported."""
    env_cfg = _resolve_with_presets("default")
    validate_runtime_compatibility(env_cfg)


def test_rtx_with_default_physx_is_valid_and_resolves_to_isaac_rtx():
    """The RTX preset chooses the Kit-compatible renderer for PhysX."""
    env_cfg = _resolve_with_presets("rtx")
    config_scan = validate_runtime_compatibility(env_cfg)

    assert isinstance(env_cfg.tiled_camera.renderer_cfg, IsaacRtxRendererCfg)
    assert config_scan.needs_kit is True


def test_rtx_with_newton_is_valid_and_resolves_to_ovrtx():
    """The RTX preset chooses OVRTX when no Isaac Sim runtime is needed."""
    env_cfg = _resolve_with_presets("newton_mjwarp,rtx")
    config_scan = validate_runtime_compatibility(env_cfg)

    assert isinstance(env_cfg.tiled_camera.renderer_cfg, OVRTXRendererCfg)
    assert config_scan.needs_kit is False


def test_rtx_with_ovphysx_is_valid_and_resolves_to_ovrtx():
    """The RTX preset chooses OVRTX for an OvPhysX kitless run."""
    env_cfg = _resolve_with_presets("ovphysx,rtx")
    config_scan = validate_runtime_compatibility(env_cfg)

    assert config_scan.has_ovphysx_physics is True
    assert isinstance(env_cfg.tiled_camera.renderer_cfg, OVRTXRendererCfg)
    assert config_scan.needs_kit is False


def test_rtx_with_kit_visualizer_is_valid_and_resolves_to_isaac_rtx():
    """The RTX preset chooses Isaac RTX when the Kit visualizer is requested."""
    env_cfg = _resolve_with_presets("newton_mjwarp,rtx")
    config_scan = validate_runtime_compatibility(env_cfg, argparse.Namespace(visualizer="kit"))

    assert isinstance(env_cfg.tiled_camera.renderer_cfg, IsaacRtxRendererCfg)
    assert config_scan.needs_kit is True


def test_livestream_rtx_injects_kit_before_auto_rtx_resolution(monkeypatch: pytest.MonkeyPatch):
    """Livestreaming should make ``presets=newton_mjwarp,rtx`` choose Isaac RTX."""
    env_cfg = _resolve_with_presets("newton_mjwarp,rtx")
    launcher_args = argparse.Namespace(livestream=2, visualizer=None, visualizer_explicit=False)
    monkeypatch.setattr(sim_launcher_module, "_ensure_isaac_sim_available", lambda: None)
    monkeypatch.setattr(isaaclab_utils, "has_kit", lambda: True)

    with launch_simulation(env_cfg, launcher_args) as physics_cfg:
        assert type(physics_cfg).__name__ == "NewtonCfg"

    assert launcher_args.visualizer == ["kit"]
    assert launcher_args.enable_cameras is True
    assert isinstance(env_cfg.tiled_camera.renderer_cfg, IsaacRtxRendererCfg)


def test_newton_plus_isaacsim_rtx_is_valid():
    """Newton + Isaac RTX renderer is supported (RTX runs in Kit, Newton syncs to USD)."""
    env_cfg = _resolve_with_presets("newton,isaacsim_rtx_renderer")
    validate_runtime_compatibility(env_cfg)


def test_kit_visualizer_with_isaacsim_rtx_is_valid():
    """``--visualizer kit`` is fine as long as no OVRTX renderer is configured."""
    env_cfg = _resolve_with_presets("newton,isaacsim_rtx_renderer")
    validate_runtime_compatibility(env_cfg, argparse.Namespace(visualizer="kit"))
