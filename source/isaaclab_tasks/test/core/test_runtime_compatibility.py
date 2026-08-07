# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for runtime-compatibility validation in ``isaaclab.app.sim_launcher``.

The OVRTX renderer is kitless and cannot run together with Isaac Sim / Kit
runtimes (``PhysxCfg`` physics or the Kit visualizer). These tests verify that
invalid combinations selected via ``presets=...`` (or ``--visualizer kit``) raise
a clear error pointing the user at the correct ``isaacsim_rtx`` preset.
No Kit/GPU required — safe for CI and beginners.
"""

import argparse
import ast
import inspect
import sys

import pytest
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.app.sim_launcher as sim_launcher_module
import isaaclab.utils as isaaclab_utils
from isaaclab.app import scan
from isaaclab.app.sim_launcher import _get_kit_runtime_sources, _validate_runtime, launch_simulation
from isaaclab.physics import PhysxAutoCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

_CAMERA_PRESETS_TASK = "Isaac-Cartpole-Camera-Direct"


def validate_runtime_compatibility(env_cfg, launcher_args=None):
    """Run the single-scan runtime validation for *env_cfg* (test adapter)."""
    config_scan = scan(env_cfg, launcher_args)
    kit_sources = _get_kit_runtime_sources(config_scan, launcher_args)
    _validate_runtime(config_scan, kit_sources)
    return config_scan


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


# ---------------------------------------------------------------------------
# Architecture: validation consumes one resolved Kit-source value
# ---------------------------------------------------------------------------


def test_runtime_validation_consumes_resolved_kit_sources():
    """Keep config and launcher interpretation outside the compatibility validator."""
    assert list(inspect.signature(_validate_runtime).parameters) == ["scan", "kit_sources"]

    tree = ast.parse(inspect.getsource(_validate_runtime))
    forbidden_scan_fields = {
        "has_kit_camera",
        "has_kit_physics",
        "needs_kit",
        "visualizer_intent",
    }
    accessed_attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert accessed_attributes.isdisjoint(forbidden_scan_fields)

    called_helpers = {
        node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_has_kit_visualizer" not in called_helpers


# ---------------------------------------------------------------------------
# Invalid: OVRTX renderer + Isaac Sim / Kit
# ---------------------------------------------------------------------------


def test_isaacsim_physx_plus_ovrtx_raises():
    """Concrete Isaac Sim PhysX plus OVRTX is the canonical invalid combination."""
    env_cfg = _resolve_with_presets("isaacsim_physx,ovrtx")
    with pytest.raises(ValueError) as excinfo:
        validate_runtime_compatibility(env_cfg)
    msg = str(excinfo.value)
    assert "PhysxCfg" in msg
    assert "isaacsim_rtx" in msg


def test_kit_visualizer_plus_ovrtx_raises():
    """``--visualizer kit`` combined with OVRTX renderer must raise.

    Use Newton physics so the only Kit-side runtime is the visualizer; this
    isolates the visualizer-vs-renderer check from the physics-vs-renderer one.
    """
    env_cfg = _resolve_with_presets("newton,ovrtx")
    launcher_args = argparse.Namespace(visualizer="kit")
    with pytest.raises(ValueError) as excinfo:
        validate_runtime_compatibility(env_cfg, launcher_args)
    msg = str(excinfo.value)
    assert "Kit visualizer" in msg
    assert "isaacsim_rtx" in msg


def test_kit_visualizer_dict_args_plus_ovrtx_raises():
    """The dict form of launcher args (used by Hydra) must also be inspected."""
    env_cfg = _resolve_with_presets("newton,ovrtx")
    with pytest.raises(ValueError, match=r"Kit visualizer"):
        validate_runtime_compatibility(env_cfg, {"visualizer": "kit,newton"})


def test_kit_renderer_plus_ovrtx_raises():
    """A Kit renderer and OVRTX cannot initialize in the same process."""
    env_cfg = _resolve_with_presets("newton,isaacsim_rtx")
    mixed_cfg = argparse.Namespace(
        physics=env_cfg.sim.physics,
        kit_camera=env_cfg.tiled_camera,
        ovrtx_renderer=OVRTXRendererCfg(),
    )

    with pytest.raises(ValueError, match="Kit-based renderer"):
        validate_runtime_compatibility(mixed_cfg)


@pytest.mark.parametrize(
    ("launcher_args", "source"),
    [
        (argparse.Namespace(experience="custom.kit"), "explicit Kit experience"),
        (argparse.Namespace(livestream=2), "livestreaming"),
    ],
)
def test_launcher_kit_source_plus_ovrtx_raises(launcher_args, source):
    """Every launcher-side Kit source must conflict with OVRTX."""
    env_cfg = _resolve_with_presets("newton,ovrtx")

    with pytest.raises(ValueError, match=source):
        validate_runtime_compatibility(env_cfg, launcher_args)


def test_default_kit_runtime_plus_ovrtx_raises(monkeypatch: pytest.MonkeyPatch):
    """A config without physics defaults to Kit, which cannot share OVRTX."""
    monkeypatch.delenv("LIVESTREAM", raising=False)
    renderer_only_cfg = argparse.Namespace(renderer=OVRTXRendererCfg())

    with pytest.raises(ValueError, match="default Isaac Sim / Kit runtime"):
        validate_runtime_compatibility(renderer_only_cfg)


# ---------------------------------------------------------------------------
# Invalid: OvPhysX physics + Isaac Sim / Kit
# ---------------------------------------------------------------------------


def test_ovphysx_plus_kit_visualizer_raises():
    """OvPhysX cannot share a process with the Kit visualizer."""
    env_cfg = _resolve_with_presets("ovphysx,isaacsim_rtx")
    launcher_args = argparse.Namespace(visualizer="kit")
    with pytest.raises(ValueError) as excinfo:
        validate_runtime_compatibility(env_cfg, launcher_args)
    msg = str(excinfo.value)
    assert "OvPhysX" in msg
    assert "Kit visualizer" in msg


def test_ovphysx_dict_args_plus_kit_visualizer_raises():
    """The dict launcher-args form must also reject OvPhysX with Kit visualization."""
    env_cfg = _resolve_with_presets("ovphysx,isaacsim_rtx")
    with pytest.raises(ValueError, match=r"OvPhysX.*Kit visualizer"):
        validate_runtime_compatibility(env_cfg, {"visualizer": "kit,newton"})


def test_ovphysx_plus_kit_physics_raises():
    """Two physics configs cannot pull OvPhysX and Kit into the same process."""
    mixed_cfg = argparse.Namespace(
        ovphysx_physics=OvPhysxCfg(),
        kit_physics=PhysxCfg(),
    )

    with pytest.raises(ValueError, match="PhysxCfg"):
        validate_runtime_compatibility(mixed_cfg)


def test_explicit_kit_experience_plus_ovphysx_raises():
    """An explicit Kit experience must conflict with OvPhysX."""
    env_cfg = _resolve_with_presets("ovphysx,ovrtx")
    launcher_args = argparse.Namespace(experience="custom.kit")

    with pytest.raises(ValueError, match="explicit Kit experience"):
        validate_runtime_compatibility(env_cfg, launcher_args)


def test_ovphysx_plus_kit_camera_without_visualizer_raises():
    """A Kit-based renderer pulls in Kit even with no visualizer, which OvPhysX cannot share.

    Without a visualizer the only Kit signal is the camera, so this is the case the visualizer-only
    guard missed: it previously reached OvPhysX's own initialization and failed there instead.
    """
    env_cfg = _resolve_with_presets("ovphysx,isaacsim_rtx")
    with pytest.raises(ValueError) as excinfo:
        validate_runtime_compatibility(env_cfg, argparse.Namespace(visualizer=None))
    msg = str(excinfo.value)
    assert "OvPhysX" in msg
    assert "renderer" in msg


# ---------------------------------------------------------------------------
# Valid combinations: must NOT raise
# ---------------------------------------------------------------------------


def test_newton_plus_ovrtx_is_valid():
    """Newton physics + OVRTX renderer is the supported kitless combination."""
    env_cfg = _resolve_with_presets("newton,ovrtx")
    validate_runtime_compatibility(env_cfg)


def test_default_isaacsim_physx_plus_ovrtx_raises():
    """The concrete default Isaac Sim PhysX backend is incompatible with OVRTX."""
    env_cfg = _resolve_with_presets("ovrtx")

    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    with pytest.raises(ValueError, match="PhysxCfg"):
        validate_runtime_compatibility(env_cfg)


def test_explicit_auto_physx_plus_ovrtx_resolves_to_ovphysx():
    """The ``physx`` preset remains automatic when paired with OVRTX."""
    env_cfg = _resolve_with_presets("physx,ovrtx")

    assert isinstance(env_cfg.sim.physics, PhysxAutoCfg)

    config_scan = validate_runtime_compatibility(env_cfg)

    assert isinstance(env_cfg.sim.physics, OvPhysxCfg)
    assert isinstance(env_cfg.tiled_camera.renderer_cfg, OVRTXRendererCfg)
    assert config_scan.needs_kit is False


def test_physx_plus_isaacsim_rtx_is_valid():
    """PhysX physics + Isaac RTX renderer is the supported Kit combination."""
    env_cfg = _resolve_with_presets("physx,isaacsim_rtx")
    config_scan = validate_runtime_compatibility(env_cfg)

    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    assert isinstance(env_cfg.tiled_camera.renderer_cfg, IsaacRtxRendererCfg)
    assert config_scan.needs_kit is True


def test_auto_physx_configured_kit_visualizer_resolves_to_isaac_sim_backends():
    """Config-declared Kit visualizers should drive automatic PhysX and RTX resolution."""

    class KitVisualizerCfg:
        visualizer_type = "kit"

    env_cfg = _resolve_with_args("physics=physx", "renderer=rtx")
    env_cfg.sim.visualizer_cfgs = KitVisualizerCfg()
    config_scan = validate_runtime_compatibility(env_cfg)

    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    assert isinstance(env_cfg.tiled_camera.renderer_cfg, IsaacRtxRendererCfg)
    assert config_scan.needs_kit is True


def test_auto_physx_livestream_without_launcher_args_resolves_to_isaac_sim_backends(
    monkeypatch: pytest.MonkeyPatch,
):
    """Livestream env vars should require Kit even when no launcher args object is provided."""
    env_cfg = _resolve_with_args("physics=physx", "renderer=rtx")
    monkeypatch.setenv("LIVESTREAM", "2")
    config_scan = validate_runtime_compatibility(env_cfg)

    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    assert isinstance(env_cfg.tiled_camera.renderer_cfg, IsaacRtxRendererCfg)
    assert config_scan.needs_kit is True


def test_auto_physx_explicit_experience_resolves_to_isaac_sim_backends():
    """An explicit Kit experience should drive automatic PhysX and RTX resolution."""
    env_cfg = _resolve_with_args("physics=physx", "renderer=rtx")
    config_scan = validate_runtime_compatibility(env_cfg, argparse.Namespace(experience="custom.kit"))

    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    assert isinstance(env_cfg.tiled_camera.renderer_cfg, IsaacRtxRendererCfg)
    assert config_scan.needs_kit is True


def test_default_preset_is_valid():
    """The default preset (PhysX + Isaac RTX) is supported."""
    env_cfg = _resolve_with_presets("default")
    validate_runtime_compatibility(env_cfg)


def test_rtx_with_default_physx_is_valid_and_resolves_to_isaac_sim_backends():
    """The RTX selector follows the default concrete Isaac Sim PhysX backend."""
    env_cfg = _resolve_with_presets("rtx")
    config_scan = validate_runtime_compatibility(env_cfg)

    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    assert isinstance(env_cfg.tiled_camera.renderer_cfg, IsaacRtxRendererCfg)
    assert config_scan.needs_kit is True


def test_renderer_selector_physx_rtx_is_valid_and_resolves_to_ovphysx_and_ovrtx():
    """The automatic PhysX and RTX selectors choose kitless backends without Kit signals."""
    env_cfg = _resolve_with_args("physics=physx", "renderer=rtx")

    assert isinstance(env_cfg.sim.physics, PhysxAutoCfg)

    config_scan = validate_runtime_compatibility(env_cfg)

    assert isinstance(env_cfg.sim.physics, OvPhysxCfg)
    assert isinstance(env_cfg.tiled_camera.renderer_cfg, OVRTXRendererCfg)
    assert config_scan.needs_kit is False


def test_renderer_selector_physx_rtx_with_kit_visualizer_resolves_to_isaac_sim_backends():
    """The automatic PhysX and RTX selectors choose Isaac Sim backends when the Kit viewer is requested."""
    env_cfg = _resolve_with_args("physics=physx", "renderer=rtx")
    config_scan = validate_runtime_compatibility(env_cfg, argparse.Namespace(visualizer="kit"))

    assert isinstance(env_cfg.sim.physics, PhysxCfg)
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
    env_cfg = _resolve_with_presets("newton,isaacsim_rtx")
    validate_runtime_compatibility(env_cfg)


def test_kit_visualizer_with_isaacsim_rtx_is_valid():
    """``--visualizer kit`` is fine as long as no OVRTX renderer is configured."""
    env_cfg = _resolve_with_presets("newton,isaacsim_rtx")
    validate_runtime_compatibility(env_cfg, argparse.Namespace(visualizer="kit"))
