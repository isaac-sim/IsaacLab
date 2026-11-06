# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for runtime-compatibility validation in ``isaaclab.app.sim_launcher``.

The OVRTX renderer is kitless and cannot run together with Isaac Sim / Kit
runtimes (``PhysxCfg`` physics or the Kit visualizer). These tests verify that
invalid composed configurations raise a clear error identifying compatible
concrete renderer configurations. No Kit/GPU required.
"""

import argparse
from collections import namedtuple

import isaaclab_physx.app as physx_app
import pytest
from isaaclab_newton.physics import NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_visualizers.kit import KitBackendCfg

from isaaclab.app import SimulationLauncher, launch_simulation, validate_launch
from isaaclab.physics import PhysxAutoCfg
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

_CAMERA_PRESETS_TASK = "Isaac-Cartpole-Camera-Direct"


# ---------------------------------------------------------------------------
# Invalid: OVRTX renderer + Isaac Sim / Kit
# ---------------------------------------------------------------------------


def test_isaacsim_physx_plus_ovrtx_raises():
    """Concrete Isaac Sim PhysX plus OVRTX is the canonical invalid combination."""
    env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, None, overrides=["presets=isaacsim_physx,ovrtx"])
    with pytest.raises(ValueError) as excinfo:
        validate_launch(env_cfg)
    msg = str(excinfo.value)
    assert "PhysxCfg" in msg
    assert "IsaacRtxRendererCfg" in msg


@pytest.mark.parametrize("visualizer", ["kit", "kit,newton_gl"])
def test_kit_visualizer_plus_ovrtx_raises(visualizer):
    """Selecting Kit alone or with another viewer must reject the OVRTX renderer.

    Use Newton physics so the only Kit-side runtime is the visualizer; this
    isolates the visualizer-vs-renderer check from the physics-vs-renderer one.
    """
    env_cfg, _ = resolve_task_config(
        _CAMERA_PRESETS_TASK,
        None,
        overrides=["physics=newton_mjwarp", "renderer=ovrtx", f"visualizer={visualizer}"],
    )
    with pytest.raises(ValueError) as excinfo:
        validate_launch(env_cfg)
    msg = str(excinfo.value)
    assert "Kit visualizer" in msg
    assert "IsaacRtxRendererCfg" in msg


def test_kit_renderer_plus_ovrtx_raises():
    """A Kit renderer and OVRTX cannot initialize in the same process."""
    env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, None, overrides=["presets=newton,isaacsim_rtx"])
    mixed_cfg = argparse.Namespace(
        physics=env_cfg.sim.physics,
        kit_camera=env_cfg.scene.tiled_camera,
        ovrtx_renderer=OVRTXRendererCfg(),
    )

    with pytest.raises(ValueError, match="Kit-based renderer"):
        validate_launch(mixed_cfg)


@pytest.mark.parametrize(
    ("launcher_args", "source"),
    [
        (argparse.Namespace(experience="custom.kit"), "explicit Kit experience"),
        (argparse.Namespace(livestream=2), "livestreaming"),
    ],
)
def test_launcher_kit_source_plus_ovrtx_raises(launcher_args, source):
    """Every launcher-side Kit source must conflict with OVRTX."""
    env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, None, overrides=["presets=newton,ovrtx"])

    with pytest.raises(ValueError, match=source):
        validate_launch(env_cfg, launcher_args)


def test_default_kit_runtime_plus_ovrtx_raises(monkeypatch: pytest.MonkeyPatch):
    """A config without physics defaults to Kit, which cannot share OVRTX."""
    monkeypatch.delenv("LIVESTREAM", raising=False)
    renderer_only_cfg = argparse.Namespace(renderer=OVRTXRendererCfg())

    with pytest.raises(ValueError, match="default Isaac Sim / Kit runtime"):
        validate_launch(renderer_only_cfg)


# ---------------------------------------------------------------------------
# Invalid: OvPhysX physics + Isaac Sim / Kit
# ---------------------------------------------------------------------------


def test_ovphysx_plus_kit_visualizer_raises():
    """OvPhysX cannot share a process with the Kit visualizer."""
    env_cfg, _ = resolve_task_config(
        _CAMERA_PRESETS_TASK,
        None,
        overrides=["physics=ovphysx", "renderer=isaacsim_rtx", "visualizer=kit"],
    )
    with pytest.raises(ValueError) as excinfo:
        validate_launch(env_cfg)
    msg = str(excinfo.value)
    assert "OvPhysX" in msg
    assert "Kit visualizer" in msg


def test_ovphysx_plus_kit_physics_raises():
    """Two physics configs cannot pull OvPhysX and Kit into the same process."""
    mixed_cfg = argparse.Namespace(
        ovphysx_physics=OvPhysxCfg(),
        kit_physics=PhysxCfg(),
    )

    with pytest.raises(ValueError, match="PhysxCfg"):
        validate_launch(mixed_cfg)


def test_explicit_kit_experience_plus_ovphysx_raises():
    """An explicit Kit experience must conflict with OvPhysX."""
    env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, None, overrides=["presets=ovphysx,ovrtx"])
    launcher_args = argparse.Namespace(experience="custom.kit")

    with pytest.raises(ValueError, match="explicit Kit experience"):
        validate_launch(env_cfg, launcher_args)


def test_ovphysx_plus_kit_camera_without_visualizer_raises():
    """A Kit-based renderer pulls in Kit even with no visualizer, which OvPhysX cannot share.

    Without a visualizer the only Kit signal is the camera, so this is the case the visualizer-only
    guard missed: it previously reached OvPhysX's own initialization and failed there instead.
    """
    env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, None, overrides=["presets=ovphysx,isaacsim_rtx"])
    with pytest.raises(ValueError) as excinfo:
        validate_launch(env_cfg, argparse.Namespace(visualizer=None))
    msg = str(excinfo.value)
    assert "OvPhysX" in msg
    assert "renderer" in msg


# ---------------------------------------------------------------------------
# Valid combinations: must NOT raise
# ---------------------------------------------------------------------------


def test_default_newton_plus_ovrtx_is_valid():
    """The default Newton backend supports the default OVRTX renderer."""
    env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, None, overrides=["presets=ovrtx"])

    assert isinstance(env_cfg.sim.physics, NewtonCfg)
    validate_launch(env_cfg)

    assert isinstance(env_cfg.scene.tiled_camera.renderer_cfg, OVRTXRendererCfg)


def test_explicit_auto_physx_plus_ovrtx_resolves_to_ovphysx():
    """The ``physx`` preset remains automatic when paired with OVRTX."""
    env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, None, overrides=["presets=physx,ovrtx"])

    assert isinstance(env_cfg.sim.physics, PhysxAutoCfg)

    validate_launch(env_cfg)

    assert isinstance(env_cfg.sim.physics, OvPhysxCfg)
    assert isinstance(env_cfg.scene.tiled_camera.renderer_cfg, OVRTXRendererCfg)


def test_physx_plus_isaacsim_rtx_is_valid():
    """PhysX physics + Isaac RTX renderer is the supported Kit combination."""
    env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, None, overrides=["presets=physx,isaacsim_rtx"])
    validate_launch(env_cfg)

    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    assert isinstance(env_cfg.scene.tiled_camera.renderer_cfg, IsaacRtxRendererCfg)


def test_auto_physx_configured_kit_visualizer_resolves_to_isaac_sim_backends():
    """Config-declared Kit visualizers should drive automatic PhysX and RTX resolution."""

    env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, None, overrides=["physics=physx", "renderer=rtx"])
    env_cfg.sim.visualizer_cfgs = VisualizerCfg(backend=KitBackendCfg())
    validate_launch(env_cfg)

    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    assert isinstance(env_cfg.scene.tiled_camera.renderer_cfg, IsaacRtxRendererCfg)


def test_auto_physx_livestream_without_launcher_args_resolves_to_isaac_sim_backends(
    monkeypatch: pytest.MonkeyPatch,
):
    """Livestream env vars should require Kit even when no launcher args object is provided."""
    env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, None, overrides=["physics=physx", "renderer=rtx"])
    monkeypatch.setenv("LIVESTREAM", "2")
    validate_launch(env_cfg)

    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    assert isinstance(env_cfg.scene.tiled_camera.renderer_cfg, IsaacRtxRendererCfg)


def test_auto_physx_explicit_experience_resolves_to_isaac_sim_backends():
    """An explicit Kit experience should drive automatic PhysX and RTX resolution."""
    env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, None, overrides=["physics=physx", "renderer=rtx"])
    validate_launch(env_cfg, argparse.Namespace(experience="custom.kit"))

    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    assert isinstance(env_cfg.scene.tiled_camera.renderer_cfg, IsaacRtxRendererCfg)


def test_default_preset_is_valid():
    """The default preset (Newton + Newton renderer) is supported."""
    env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, None, overrides=["presets=default"])
    validate_launch(env_cfg)


def test_rtx_with_default_newton_is_valid_and_resolves_to_ovrtx():
    """The RTX selector resolves to OVRTX with the default Newton backend."""
    env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, None, overrides=["presets=rtx"])
    validate_launch(env_cfg)

    assert isinstance(env_cfg.sim.physics, NewtonCfg)
    assert isinstance(env_cfg.scene.tiled_camera.renderer_cfg, OVRTXRendererCfg)


def test_renderer_selector_physx_rtx_is_valid_and_resolves_to_ovphysx_and_ovrtx():
    """The automatic PhysX and RTX selectors choose kitless backends without Kit signals."""
    env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, None, overrides=["physics=physx", "renderer=rtx"])

    assert isinstance(env_cfg.sim.physics, PhysxAutoCfg)

    validate_launch(env_cfg)

    assert isinstance(env_cfg.sim.physics, OvPhysxCfg)
    assert isinstance(env_cfg.scene.tiled_camera.renderer_cfg, OVRTXRendererCfg)


def test_renderer_selector_physx_rtx_with_kit_visualizer_resolves_to_isaac_sim_backends():
    """The automatic PhysX and RTX selectors choose Isaac Sim backends when the Kit viewer is requested."""
    env_cfg, _ = resolve_task_config(
        _CAMERA_PRESETS_TASK, None, overrides=["physics=physx", "renderer=rtx", "visualizer=kit"]
    )
    validate_launch(env_cfg)

    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    assert isinstance(env_cfg.scene.tiled_camera.renderer_cfg, IsaacRtxRendererCfg)


def test_rtx_with_ovphysx_is_valid_and_resolves_to_ovrtx():
    """The RTX preset chooses OVRTX for an OvPhysX kitless run."""
    env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, None, overrides=["presets=ovphysx,rtx"])
    validate_launch(env_cfg)

    assert isinstance(env_cfg.scene.tiled_camera.renderer_cfg, OVRTXRendererCfg)


@pytest.mark.parametrize(
    "presets, expected_renderer",
    [("physx,rtx", OVRTXRendererCfg), ("isaacsim_physx,rtx", IsaacRtxRendererCfg)],
)
def test_repeated_launch_preserves_selected_configs_and_aliases(presets, expected_renderer):
    """Replace shared renderer references without rebuilding unrelated named tuples."""
    env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, None, overrides=[f"presets={presets}"])
    assembly = namedtuple("Assembly", "socket")("fixture")
    tree = {"env": env_cfg, "renderers": (env_cfg.scene.tiled_camera.renderer_cfg,), "assemblies": (assembly,)}
    args = argparse.Namespace()
    configs, _ = validate_launch(tree, args)
    physics, renderer = env_cfg.sim.physics, env_cfg.scene.tiled_camera.renderer_cfg
    assert isinstance(renderer, expected_renderer)
    assert sum(node is renderer for node in configs) == 1
    assert sum(node is physics for node in configs) == 1
    assert tree["renderers"][0] is renderer
    assert tree["assemblies"][0] is assembly
    validate_launch(tree, args)
    assert env_cfg.sim.physics is physics and env_cfg.scene.tiled_camera.renderer_cfg is renderer


def test_rtx_with_kit_visualizer_is_valid_and_resolves_to_isaac_rtx():
    """The RTX preset chooses Isaac RTX when the Kit visualizer is requested."""
    env_cfg, _ = resolve_task_config(
        _CAMERA_PRESETS_TASK,
        None,
        overrides=["physics=newton_mjwarp", "renderer=rtx", "visualizer=kit"],
    )
    validate_launch(env_cfg)

    assert isinstance(env_cfg.scene.tiled_camera.renderer_cfg, IsaacRtxRendererCfg)


def test_livestream_rtx_injects_kit_before_auto_rtx_resolution(monkeypatch: pytest.MonkeyPatch):
    """Livestreaming should make ``presets=newton_mjwarp,rtx`` choose Isaac RTX."""
    env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, None, overrides=["presets=newton_mjwarp,rtx"])
    launcher_args = argparse.Namespace(livestream=2, visualizer=None)
    received = {}

    class KitRuntime(SimulationLauncher):
        def __init__(self, args):
            received.update(args)

    monkeypatch.setattr(physx_app, "KitLauncher", KitRuntime)

    with launch_simulation(env_cfg, launcher_args) as physics_cfg:
        assert type(physics_cfg).__name__ == "NewtonCfg"

    assert isinstance(env_cfg.sim.visualizer_cfgs[0].backend, KitBackendCfg)
    assert received["enable_cameras"] is True
    assert launcher_args.visualizer is None
    assert isinstance(env_cfg.scene.tiled_camera.renderer_cfg, IsaacRtxRendererCfg)


def test_kit_visualizer_with_isaacsim_rtx_is_valid():
    """``--visualizer kit`` is fine as long as no OVRTX renderer is configured."""
    env_cfg, _ = resolve_task_config(
        _CAMERA_PRESETS_TASK,
        None,
        overrides=["physics=newton_mjwarp", "renderer=isaacsim_rtx", "visualizer=kit"],
    )
    validate_launch(env_cfg)
