# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration-style tests for visualizer intent plumbing in sim launcher."""

from __future__ import annotations

import argparse
import logging
import sys
import types

import isaaclab_physx.app as physx_app
import pytest
from isaaclab_newton.physics import NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

import isaaclab.app.sim_launcher as sim_launcher
from isaaclab.app import SimulationLauncher, get_settings_manager
from isaaclab.sim import SimulationCfg
from isaaclab.visualizers import VisualizerCfg


def _force_kitless(monkeypatch):
    """Wrap ``scan`` so the resulting scan reports ``needs_kit=False``."""
    real_scan = sim_launcher.scan

    def fake_scan(*args, **kwargs):
        result = real_scan(*args, **kwargs)
        result.needs_kit = False
        return result

    monkeypatch.setattr(sim_launcher, "scan", fake_scan)


class _DummySimCfg:
    def __init__(self, visualizer_cfgs):
        self.visualizer_cfgs = visualizer_cfgs


class _DummyEnvCfg:
    def __init__(self, sim_cfg):
        self.sim = sim_cfg


def test_launch_simulation_passes_kit_visualizer_to_kit_launcher(monkeypatch):
    """Ensure canonical launcher path forwards the resolved Kit visualizer request."""
    captured: dict[str, object] = {}

    class _FakeKitLauncher:
        device = None

        def __init__(self, launcher_args):
            captured["launcher_args"] = launcher_args
            captured["closed"] = False

        def close(self, exit_code=0):
            captured["closed"] = True

    monkeypatch.setitem(sys.modules, "isaaclab.utils", types.SimpleNamespace(has_kit=lambda: False))
    monkeypatch.setitem(
        sys.modules,
        "isaaclab.utils.assets",
        types.SimpleNamespace(configure_storage_profile=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "isaaclab_physx.app", types.SimpleNamespace(KitLauncher=_FakeKitLauncher))

    env_cfg = _DummyEnvCfg(
        _DummySimCfg([VisualizerCfg(visualizer_type="kit"), VisualizerCfg(visualizer_type="newton_gl")])
    )
    launcher_args = argparse.Namespace()

    with sim_launcher.launch_simulation(env_cfg, launcher_args):
        pass

    # the launcher gets the caller's namespace as a dict, so resolved values reach both
    assert captured["launcher_args"]["kit_visualizer"] is True
    assert launcher_args.kit_visualizer is True
    assert captured["closed"] is True


@pytest.fixture
def kit_launcher_args(monkeypatch) -> list:
    """Record the launcher args of each Kit launch, without starting Kit."""
    launches = []

    class _FakeKitLauncher(SimulationLauncher):
        def __init__(self, launcher_args):
            launches.append(launcher_args)

    monkeypatch.setattr(physx_app, "KitLauncher", _FakeKitLauncher)
    return launches


@pytest.mark.parametrize(
    "default_visualizer_cfg",
    [
        KitVisualizerCfg(streaming_view=True),
        NewtonRTXVisualizerCfg(),
        VisualizerCfg(streaming_cam_renderer_cfg=OVRTXRendererCfg()),
    ],
    ids=["kit", "newton_rtx", "nested_renderer"],
)
def test_default_visualizer_settings_do_not_launch_runtime(monkeypatch, default_visualizer_cfg):
    """Viewer defaults must not start a runtime when no visualizer is selected."""
    runtime_launches = []

    def resolve_launcher(launcher_type):
        runtime_launches.append(launcher_type)
        return SimulationLauncher

    monkeypatch.setattr(sim_launcher, "string_to_callable", resolve_launcher)
    cfg = SimulationCfg(physics=NewtonCfg(), default_visualizer_cfg=default_visualizer_cfg)

    with sim_launcher.launch_simulation(cfg):
        pass

    assert runtime_launches == []


def test_default_kit_streaming_view_enables_cameras_when_kit_is_selected(kit_launcher_args):
    """Selecting Kit must retain the camera requirements of its shared streaming settings."""
    cfg = SimulationCfg(physics=NewtonCfg(), default_visualizer_cfg=KitVisualizerCfg(streaming_view=True))

    with sim_launcher.launch_simulation(cfg, {"visualizer": ["kit"]}):
        pass

    assert kit_launcher_args[0]["enable_cameras"] is True


def test_caller_visualizer_intent_requests_kit_visualizer(kit_launcher_args):
    """A caller's ``visualizer_intent`` requests the Kit visualizer even without a visualizer config."""
    with sim_launcher.launch_simulation(None, {"require_kit": True, "visualizer_intent": {"has_kit_visualizer": True}}):
        pass

    assert kit_launcher_args[0]["kit_visualizer"] is True


@pytest.mark.parametrize("physics_cfg", [NewtonCfg(), OvPhysxCfg()], ids=["newton", "ovphysx"])
def test_cli_visualizer_selection_overrides_config_kit_visualizer(kit_launcher_args, physics_cfg):
    """``--viz rerun`` drops a configured Kit visualizer, so the run stays kitless."""
    cfg = argparse.Namespace(physics=physics_cfg, visualizer_cfgs=[VisualizerCfg(visualizer_type="kit")])
    launcher_args = argparse.Namespace(visualizer=["rerun"])

    with sim_launcher.launch_simulation(cfg, launcher_args):
        pass

    assert kit_launcher_args == []
    assert launcher_args.kit_visualizer is False


@pytest.mark.parametrize("require_kit", [False, True], ids=["kitless", "kit"])
def test_launch_simulation_writes_max_visible_envs_without_visualizer(kit_launcher_args, require_kit):
    """``--max_visible_envs`` reaches the settings with or without Kit, even when no visualizer is selected."""
    settings = get_settings_manager()
    settings.set("/isaaclab/visualizer/max_visible_envs", -1)

    with sim_launcher.launch_simulation(NewtonCfg(), {"require_kit": require_kit, "max_visible_envs": 3}):
        pass

    assert len(kit_launcher_args) == int(require_kit)
    assert settings.get("/isaaclab/visualizer/max_visible_envs") == 3
    settings.set("/isaaclab/visualizer/max_visible_envs", -1)


def test_launch_simulation_kitless_viz_none_sets_disable_all(monkeypatch):
    """Kitless mode should persist explicit disable-all semantics for --viz none."""
    captured = {"types": None, "explicit": None, "disable_all": None}

    def fake_sync(launcher_args: dict) -> None:
        captured["types"] = " ".join(launcher_args["visualizer"]) if launcher_args.get("visualizer") else ""
        captured["explicit"] = launcher_args["visualizer"] is not None
        captured["disable_all"] = launcher_args["visualizer"] == []

    _force_kitless(monkeypatch)
    monkeypatch.setattr(sim_launcher, "_sync_visualizer_cli_settings", fake_sync)

    env_cfg = _DummyEnvCfg(_DummySimCfg(None))
    launcher_args = argparse.Namespace(visualizer=["none"])
    with sim_launcher.launch_simulation(env_cfg, launcher_args):
        pass

    assert captured == {"types": "", "explicit": True, "disable_all": True}


def test_launch_simulation_kitless_applies_python_logging_level(monkeypatch):
    """Kitless mode should apply the resolved Python logging level before yielding."""
    captured: dict[str, object] = {}

    def fake_apply(level):
        captured["applied_level"] = level

    _force_kitless(monkeypatch)
    monkeypatch.setattr(sim_launcher, "apply_python_logging_level", fake_apply)

    env_cfg = _DummyEnvCfg(_DummySimCfg(None))
    launcher_args = argparse.Namespace(visualizer=["none"], verbose=True)
    with sim_launcher.launch_simulation(env_cfg, launcher_args):
        pass

    assert captured["applied_level"] == logging.DEBUG
