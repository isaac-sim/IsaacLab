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

import isaaclab.app.sim_launcher as sim_launcher
from isaaclab.app import SimulationLauncher, get_settings_manager
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
    monkeypatch.setattr(sim_launcher, "configure_storage_profile", lambda: None)
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


def test_cli_visualizer_selection_drops_configured_kit_streaming_view():
    """A Kit visualizer that ``--viz`` drops does not request its streaming view."""
    kit_cfg = VisualizerCfg(visualizer_type="kit", streaming_view=True)
    cfg = argparse.Namespace(physics=NewtonCfg(), visualizer_cfgs=[kit_cfg])

    assert sim_launcher.scan(cfg, {"visualizer": ["rerun"]}).visualizer_intent["has_kit_streaming_view"] is False


def test_cli_visualizer_selection_drops_configured_newton_rtx(kit_launcher_args, monkeypatch):
    """``--viz kit`` drops a configured newton_rtx visualizer, so OVRTX neither conflicts with Kit nor starts."""
    started = []
    real_string_to_callable = sim_launcher.string_to_callable
    monkeypatch.setattr(
        sim_launcher, "string_to_callable", lambda name: started.append(name) or real_string_to_callable(name)
    )
    cfg = argparse.Namespace(physics=NewtonCfg(), visualizer_cfgs=[VisualizerCfg(visualizer_type="newton_rtx")])

    with sim_launcher.launch_simulation(cfg, {"visualizer": ["kit"]}):
        pass

    assert len(kit_launcher_args) == 1
    assert sim_launcher.OVRTXRendererCfg.launcher_type not in started
    assert [cfg.visualizer_type for cfg in cfg.visualizer_cfgs] == ["kit"]


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


@pytest.mark.parametrize(
    "visualizer, expected_types",
    [(None, ["kit", "newton_gl"]), (["none"], []), (["newton_gl", "rerun"], ["newton_gl", "rerun"])],
    ids=["config", "none", "explicit"],
)
def test_launch_simulation_resolves_visualizers_into_config(kit_launcher_args, visualizer, expected_types):
    """The launch writes the run's visualizers into the config; an explicit selection keeps configured settings."""
    newton_cfg = VisualizerCfg(visualizer_type="newton_gl", max_visible_envs=7)
    sim_cfg = _DummySimCfg([VisualizerCfg(visualizer_type="kit"), newton_cfg])
    sim_cfg.physics = NewtonCfg()

    with sim_launcher.launch_simulation(_DummyEnvCfg(sim_cfg), {"visualizer": visualizer, "require_kit": True}):
        pass

    assert [cfg.visualizer_type for cfg in sim_cfg.visualizer_cfgs] == expected_types
    if "newton_gl" in expected_types:
        assert sim_cfg.visualizer_cfgs[expected_types.index("newton_gl")] is newton_cfg
    # nothing is left in the settings for a later SimulationCfg to re-resolve
    assert get_settings_manager().get("/isaaclab/visualizer/types") == ""


def test_launch_simulation_leaves_selection_for_a_config_built_after_launch(kit_launcher_args):
    """Without a SimulationCfg to write into, the selection reaches the SimulationContext through the settings."""
    settings = get_settings_manager()

    with sim_launcher.launch_simulation(NewtonCfg(), {"visualizer": ["none"], "require_kit": True}):
        assert settings.get("/isaaclab/visualizer/types") == "none"
    with sim_launcher.launch_simulation(_DummyEnvCfg(_DummySimCfg([])), {"require_kit": True}):
        assert settings.get("/isaaclab/visualizer/types") == ""
    settings.set("/isaaclab/visualizer/max_visible_envs", -1)


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
