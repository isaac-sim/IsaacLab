# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration-style tests for how the sim launcher selects and configures visualizers."""

from __future__ import annotations

import argparse
import logging

import isaaclab_physx.app as physx_app
import pytest
from isaaclab_newton.physics import NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_visualizers.rerun import RerunVisualizerCfg

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


class _DummyEnvCfg:
    def __init__(self, sim_cfg):
        self.sim = sim_cfg


@pytest.fixture
def kit_launcher_args(monkeypatch) -> list:
    """Record the launcher args of each Kit launch, without starting Kit."""
    launches = []

    class _FakeKitLauncher(SimulationLauncher):
        def __init__(self, launcher_args):
            launches.append(launcher_args)

    monkeypatch.setattr(physx_app, "KitLauncher", _FakeKitLauncher)
    return launches


def test_cli_visualizer_selection_drops_configured_newton_rtx(kit_launcher_args, monkeypatch):
    """``--viz kit`` drops a configured newton_rtx visualizer, so OVRTX neither conflicts with Kit nor starts."""
    started = []
    real_string_to_callable = sim_launcher.string_to_callable
    monkeypatch.setattr(
        sim_launcher, "string_to_callable", lambda name: started.append(name) or real_string_to_callable(name)
    )
    cfg = SimulationCfg(physics=NewtonCfg(), visualizer_cfgs=[VisualizerCfg(visualizer_type="newton_rtx")])

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
    settings.set("/isaaclab/visualizer/types", None)


@pytest.mark.parametrize(
    "physics_cfg, visualizer, expected_types",
    [
        (NewtonCfg(), None, []),
        (OvPhysxCfg(), None, []),
        (NewtonCfg(), ["kit"], ["kit"]),
        (NewtonCfg(), ["rerun"], ["rerun"]),
        (NewtonCfg(), ["newton_gl", "kit"], ["kit", "newton_gl"]),
    ],
    ids=["no-viz", "no-viz-ovphysx", "configured", "default", "configured-pair"],
)
def test_launch_simulation_resolves_visualizers_into_config(kit_launcher_args, physics_cfg, visualizer, expected_types):
    """The launch runs exactly the selected types, reusing a configured visualizer of each type with its settings.

    Only a selected Kit visualizer starts Kit, with cameras for its ``streaming_view``.
    """
    sim_cfg = SimulationCfg(
        physics=physics_cfg,
        visualizer_cfgs=[
            KitVisualizerCfg(streaming_view=True),
            VisualizerCfg(visualizer_type="newton_gl", max_visible_envs=7),
        ],
    )
    configured = {cfg.visualizer_type: cfg for cfg in sim_cfg.visualizer_cfgs}
    settings = get_settings_manager()
    # an empty selection left by an earlier launch in this process
    settings.set("/isaaclab/visualizer/types", "")

    with sim_launcher.launch_simulation(_DummyEnvCfg(sim_cfg), {"visualizer": visualizer}):
        pass

    assert [cfg.visualizer_type for cfg in sim_cfg.visualizer_cfgs] == expected_types
    for cfg in sim_cfg.visualizer_cfgs:
        assert cfg is configured.get(cfg.visualizer_type) or isinstance(cfg, RerunVisualizerCfg)
    assert [args["enable_cameras"] for args in kit_launcher_args] == [True] * ("kit" in expected_types)
    # the launch replaces the earlier selection with its visualizers, so its SimulationContext keeps them
    assert settings.get("/isaaclab/visualizer/types") == ",".join(expected_types)

    # without a SimulationCfg to write into, the selection, possibly empty, reaches the SimulationContext
    settings.set("/isaaclab/visualizer/types", None)
    with sim_launcher.launch_simulation(physics_cfg, {"visualizer": visualizer}):
        assert settings.get("/isaaclab/visualizer/types") == ",".join(visualizer or [])
    settings.set("/isaaclab/visualizer/types", None)
    settings.set("/isaaclab/visualizer/max_visible_envs", -1)


def test_launch_simulation_kitless_applies_python_logging_level(monkeypatch):
    """Kitless mode should apply the resolved Python logging level before yielding."""
    captured: dict[str, object] = {}

    def fake_apply(level):
        captured["applied_level"] = level

    _force_kitless(monkeypatch)
    monkeypatch.setattr(sim_launcher, "apply_python_logging_level", fake_apply)

    env_cfg = _DummyEnvCfg(SimulationCfg(visualizer_cfgs=None))
    launcher_args = argparse.Namespace(visualizer=None, verbose=True)
    with sim_launcher.launch_simulation(env_cfg, launcher_args):
        pass

    assert captured["applied_level"] == logging.DEBUG
