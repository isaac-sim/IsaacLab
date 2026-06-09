# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration-style tests for visualizer intent plumbing in sim launcher."""

from __future__ import annotations

import argparse
import sys
import types

import isaaclab.app.sim_launcher as sim_launcher


def _force_kitless(monkeypatch):
    """Wrap ``scan`` so the resulting scan reports ``needs_kit=False``."""
    real_scan = sim_launcher.scan

    def fake_scan(*args, **kwargs):
        result = real_scan(*args, **kwargs)
        result.needs_kit = False
        return result

    monkeypatch.setattr(sim_launcher, "scan", fake_scan)


class _DummyVizCfg:
    def __init__(self, visualizer_type: str):
        self.visualizer_type = visualizer_type


class _DummySimCfg:
    def __init__(self, visualizer_cfgs):
        self.visualizer_cfgs = visualizer_cfgs


class _DummyEnvCfg:
    def __init__(self, sim_cfg):
        self.sim = sim_cfg


def test_launch_simulation_passes_visualizer_intent_to_applauncher(monkeypatch):
    """Ensure canonical launcher path forwards visualizer intent upstream."""
    captured: dict[str, object] = {}

    class _FakeAppLauncher:
        def __init__(self, launcher_args):
            captured["launcher_args"] = launcher_args
            captured["closed"] = False
            self.app = types.SimpleNamespace(close=lambda: captured.update({"closed": True}))

    monkeypatch.setitem(sys.modules, "isaaclab.app", types.SimpleNamespace(AppLauncher=_FakeAppLauncher))
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name == "omni.kit" else None)

    env_cfg = _DummyEnvCfg(_DummySimCfg([_DummyVizCfg("kit"), _DummyVizCfg("newton")]))
    launcher_args = argparse.Namespace()

    with sim_launcher.launch_simulation(env_cfg, launcher_args):
        pass

    forwarded_args = captured["launcher_args"]
    assert isinstance(forwarded_args, argparse.Namespace)
    assert getattr(forwarded_args, "visualizer_intent") == {
        "has_any_visualizers": True,
        "has_kit_visualizer": True,
    }
    assert captured["closed"] is True


def test_launch_simulation_kitless_viz_none_sets_disable_all(monkeypatch):
    """Kitless mode should persist explicit disable-all semantics for --viz none."""
    captured = {"types": None, "explicit": None, "disable_all": None}

    class _FakeAppLauncher:
        @staticmethod
        def sync_visualizer_cli_settings_to_carb(launcher_args: dict) -> None:
            captured["types"] = " ".join(launcher_args["visualizer"]) if launcher_args.get("visualizer") else ""
            captured["explicit"] = launcher_args["visualizer_explicit"]
            captured["disable_all"] = launcher_args["visualizer_disable_all"]

    _force_kitless(monkeypatch)
    monkeypatch.setitem(sys.modules, "isaaclab.app", types.SimpleNamespace(AppLauncher=_FakeAppLauncher))

    env_cfg = _DummyEnvCfg(_DummySimCfg(None))
    launcher_args = argparse.Namespace(visualizer=None, visualizer_explicit=True)
    with sim_launcher.launch_simulation(env_cfg, launcher_args):
        pass

    assert captured == {"types": "", "explicit": True, "disable_all": True}
