# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Preset composition selects concrete viewers before any process runtime starts."""

import argparse
import importlib.util

import isaaclab_ov.app as ov_app
import isaaclab_physx.app as physx_app
import pytest
from isaaclab_newton.physics import NewtonCfg
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

import isaaclab.app.sim_launcher as sim_launcher
from isaaclab.app import SimulationLauncher
from isaaclab.sim import SimulationCfg
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import resolve_presets, resolve_task_config, setup_preset_cli
from isaaclab_tasks.utils.presets import MultiBackendVisualizerCfg


@pytest.mark.parametrize("selection", ["kit", "newton_gl", "newton_rtx", "rerun", "viser", "none", "kit,newton_gl"])
def test_visualizer_aliases_compose_the_same_task(selection, monkeypatch):
    """All selector spellings compose cameras and viewers before runtime selection."""
    monkeypatch.setattr(physx_app.KitLauncher, "__init__", SimulationLauncher.__init__)
    monkeypatch.setattr(physx_app.KitLauncher, "close", SimulationLauncher.close)
    monkeypatch.setattr(ov_app, "OvrtxLauncher", SimulationLauncher)
    options = (
        ["physics=newton_mjwarp", "renderer=newton_renderer", f"visualizer={selection}"],
        ["--physics", "newton_mjwarp", "--renderer", "newton_renderer", "--viz", selection],
        ["--physics=newton_mjwarp", "--renderer=newton_renderer", "--visualizer", selection],
    )
    for option in options:
        parser = argparse.ArgumentParser()
        sim_launcher.add_launcher_args(parser)
        args, remaining = setup_preset_cli(parser, option)
        cfg, _ = resolve_task_config("Isaac-Cartpole-Camera-Direct", "", overrides=remaining)
        physics, renderer = cfg.sim.physics, cfg.scene.tiled_camera.renderer_cfg
        selected = cfg.sim.visualizer_cfgs
        selected = selected if isinstance(selected, list) else [selected]
        expected = [] if selection == "none" else selection.split(",")
        assert not {"physics", "renderer", "visualizer"}.intersection(vars(args))
        with sim_launcher.launch_simulation(cfg, args):
            assert [viewer.backend.visualizer_type for viewer in cfg.sim.visualizer_cfgs] == expected
            assert all(viewer.eye == (20.0, 20.0, 20.0) for viewer in cfg.sim.visualizer_cfgs)
        assert cfg.sim.physics is physics and isinstance(physics, NewtonCfg)
        assert cfg.scene.tiled_camera.renderer_cfg is renderer and renderer.renderer_type == "newton_warp"
        if selection == "none":
            with pytest.raises(ValueError, match="Livestreaming requires the Kit visualizer"):
                with sim_launcher.launch_simulation(cfg, {"livestream": 2, "visualizer": []}):
                    pass


def test_visualizer_preset_preserves_custom_settings_and_validates_selection():
    """Preset selection preserves declared settings; physics aliases do not also select a viewer."""
    assert importlib.util.find_spec("isaaclab.utils.presets") is None
    assert importlib.util.find_spec("isaaclab_visualizers.presets") is None
    assert "resolve_simulation_cfg" not in vars(sim_launcher)

    cfg = SimulationCfg(
        physics=NewtonCfg(),
        device="cpu",
        visualizer_cfgs=VisualizerCfg(max_visible_envs=16, backend=MultiBackendVisualizerCfg()),
    )
    viewer = cfg.visualizer_cfgs
    assert resolve_presets(cfg, overrides={"visualizer_cfgs.backend": "newton_gl"}) is cfg
    assert cfg.visualizer_cfgs is viewer
    assert isinstance(viewer.backend, NewtonGLVisualizerCfg)
    assert viewer.max_visible_envs == 16
    with sim_launcher.launch_simulation(cfg, {"max_visible_envs": 3}):
        assert cfg.visualizer_cfgs == [viewer]
        assert viewer.max_visible_envs == 3
    with pytest.raises(ValueError, match="Unknown preset"):
        resolve_presets(MultiBackendVisualizerCfg(), overrides={"": "missing"})
    with pytest.warns(FutureWarning, match="newton_gl"):
        viewer = resolve_presets(MultiBackendVisualizerCfg(), overrides={"": "newton"})
    assert isinstance(viewer, NewtonGLVisualizerCfg)
    assert resolve_presets(MultiBackendVisualizerCfg(), selected=["newton"]) is None


def test_launch_simulation_uses_configured_viewers_and_releases_runtime(monkeypatch):
    """Runtime metadata belongs to the selected cfg, not settings or a caller-supplied intent dictionary."""
    launches, closed = [], []

    class KitRuntime(SimulationLauncher):
        def __init__(self, launcher_args):
            launches.append(launcher_args)

        def close(self, exit_code=0):
            closed.append(exit_code)

    monkeypatch.setattr(physx_app, "KitLauncher", KitRuntime)
    cfg = SimulationCfg(
        physics=NewtonCfg(),
        visualizer_cfgs=[VisualizerCfg(backend=KitVisualizerCfg()), VisualizerCfg(backend=NewtonGLVisualizerCfg())],
    )
    args = argparse.Namespace(max_visible_envs=3)
    with sim_launcher.launch_simulation(cfg, args):
        assert launches == [vars(args)]
        assert args.kit_visualizer
        assert all(viewer.max_visible_envs == 3 for viewer in cfg.visualizer_cfgs)
    assert closed == [0]

    monkeypatch.setattr(sim_launcher, "resolve_python_logging_level", lambda _: 42)
    levels = []
    monkeypatch.setattr(sim_launcher, "apply_python_logging_level", levels.append)
    with sim_launcher.launch_simulation(SimulationCfg(physics=NewtonCfg()), {}):
        pass
    assert levels == [42]
    assert len(launches) == 1
