# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Preset composition selects concrete viewers before any process runtime starts."""

import argparse

import isaaclab_physx.app as physx_app
import pytest
from isaaclab_newton.physics import NewtonCfg
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
from isaaclab_visualizers.presets import MultiBackendVisualizerCfg

import isaaclab.app.sim_launcher as sim_launcher
from isaaclab.app import SimulationLauncher
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, resolve_presets

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli


@pytest.mark.parametrize("selection", ["kit", "newton_gl", "newton_rtx", "rerun", "viser", "none", "kit,newton_gl"])
def test_visualizer_aliases_compose_the_same_task(selection):
    """All selector spellings compose cameras and viewers before runtime selection."""
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
        assert [viewer.visualizer_type for viewer in selected] == expected
        assert not {"physics", "renderer", "visualizer"}.intersection(vars(args))
        assert sim_launcher.resolve_simulation_cfg(cfg, args) is cfg
        assert cfg.sim.physics is physics and isinstance(physics, NewtonCfg)
        assert cfg.scene.tiled_camera.renderer_cfg is renderer and renderer.renderer_type == "newton_warp"
        if selection == "none":
            with pytest.raises(ValueError, match="Livestreaming requires the Kit visualizer"):
                sim_launcher.resolve_simulation_cfg(cfg, {"livestream": 2})


def test_visualizer_preset_preserves_custom_settings_and_validates_selection():
    """Preset selection preserves declared settings; physics aliases do not also select a viewer."""

    @configclass
    class Viewers(MultiBackendVisualizerCfg):
        kit = KitVisualizerCfg(streaming_view=True)
        newton_gl = NewtonGLVisualizerCfg(max_visible_envs=2)

    cfg = SimulationCfg(physics=NewtonCfg(), visualizer_cfgs=Viewers())
    viewer = cfg.visualizer_cfgs.newton_gl
    args = {"visualizer": "newton_gl", "max_visible_envs": 3}
    result = sim_launcher.resolve_simulation_cfg(cfg, args)
    assert args == {"visualizer": "newton_gl", "max_visible_envs": 3}
    assert result.visualizer_cfgs.max_visible_envs == 3
    assert result.visualizer_cfgs is viewer
    assert sim_launcher.resolve_simulation_cfg(cfg, args).visualizer_cfgs is result.visualizer_cfgs
    with pytest.raises(ValueError, match="Unknown preset"):
        sim_launcher.resolve_simulation_cfg(SimulationCfg(), {"visualizer": "missing"})
    with pytest.warns(FutureWarning, match="newton_gl"):
        cfg = sim_launcher.resolve_simulation_cfg(SimulationCfg(physics=NewtonCfg()), {"visualizer": "newton"})
    assert isinstance(cfg.visualizer_cfgs, NewtonGLVisualizerCfg)
    assert resolve_presets(SimulationCfg(), selected=["newton"]).visualizer_cfgs is None


def test_launch_simulation_uses_configured_viewers_and_releases_runtime(monkeypatch):
    """Runtime metadata belongs to the selected cfg, not settings or a caller-supplied intent dictionary."""
    launches, closed = [], []

    class KitRuntime(SimulationLauncher):
        def __init__(self, launcher_args):
            launches.append(launcher_args)

        def close(self, exit_code=0):
            closed.append(exit_code)

    monkeypatch.setattr(physx_app, "KitLauncher", KitRuntime)
    cfg = SimulationCfg(physics=NewtonCfg(), visualizer_cfgs=[KitVisualizerCfg(), NewtonGLVisualizerCfg()])
    args = argparse.Namespace(max_visible_envs=3)
    with sim_launcher.launch_simulation(cfg, args):
        assert launches == [vars(args)]
        assert args.kit_visualizer
        assert all(viewer.max_visible_envs == 3 for viewer in cfg.visualizer_cfgs)
    assert closed == [0]

    monkeypatch.setattr(sim_launcher, "_resolve_python_logging_level", lambda _: 42)
    levels = []
    monkeypatch.setattr(sim_launcher, "apply_python_logging_level", levels.append)
    with sim_launcher.launch_simulation(SimulationCfg(physics=NewtonCfg()), {}):
        pass
    assert levels == [42]
    assert len(launches) == 1
