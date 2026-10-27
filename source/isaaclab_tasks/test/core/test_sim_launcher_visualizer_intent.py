# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Preset composition selects concrete viewers before any process runtime starts."""

import argparse
from types import SimpleNamespace

import isaaclab_ov.app as ov_app
import isaaclab_physx.app as physx_app
import pytest
from isaaclab_newton.physics import NewtonCfg
from isaaclab_visualizers.kit import KitBackendCfg
from isaaclab_visualizers.newton import NewtonGLBackendCfg, NewtonRTXBackendCfg

import isaaclab.app.sim_launcher as sim_launcher
from isaaclab.app import SimulationLauncher
from isaaclab.sim import SimulationCfg
from isaaclab.visualizers import VisualizerCfg, WindowCfg

import isaaclab_tasks.utils as task_utils
from isaaclab_tasks.utils import resolve_presets, resolve_task_config, setup_preset_cli
from isaaclab_tasks.utils.hydra import register_task
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


def test_visualizer_preset_preserves_custom_settings_and_validates_selection(monkeypatch):
    """Preset selection preserves declared settings; physics aliases do not also select a viewer."""
    assert not hasattr(task_utils, "launch_simulation")
    for task in (
        "Isaac-Reorient-Cube-Allegro",
        "Isaac-Reorient-Cube-Allegro-Direct",
        "Isaac-Reorient-Cube-Shadow-Camera-Direct",
    ):
        cfg, _ = resolve_task_config(task, "", play_mode=True, overrides=["visualizer=newton_gl"])
        assert [viewer.backend.visualizer_type for viewer in cfg.sim.visualizer_cfgs] == ["newton_gl"]
        assert all(viewer.focal_length == 35.0 for viewer in cfg.sim.visualizer_cfgs)
    cfg = SimulationCfg(
        physics=NewtonCfg(),
        device="cpu",
        visualizer_cfgs=VisualizerCfg(max_visible_envs=16, backend=MultiBackendVisualizerCfg()),
    )
    viewer = cfg.visualizer_cfgs
    viewer.backend.newton_gl.particle_color = (0.1, 0.2, 0.3)
    args = argparse.Namespace(visualizer="newton_gl", max_visible_envs=3)
    cfg = resolve_presets(cfg, overrides={"visualizer_cfgs.backend": args.visualizer})
    assert isinstance(viewer.backend, NewtonGLBackendCfg)
    with sim_launcher.launch_simulation(cfg, args):
        assert cfg.visualizer_cfgs == [viewer]
        assert viewer.backend.particle_color == (0.1, 0.2, 0.3)
        assert viewer.max_visible_envs == 3
    assert args.visualizer == "newton_gl"

    for selection in ("default", "none", []):
        cfg.visualizer_cfgs = VisualizerCfg(backend=MultiBackendVisualizerCfg())
        cfg = resolve_presets(cfg, overrides={"visualizer_cfgs.backend": selection})
        with sim_launcher.launch_simulation(cfg):
            assert cfg.visualizer_cfgs == []
    cfg.visualizer_cfgs = VisualizerCfg(backend=MultiBackendVisualizerCfg())
    cfg = resolve_presets(cfg, overrides={"visualizer_cfgs.backend": "none"})
    with pytest.raises(ValueError, match="Livestreaming requires the Kit visualizer"):
        with sim_launcher.launch_simulation(cfg, {"visualizer": "none", "livestream": 2}):
            pass

    with pytest.raises(ValueError, match="Unknown preset"):
        resolve_presets(MultiBackendVisualizerCfg(), overrides={"": "missing"})
    with pytest.warns(FutureWarning, match="newton_gl"):
        viewer = resolve_presets(MultiBackendVisualizerCfg(), overrides={"": "newton"})
    assert isinstance(viewer, NewtonGLBackendCfg)
    assert resolve_presets(MultiBackendVisualizerCfg(), selected=["newton"]) == []

    viewer = VisualizerCfg(max_visible_envs=3, backend=viewer)
    other = VisualizerCfg(max_visible_envs=5, backend=NewtonGLBackendCfg())
    windows = [WindowCfg(size=(640, 480)), WindowCfg(size=(800, 600))]
    for declaration, overrides, expected in (
        ([viewer, other], ["visualizer=newton_gl"], [viewer, other]),
        (
            VisualizerCfg(max_visible_envs=3, backend=[viewer.backend, KitBackendCfg()]),
            ["visualizer=newton_gl"],
            [viewer],
        ),
        (
            VisualizerCfg(max_visible_envs=3, backend=MultiBackendVisualizerCfg(newton_gl=viewer.backend)),
            ["visualizer=kit", "env.sim.visualizer_cfgs.backend=newton_gl"],
            [viewer],
        ),
        (
            [VisualizerCfg(window=window, backend=MultiBackendVisualizerCfg()) for window in windows],
            ["visualizer=newton_gl,newton_rtx"],
            [VisualizerCfg(window=window, backend=[NewtonGLBackendCfg(), NewtonRTXBackendCfg()]) for window in windows],
        ),
    ):
        env_cfg = SimpleNamespace(sim=SimulationCfg(physics=NewtonCfg(), visualizer_cfgs=declaration))
        monkeypatch.setattr("isaaclab_tasks.utils.parse_cfg.load_cfg_from_registry", lambda *_: env_cfg)
        resolved, _, _ = register_task("Test", None, overrides=overrides)
        selected = resolved.sim.visualizer_cfgs
        selected = selected if isinstance(selected, list) else [selected]
        assert selected == expected


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
        visualizer_cfgs=[VisualizerCfg(backend=KitBackendCfg()), VisualizerCfg(backend=NewtonGLBackendCfg())],
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
