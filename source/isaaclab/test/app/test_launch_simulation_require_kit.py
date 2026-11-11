# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ``require_kit`` launcher argument read by :func:`isaaclab.app.launch_simulation`.

``require_kit`` lets a tool state that it needs Kit for a reason the config cannot express --
the URDF/MJCF converters set it because they reach a Kit-only importer extension whenever the
standalone importer wheel is absent. The override is additive: it can turn a kitless launch into
a Kit one, never the reverse.

Kit is never actually started here: the Kit launcher named by the config is faked, and constructing it
is the signal that the Kit branch was taken. No Kit/GPU required.
"""

import argparse
import contextlib
import signal
import sys
import types
import warnings

import isaaclab_physx.app as physx_app
import pytest
from isaaclab_visualizers.newton import NewtonGLBackendCfg

import isaaclab.app.sim_launcher as sim_launcher
from isaaclab.app import SimulationLauncher, launch_simulation
from isaaclab.physics import PhysicsCfg
from isaaclab.renderers import RendererCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.visualizers import VisualizerCfg


@pytest.fixture
def kit_branch_taken(monkeypatch: pytest.MonkeyPatch):
    """Report whether ``launch_simulation`` entered its Kit branch, without starting Kit."""
    taken: list[bool] = []

    class FakeKitLauncher(SimulationLauncher):
        def __init__(self, launcher_args):
            taken.append(True)

    monkeypatch.setattr(physx_app, "KitLauncher", FakeKitLauncher)
    return taken


@pytest.mark.parametrize(
    "launcher_args",
    [{"require_kit": True}, argparse.Namespace(require_kit=True), {"visualizer": ["kit"], "require_kit": False}],
    ids=["dict", "namespace", "false_preserves_kit_requirement"],
)
def test_require_kit_is_additive(kit_branch_taken, launcher_args):
    with launch_simulation(cfg=SimulationCfg(physics=PhysicsCfg(), device="cpu"), launcher_args=launcher_args):
        pass

    assert kit_branch_taken == [True]


def test_kit_rejects_ovrtx_before_loading_runtime(monkeypatch: pytest.MonkeyPatch):
    class CustomPhysxCfg(sim_launcher.PhysxCfg):
        pass

    calls = []
    monkeypatch.setitem(
        sys.modules, "ovrtx", types.SimpleNamespace(register_schema_paths=lambda: calls.append("register"))
    )
    monkeypatch.setattr(physx_app, "KitLauncher", lambda _: pytest.fail("Invalid runtime reached Kit launch"))

    for cfg, launcher_args in (
        (types.SimpleNamespace(physics=PhysicsCfg(), renderer=sim_launcher.OVRTXRendererCfg()), {"require_kit": True}),
        (
            SimulationCfg(
                physics=CustomPhysxCfg(),
                visualizer_cfgs=VisualizerCfg(backend=NewtonGLBackendCfg(initial_renderer="rtx")),
            ),
            {"visualizer": ["newton"]},
        ),
    ):
        with pytest.raises(ValueError, match="OVRTX runtime"):
            with launch_simulation(cfg, launcher_args):
                pass

    assert calls == []


def test_kitless_ovrtx_registers_before_user_code(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.setitem(
        sys.modules, "ovrtx", types.SimpleNamespace(register_schema_paths=lambda: calls.append("register"))
    )
    cfg = types.SimpleNamespace(
        sim=SimulationCfg(
            physics=sim_launcher.NewtonCfg(),
            device="cpu",
            visualizer_cfgs=VisualizerCfg(backend=NewtonGLBackendCfg(initial_renderer="rtx")),
        ),
        scene=InteractiveSceneCfg(num_envs=2, env_spacing=4.0),
    )
    with launch_simulation(cfg, {"visualizer": ["newton"]}):
        calls.append("user")

    assert calls == ["register", "user"]


def test_newton_rtx_selection_points_at_newton():
    with pytest.raises(ValueError, match="newton_rtx.*Use the 'newton' visualizer"):
        sim_launcher.validate_launch(SimulationCfg(physics=sim_launcher.NewtonCfg()), {"visualizer": ["newton_rtx"]})


def test_newton_gl_alias_selects_newton_without_a_warning():
    cfg = SimulationCfg(physics=sim_launcher.NewtonCfg())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sim_launcher.validate_launch(cfg, {"visualizer": ["newton_gl"]})
    assert [viewer.backend.visualizer_type for viewer in cfg.visualizer_cfgs] == ["newton"]


def test_initial_renderer_opens_in_opengl_and_rtx_needs_the_viewer_camera():
    assert NewtonGLBackendCfg().initial_renderer == "opengl"
    cfg = types.SimpleNamespace(
        sim=SimulationCfg(
            physics=sim_launcher.NewtonCfg(),
            visualizer_cfgs=VisualizerCfg(backend=NewtonGLBackendCfg(initial_renderer="rtx", rtx_camera=False)),
        ),
        scene=InteractiveSceneCfg(num_envs=2, env_spacing=4.0),
    )
    with pytest.raises(ValueError, match="initial_renderer='rtx' requires rtx_camera"):
        sim_launcher.validate_launch(cfg, {})


def test_kit_run_drops_the_implicit_ovrtx_viewer_camera(monkeypatch: pytest.MonkeyPatch):
    """A Kit run cannot load the OVRTX runtime, so its Newton GL window keeps the OpenGL view only."""
    monkeypatch.setattr(sim_launcher, "_fit_window_to_display", lambda size: (640, 360))  # As on a small display.
    from isaaclab_visualizers.newton import NewtonGLBackendCfg

    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.visualizers import VisualizerCfg, WindowCfg

    visualizer = VisualizerCfg(window=WindowCfg(size=(5000, 3000)), backend=NewtonGLBackendCfg(rtx_camera=True))
    cfg = types.SimpleNamespace(
        sim=SimulationCfg(physics=sim_launcher.NewtonCfg(), visualizer_cfgs=visualizer),
        scene=InteractiveSceneCfg(num_envs=2, env_spacing=4.0),
    )

    configs, kit_sources = sim_launcher.validate_launch(cfg, {"require_kit": True})

    assert kit_sources
    assert not [name for name in vars(cfg.scene) if name.startswith("viewer_camera_")]
    # The camera only justified fitting the window to the display and listing scene cameras, so none of it stays.
    launched = cfg.sim.visualizer_cfgs[0]  # The launcher may hold a copy of the declaration.
    assert launched.window.size == (5000, 3000) and not launched.streaming_view
    assert not [node for node in configs if isinstance(node, sim_launcher.OVRTXRendererCfg)]


@pytest.mark.parametrize(("outcome", "exit_code"), [("ok", 0), ("interrupt", 130), ("error", 1)])
def test_config_named_launcher_starts_and_closes(monkeypatch: pytest.MonkeyPatch, outcome, exit_code):
    """A config-selected launcher closes on success, interruption or failure with the matching exit status."""
    events = []

    class FakeLauncher(SimulationLauncher):
        def __init__(self, launcher_args):
            events.append("start")

        def close(self, exit_code=0):
            events.append(exit_code)

    class CustomRendererCfg(RendererCfg):
        launcher_type = "fake_runtime:FakeLauncher"

    monkeypatch.setitem(sys.modules, "fake_runtime", types.SimpleNamespace(FakeLauncher=FakeLauncher))
    cfg = argparse.Namespace(physics=sim_launcher.NewtonCfg(), renderer=CustomRendererCfg())

    with contextlib.suppress(KeyboardInterrupt, RuntimeError), launch_simulation(cfg):
        events.append("user")
        if outcome == "interrupt":
            signal.raise_signal(signal.SIGINT)
        elif outcome == "error":
            raise RuntimeError("user failure")

    assert events == ["start", "user", exit_code]


@pytest.mark.parametrize("producer_count", [0, 1, 2])
def test_shared_view_recording_selects_its_producer_without_a_window(producer_count):
    """Share each perspective producer across recorders, independently of other same-backend viewers."""
    from types import SimpleNamespace

    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_visualizers.newton import NewtonGLBackendCfg

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.visualizers import ImageViewCfg, PerspectiveCameraCfg, WindowCfg

    views = [ImageViewCfg(source=PerspectiveCameraCfg()) for _ in range(producer_count)]
    visualizers = [
        VisualizerCfg(view=view, window=WindowCfg(size=(128, 96)), backend=NewtonGLBackendCfg()) for view in views
    ]
    if producer_count == 1:
        visualizers = visualizers[0]  # SimulationCfg also accepts a single configuration.
        visualizers.backend = []  # Recording supplies the default producer.
    else:
        visualizers.insert(
            0, VisualizerCfg(backend=NewtonGLBackendCfg())
        )  # An unrelated viewer of the same backend must stay inactive.
    cfg = SimpleNamespace(
        sim=SimulationCfg(device="cpu", physics=NewtonCfg(), visualizer_cfgs=visualizers),
        video_recorders=[
            VideoRecorderCfg(source=view) for view in views or [ImageViewCfg(source="front")] for _ in range(2)
        ],
    )
    with launch_simulation(cfg, {"visualizer": [], "headless": True}):
        assert len(cfg.sim.visualizer_cfgs) == producer_count
        for producer, view in zip(cfg.sim.visualizer_cfgs, views, strict=True):
            assert producer.headless
            assert producer.view is view
            assert producer.window.size == (128, 96)


@pytest.mark.parametrize("visualizers,source", [([], "newton"), (["newton"], "newton"), ([], "kit")])
def test_legacy_recording_reuses_the_display_producer(kit_branch_taken, visualizers, source):
    """Display and multiple legacy recorders create one producer when none was configured."""
    from types import SimpleNamespace

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab.sim import SimulationCfg

    cfg = SimpleNamespace(
        sim=SimulationCfg(device="cpu", physics=sim_launcher.NewtonCfg()),
        video_recorders=[VideoRecorderCfg(source=f"viz:{source}"), VideoRecorderCfg(source=f"viz:{source}")],
    )
    args = {"visualizer": visualizers}
    with launch_simulation(cfg, args):
        assert len(cfg.sim.visualizer_cfgs) == 1
        assert cfg.sim.visualizer_cfgs[0].headless == (not visualizers)
        assert args.get("enable_cameras", False) == (source == "kit")
    assert kit_branch_taken == [True] * (source == "kit")
