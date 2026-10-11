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
import signal
import sys
import types

import isaaclab_physx.app as physx_app
import pytest

import isaaclab.app.sim_launcher as sim_launcher
from isaaclab.app import SimulationLauncher, launch_simulation
from isaaclab.physics import PhysicsCfg
from isaaclab.renderers import RendererCfg


@pytest.fixture
def kit_branch_taken(monkeypatch: pytest.MonkeyPatch):
    """Report whether ``launch_simulation`` entered its Kit branch, without starting Kit."""
    taken: list[bool] = []

    class FakeKitLauncher(SimulationLauncher):
        def __init__(self, launcher_args):
            taken.append(True)

    monkeypatch.setattr(physx_app, "KitLauncher", FakeKitLauncher)
    return taken


def test_default_stays_kitless_for_a_kitless_config(kit_branch_taken):
    with launch_simulation(cfg=PhysicsCfg(), launcher_args={}):
        pass

    assert kit_branch_taken == []


def test_kitless_launch_configures_storage_before_user_code(kit_branch_taken, monkeypatch: pytest.MonkeyPatch):
    """A direct OmniClient read inside a kitless runtime must see profile routing."""
    events = []
    monkeypatch.setattr(sim_launcher, "configure_storage_profile", lambda: events.append("configured"))

    with launch_simulation(cfg=PhysicsCfg(), launcher_args={}):
        events.append("user-code")

    assert events == ["configured", "user-code"]
    assert kit_branch_taken == []


def test_storage_profile_failure_closes_started_kit(monkeypatch: pytest.MonkeyPatch):
    """A profile failure after Kit starts must still close the application."""
    close_calls = []

    class FakeKitLauncher(SimulationLauncher):
        def close(self, exit_code=0):
            close_calls.append(exit_code)

    def reject_profile():
        raise RuntimeError("profile rejected")

    monkeypatch.setattr(physx_app, "KitLauncher", FakeKitLauncher)
    monkeypatch.setattr(sim_launcher, "configure_storage_profile", reject_profile)

    with pytest.raises(RuntimeError, match="profile rejected"):
        with launch_simulation(cfg=PhysicsCfg(), launcher_args={"require_kit": True}):
            pass

    assert close_calls == [1]


def test_require_kit_launches_kit_for_a_kitless_config(kit_branch_taken):
    with launch_simulation(cfg=PhysicsCfg(), launcher_args={"require_kit": True}):
        pass

    assert kit_branch_taken == [True]


def test_require_kit_reads_from_a_namespace(kit_branch_taken):
    # scripts pass their argparse namespace straight through, so the key is read off it too
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=argparse.Namespace(require_kit=True)):
        pass

    assert kit_branch_taken == [True]


def test_require_kit_rejects_ovrtx_runtime():
    cfg = types.SimpleNamespace(physics=PhysicsCfg(), renderer=sim_launcher.OVRTXRendererCfg())
    with pytest.raises(ValueError, match="OVRTX runtime"):
        with launch_simulation(cfg, {"require_kit": True}):
            pass


def test_newton_rtx_rejects_kit_before_loading_ovrtx(monkeypatch: pytest.MonkeyPatch):
    class CustomPhysxCfg(sim_launcher.PhysxCfg):
        pass

    calls = []
    monkeypatch.setitem(
        sys.modules, "ovrtx", types.SimpleNamespace(register_schema_paths=lambda: calls.append("register"))
    )
    monkeypatch.setattr(physx_app, "KitLauncher", lambda _: pytest.fail("Invalid runtime reached Kit launch"))

    with pytest.raises(ValueError, match="OVRTX runtime"):
        with launch_simulation(CustomPhysxCfg(), {"visualizer": ["newton_rtx"]}):
            pass

    assert calls == []


def test_kitless_ovrtx_registers_before_user_code(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.setitem(
        sys.modules, "ovrtx", types.SimpleNamespace(register_schema_paths=lambda: calls.append("register"))
    )
    monkeypatch.setattr(sim_launcher, "configure_storage_profile", lambda: calls.append("storage"))
    with launch_simulation(sim_launcher.NewtonCfg(), {"visualizer": ["newton_rtx"]}):
        calls.append("user")

    assert calls == ["register", "storage", "user"]


@pytest.mark.parametrize("interrupt", [False, True])
def test_config_named_launcher_starts_and_closes(monkeypatch: pytest.MonkeyPatch, interrupt):
    """A config-selected launcher closes on success or interruption with the correct exit status."""
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

    try:
        with launch_simulation(cfg):
            events.append("user")
            if interrupt:
                signal.raise_signal(signal.SIGINT)
    except KeyboardInterrupt:
        assert interrupt
    else:
        assert not interrupt

    assert events == ["start", "user", 130 if interrupt else 0]


def test_require_kit_false_does_not_suppress_a_kit_config(kit_branch_taken):
    # '--viz kit' requires Kit on its own; require_kit=False must not override that
    launcher_args = {"visualizer": ["kit"], "require_kit": False}

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=launcher_args):
        pass

    assert kit_branch_taken == [True]


@pytest.mark.parametrize("producer_count", [0, 1, 2])
def test_shared_view_recording_selects_its_producer_without_a_window(producer_count):
    """Retain each recorded perspective producer, including when viewers share the same backend."""
    from types import SimpleNamespace

    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.visualizers import ImageViewCfg, PerspectiveCameraCfg, WindowCfg, visualizer_cfg

    assert "_resolve_video_sources" not in vars(sim_launcher)
    assert "resolve_visualizer_cfgs" not in vars(visualizer_cfg)
    views = [ImageViewCfg(source=PerspectiveCameraCfg()) for _ in range(producer_count)]
    visualizers = [NewtonGLVisualizerCfg(view=view, window=WindowCfg(size=(128, 96))) for view in views]
    if producer_count == 1:
        visualizers = visualizers[0]  # SimulationCfg also accepts a single configuration.
    else:
        visualizers.insert(0, NewtonGLVisualizerCfg())  # An unrelated viewer of the same backend must stay inactive.
    cfg = SimpleNamespace(
        sim=SimulationCfg(device="cpu", physics=NewtonCfg(), visualizer_cfgs=visualizers),
        video_recorders=[VideoRecorderCfg(view=view) for view in views or [ImageViewCfg(source="front")]],
    )
    declared = cfg.sim.visualizer_cfgs
    preview = sim_launcher.scan(cfg, {"visualizer": [], "headless": True})
    assert cfg.sim.visualizer_cfgs is declared
    assert len(preview.visualizer_cfgs) == producer_count
    with launch_simulation(cfg, {"visualizer": [], "headless": True}):
        assert len(cfg.sim.visualizer_cfgs) == producer_count
        for producer, recorder in zip(cfg.sim.visualizer_cfgs, cfg.video_recorders):
            assert producer.headless
            assert producer.view is recorder.view
            assert producer.window.size == (128, 96)


@pytest.mark.parametrize("visualizers", [[], ["newton_gl"]])
def test_legacy_recording_reuses_the_display_producer(visualizers):
    """Display and multiple legacy recorders create one producer when none was configured."""
    from types import SimpleNamespace

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab.sim import SimulationCfg

    cfg = SimpleNamespace(
        sim=SimulationCfg(device="cpu", physics=sim_launcher.NewtonCfg(), visualizer_cfgs=None),
        video_recorders=[VideoRecorderCfg(source="viz:newton_gl"), VideoRecorderCfg(source="viz:newton_gl")],
    )
    with launch_simulation(cfg, {"visualizer": visualizers}):
        assert len(cfg.sim.visualizer_cfgs) == 1
        assert cfg.sim.visualizer_cfgs[0].headless == (not visualizers)
