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

from pxr import Gf

import isaaclab.app.sim_launcher as sim_launcher
from isaaclab.app import SimulationLauncher, launch_simulation
from isaaclab.physics import PhysicsCfg
from isaaclab.renderers import RendererCfg
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
        (SimulationCfg(physics=CustomPhysxCfg()), {"visualizer": ["newton_rtx"]}),
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
    monkeypatch.setattr(sim_launcher, "configure_storage_profile", lambda: calls.append("storage"))
    with launch_simulation(
        SimulationCfg(physics=sim_launcher.NewtonCfg(), device="cpu"), {"visualizer": ["newton_rtx"]}
    ):
        calls.append("user")

    assert calls == ["register", "storage", "user"]


@pytest.mark.parametrize("channels", [("rgb",), ("rgb", "depth")])
def test_rtx_perspective_configuration_declares_scene_cameras(channels):
    """Launch declares selected cameras without adding unrelated sensors to the window."""
    from isaaclab_visualizers.newton import NewtonRTXBackendCfg

    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.visualizers import ImageViewCfg, PerspectiveCameraCfg, SceneCameraCfg, VisualizerCfg, WindowCfg

    view = ImageViewCfg(source=PerspectiveCameraCfg(eye=(0.0, -5.0, 0.0), lookat=(0.0, 0.0, 0.0)))
    cfg = types.SimpleNamespace(
        sim=SimulationCfg(
            physics=sim_launcher.NewtonCfg(),
            visualizer_cfgs=[
                VisualizerCfg(
                    window=WindowCfg(size=(160, 120)), streaming_gt_types=channels, backend=NewtonRTXBackendCfg()
                ),
                VisualizerCfg(window=WindowCfg(size=(160, 120)), view=view, backend=NewtonRTXBackendCfg()),
            ],
        ),
        scene=InteractiveSceneCfg(num_envs=2, env_spacing=4.0),
    )
    cfg.scene.observation_camera = sim_launcher.CameraCfg(
        prim_path="{ENV_REGEX_NS}/ObservationCamera", width=64, height=48,
        data_types=["rgba", "depth"], renderer_cfg=sim_launcher.OVRTXRendererCfg(),
    )  # fmt: skip
    sim_launcher.validate_launch(cfg, {"visualizer": ["newton_rtx"]})
    cameras = [sensor for sensor in vars(cfg.scene).values() if isinstance(sensor, sim_launcher.CameraCfg)]
    assert cameras[0] is cfg.scene.observation_camera
    assert len(cameras) == 3
    assert cfg.sim.visualizer_cfgs[0].cameras == [SceneCameraCfg(prim_path=cameras[1].prim_path)]
    assert vars(cfg.scene)[view.source] is cameras[2]
    assert cameras[2].offset.pos == (0.0, -5.0, 0.0)
    assert cameras[2].offset.rot == pytest.approx((2**-0.5, 0.0, 0.0, 2**-0.5))
    assert cameras[1].offset.pos == cfg.sim.visualizer_cfgs[0].eye
    assert cameras[1].data_types == ["rgba" if channel == "rgb" else channel for channel in channels]
    assert all((camera.width, camera.height) == (160, 120) for camera in cameras[1:])
    for camera in cameras[1:]:
        assert camera.spawn.vertical_aperture == pytest.approx(Gf.Camera().verticalAperture)
        assert camera.spawn.horizontal_aperture / camera.spawn.vertical_aperture == pytest.approx(160 / 120)
    assert all(camera.renderer_cfg == cfg.sim.visualizer_cfgs[0].backend.renderer_cfg for camera in cameras[1:])
    sim_launcher.validate_launch(cfg, {"visualizer": ["newton_rtx"]})
    assert [item for item in vars(cfg.scene).values() if isinstance(item, sim_launcher.CameraCfg)] == cameras


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


@pytest.mark.parametrize("visualizers,source", [([], "newton_gl"), (["newton_gl"], "newton_gl"), ([], "kit")])
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
