# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless tests for :func:`isaaclab.app.launch_simulation` and its config scan.

Kit is never started: availability and ``AppLauncher`` are faked, and reaching
``_ensure_isaac_sim_available`` is the signal that the Kit branch was taken.
"""

import argparse
import sys
import types

import pytest
from isaaclab_newton.physics import NewtonCfg, VBDSolverCfg

import isaaclab.app as isaaclab_app
import isaaclab.app.sim_launcher as sim_launcher
import isaaclab.utils as isaaclab_utils
import isaaclab.utils.assets as assets_utils
from isaaclab.app import launch_simulation
from isaaclab.physics import PhysicsCfg

pytestmark = pytest.mark.unit

_NO_VISUALIZERS = {"has_any_visualizers": False, "has_kit_visualizer": False}


def _scan(**overrides) -> sim_launcher.Scan:
    """A kitless scan result whose fields can be overridden per test."""
    fields = dict(
        resolved_physics_cfg=None,
        effective_cfg=object(),
        visualizer_intent=_NO_VISUALIZERS,
        has_ovrtx=False,
        has_kit_camera=False,
        has_kit_physics=False,
        has_kitless_physics=True,
        has_ovphysx_physics=False,
        needs_kit=False,
    )
    fields.update(overrides)
    return sim_launcher.Scan(**fields)


class _FakeApp:
    def __init__(self):
        self.close_calls = []

    def close(self, **kwargs):
        self.close_calls.append(kwargs)


@pytest.fixture
def fake_kit(monkeypatch):
    """Take the Kit branch without Kit: records the launcher arguments and the app's close calls."""
    launched = {}

    class _FakeAppLauncher:
        def __init__(self, launcher_args):
            launched.update(vars(launcher_args) if isinstance(launcher_args, argparse.Namespace) else launcher_args)
            self.app = launched.setdefault("app", _FakeApp())

    monkeypatch.setattr(sim_launcher, "_ensure_isaac_sim_available", lambda: None)
    monkeypatch.setattr(isaaclab_utils, "has_kit", lambda: False)
    monkeypatch.setattr(isaaclab_app, "AppLauncher", _FakeAppLauncher)
    monkeypatch.setattr(assets_utils, "configure_storage_profile", lambda: None)
    monkeypatch.delenv("LIVESTREAM", raising=False)
    return launched


@pytest.fixture
def kit_branch_taken(monkeypatch):
    """Report whether ``launch_simulation`` entered its Kit branch, pretending Kit already runs."""
    taken: list[bool] = []
    monkeypatch.setattr(isaaclab_utils, "has_kit", lambda: True)
    monkeypatch.setattr(sim_launcher, "_ensure_isaac_sim_available", lambda: taken.append(True))
    return taken


def test_make_physics_cfg_builds_core_vbd():
    physics_cfg = sim_launcher.make_physics_cfg("newton_vbd")

    assert isinstance(physics_cfg, NewtonCfg)
    assert isinstance(physics_cfg.solver_cfg, VBDSolverCfg)
    with pytest.raises(ValueError, match="Invalid physics config"):
        sim_launcher.make_physics_cfg("bullet")


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (argparse.Namespace(livestream=2, visualizer=None, visualizer_explicit=False), ["kit"]),
        (argparse.Namespace(livestream=2, visualizer="rerun", visualizer_explicit=False), ["rerun", "kit"]),
        (argparse.Namespace(livestream=0, visualizer=None, visualizer_explicit=False), None),
    ],
    ids=["injected", "appended", "no-livestream"],
)
def test_livestream_requests_kit_visualizer(args, expected):
    """Livestreaming needs a video-producing viewport, so the Kit visualizer is added when missing."""
    sim_launcher._ensure_livestream_kit_visualizer(args)
    assert args.visualizer == expected


def test_livestream_rejects_disabled_visualizers():
    args = argparse.Namespace(livestream=2, visualizer=None, visualizer_explicit=True)

    with pytest.raises(ValueError, match="Livestreaming requires the Kit visualizer"):
        sim_launcher._ensure_livestream_kit_visualizer(args)


@pytest.mark.parametrize(
    ("launcher_args", "expected_source"),
    [
        (argparse.Namespace(experience="isaaclab.python.kit", visualizer=None), "explicit Kit experience"),
        ({"visualizer": ["kit"]}, "Kit visualizer"),
        ({"require_kit": True}, "explicit Kit requirement"),
        ({"livestream": 1}, "livestreaming"),
        ({}, None),
    ],
)
def test_kit_runtime_sources_from_launcher_arguments(launcher_args, expected_source):
    """Launcher arguments can require Kit on top of a kitless physics configuration."""
    sources = sim_launcher._get_kit_runtime_sources(_scan(), launcher_args)
    if expected_source is None:
        assert sources == ()
    else:
        assert any(expected_source in source for source in sources)


@pytest.mark.parametrize(
    ("launcher_args", "expected_taken"),
    [
        ({}, False),
        ({"require_kit": True}, True),
        (argparse.Namespace(require_kit=True), True),
        # '--viz kit' requires Kit on its own; require_kit=False must not override that
        ({"visualizer": ["kit"], "visualizer_explicit": True, "require_kit": False}, True),
    ],
    ids=["default", "require-kit", "namespace", "require-kit-false"],
)
def test_require_kit_is_additive(kit_branch_taken, launcher_args, expected_taken):
    """``require_kit`` can turn a kitless launch into a Kit one, never the reverse."""
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=launcher_args):
        pass

    assert kit_branch_taken == [True] * expected_taken


def test_kitless_launch_configures_storage_and_ovrtx_before_user_code(kit_branch_taken, monkeypatch):
    """OVRTX schema registration and storage profile routing happen before the user's code runs."""
    events = []
    monkeypatch.setattr(assets_utils, "configure_storage_profile", lambda: events.append("storage"))
    monkeypatch.setitem(
        sys.modules, "ovrtx", types.SimpleNamespace(register_schema_paths=lambda: events.append("register"))
    )
    cfg = argparse.Namespace(physics=NewtonCfg(), visualizer_cfgs=argparse.Namespace(visualizer_type="newton_rtx"))

    with launch_simulation(cfg):
        events.append("user")

    assert events == ["register", "storage", "user"]
    assert kit_branch_taken == []


@pytest.mark.parametrize(
    ("scan_overrides", "launcher_args", "message"),
    [
        ({"has_ovrtx": True}, {"require_kit": True}, "OVRTX runtime"),
        ({"has_ovphysx_physics": True}, {"visualizer": ["kit"]}, "OvPhysX physics"),
    ],
)
def test_kitless_runtimes_cannot_share_a_process_with_kit(monkeypatch, scan_overrides, launcher_args, message):
    """OVRTX and OvPhysX are kitless and are rejected together with any Kit source, before loading anything."""
    calls = []
    monkeypatch.setitem(sys.modules, "ovrtx", types.SimpleNamespace(register_schema_paths=lambda: calls.append(1)))
    monkeypatch.setattr(sim_launcher, "scan", lambda _cfg, _launcher_args: _scan(**scan_overrides))

    with pytest.raises(ValueError, match=message):
        with launch_simulation(cfg=object(), launcher_args=launcher_args):
            pass
    assert calls == []


def test_newton_rtx_visualizer_rejects_kit_physics(monkeypatch):
    """The real scan flags the ``newton_rtx`` visualizer as OVRTX, which Isaac Sim PhysX cannot join."""
    monkeypatch.setitem(sys.modules, "ovrtx", types.SimpleNamespace(register_schema_paths=lambda: None))

    with pytest.raises(ValueError, match="OVRTX runtime"):
        with launch_simulation(sim_launcher.PhysxCfg(), {"visualizer": ["newton_rtx"]}):
            pass


def test_kit_launch_auto_enables_cameras_and_propagates_the_device(fake_kit, monkeypatch):
    """Kit camera sensors enable camera rendering, and AppLauncher's resolved device reaches the sim config."""
    cfg = argparse.Namespace(sim=argparse.Namespace(device="cuda:0"))
    monkeypatch.setattr(sim_launcher, "scan", lambda *_: _scan(effective_cfg=cfg, has_kit_camera=True, needs_kit=True))
    fake_kit["app"] = _FakeApp()

    with launch_simulation(cfg):
        pass

    assert fake_kit["enable_cameras"] is True
    assert fake_kit["app"].close_calls == [{}]


@pytest.mark.parametrize("failure", ["user-code", "storage-profile"])
def test_kit_launch_closes_the_app_with_a_failure_status(fake_kit, monkeypatch, failure):
    """A failure inside the launch block, or while configuring storage, still closes Kit with exit code 1."""
    monkeypatch.setattr(sim_launcher, "scan", lambda *_: _scan(has_kit_physics=True, needs_kit=True))
    fake_kit["app"] = _FakeApp()
    if failure == "storage-profile":

        def _reject():
            raise RuntimeError("sentinel")

        monkeypatch.setattr(assets_utils, "configure_storage_profile", _reject)

    with pytest.raises(RuntimeError, match="sentinel"):
        with launch_simulation(object(), argparse.Namespace()):
            raise RuntimeError("sentinel")

    assert fake_kit["app"].close_calls == [{"exit_code": 1}]
