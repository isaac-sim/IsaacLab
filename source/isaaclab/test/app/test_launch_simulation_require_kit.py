# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ``require_kit`` launcher argument read by :func:`isaaclab.app.launch_simulation`.

``require_kit`` lets a tool state that it needs Kit for a reason the config cannot express --
the URDF/MJCF converters set it because they reach a Kit-only importer extension whenever the
standalone importer wheel is absent. The override is additive: it can turn a kitless launch into
a Kit one, never the reverse.

Kit is never actually started here: availability and ``AppLauncher`` are faked, and reaching
``_ensure_isaac_sim_available`` is the signal that the Kit branch was taken. No Kit/GPU required.
"""

import argparse

import pytest

import isaaclab.app as isaaclab_app
import isaaclab.app.sim_launcher as sim_launcher
import isaaclab.utils as isaaclab_utils
import isaaclab.utils.assets as assets_utils
from isaaclab.app import launch_simulation
from isaaclab.physics import PhysicsCfg


@pytest.fixture
def kit_branch_taken(monkeypatch: pytest.MonkeyPatch):
    """Report whether ``launch_simulation`` entered its Kit branch, without starting Kit."""
    taken: list[bool] = []
    # pretend Kit is already running so the launcher skips constructing AppLauncher
    monkeypatch.setattr(isaaclab_utils, "has_kit", lambda: True)
    monkeypatch.setattr(sim_launcher, "_ensure_isaac_sim_available", lambda: taken.append(True))
    return taken


def test_default_stays_kitless_for_a_kitless_config(kit_branch_taken):
    with launch_simulation(cfg=PhysicsCfg(), launcher_args={}):
        pass

    assert kit_branch_taken == []


def test_kitless_launch_configures_storage_before_user_code(kit_branch_taken, monkeypatch: pytest.MonkeyPatch):
    """A direct OmniClient read inside a kitless runtime must see profile routing."""
    events = []
    monkeypatch.setattr(assets_utils, "configure_storage_profile", lambda: events.append("configured"))

    with launch_simulation(cfg=PhysicsCfg(), launcher_args={}):
        events.append("user-code")

    assert events == ["configured", "user-code"]
    assert kit_branch_taken == []


def test_storage_profile_failure_closes_started_kit(monkeypatch: pytest.MonkeyPatch):
    """A profile failure after Kit starts must still close the application."""
    close_calls = []

    class FakeApp:
        def close(self, **kwargs):
            close_calls.append(kwargs)

    class FakeAppLauncher:
        def __init__(self, _launcher_args):
            self.app = FakeApp()

    def reject_profile():
        raise RuntimeError("profile rejected")

    monkeypatch.setattr(sim_launcher, "_ensure_isaac_sim_available", lambda: None)
    monkeypatch.setattr(isaaclab_utils, "has_kit", lambda: False)
    monkeypatch.setattr(isaaclab_app, "AppLauncher", FakeAppLauncher)
    monkeypatch.setattr(assets_utils, "configure_storage_profile", reject_profile)

    with pytest.raises(RuntimeError, match="profile rejected"):
        with launch_simulation(cfg=PhysicsCfg(), launcher_args={"require_kit": True}):
            pass

    assert close_calls == [{"exit_code": 1}]


def test_require_kit_launches_kit_for_a_kitless_config(kit_branch_taken):
    with launch_simulation(cfg=PhysicsCfg(), launcher_args={"require_kit": True}):
        pass

    assert kit_branch_taken == [True]


def test_require_kit_reads_from_a_namespace(kit_branch_taken):
    # scripts pass their argparse namespace straight through, so the key is read off it too
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=argparse.Namespace(require_kit=True)):
        pass

    assert kit_branch_taken == [True]


def test_require_kit_rejects_ovrtx_runtime(monkeypatch: pytest.MonkeyPatch):
    config_scan = sim_launcher.Scan(
        resolved_physics_cfg=None,
        effective_cfg=object(),
        visualizer_intent={"has_any_visualizers": False, "has_kit_visualizer": False},
        has_ovrtx=True,
        has_kit_camera=False,
        has_kit_physics=False,
        has_kitless_physics=True,
        has_ovphysx_physics=False,
        needs_kit=False,
    )
    monkeypatch.setattr(sim_launcher, "scan", lambda _cfg, _launcher_args: config_scan)

    with pytest.raises(ValueError, match="OVRTX renderer"):
        with launch_simulation(cfg=object(), launcher_args={"require_kit": True}):
            pass


def test_require_kit_false_does_not_suppress_a_kit_config(kit_branch_taken):
    # '--viz kit' requires Kit on its own; require_kit=False must not override that
    launcher_args = {"visualizer": ["kit"], "visualizer_explicit": True, "require_kit": False}

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=launcher_args):
        pass

    assert kit_branch_taken == [True]
