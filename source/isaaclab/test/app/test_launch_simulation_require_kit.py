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
from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

import isaaclab.app.sim_launcher as sim_launcher
import isaaclab.utils.assets as assets_utils
from isaaclab.app import SimulationLauncher, launch_simulation
from isaaclab.physics import PhysicsCfg
from isaaclab.renderers import RendererCfg
from isaaclab.sim import SimulationCfg


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
    monkeypatch.setattr(assets_utils, "configure_storage_profile", lambda: events.append("configured"))

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
    monkeypatch.setattr(assets_utils, "configure_storage_profile", reject_profile)

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


def test_require_kit_rejects_ovrtx_runtime(monkeypatch: pytest.MonkeyPatch):
    config_scan = sim_launcher.Scan(
        resolved_physics_cfg=None,
        effective_cfg=object(),
        visualizer_cfgs=[],
        has_ovrtx=True,
        has_kit_camera=False,
        has_kit_physics=False,
        has_ovphysx_physics=False,
        needs_kit=False,
    )
    monkeypatch.setattr(sim_launcher, "scan", lambda _cfg, _launcher_args: config_scan)

    with pytest.raises(ValueError, match="OVRTX runtime"):
        with launch_simulation(cfg=object(), launcher_args={"require_kit": True}):
            pass


def test_newton_rtx_rejects_kit_before_loading_ovrtx(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.setitem(
        sys.modules, "ovrtx", types.SimpleNamespace(register_schema_paths=lambda: calls.append("register"))
    )

    with pytest.raises(ValueError, match="OVRTX runtime"):
        with launch_simulation(SimulationCfg(physics=sim_launcher.PhysxCfg()), {"visualizer": "newton_rtx"}):
            pass

    assert calls == []


def test_kitless_ovrtx_registers_before_user_code(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.setitem(
        sys.modules, "ovrtx", types.SimpleNamespace(register_schema_paths=lambda: calls.append("register"))
    )
    monkeypatch.setattr(assets_utils, "configure_storage_profile", lambda: calls.append("storage"))
    cfg = SimulationCfg(physics=sim_launcher.NewtonCfg(), visualizer_cfgs=NewtonRTXVisualizerCfg())

    with launch_simulation(cfg):
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

    with launch_simulation(cfg=SimulationCfg(physics=PhysicsCfg()), launcher_args=launcher_args):
        pass

    assert kit_branch_taken == [True]
