# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture source ownership, environment binding, and reset event delivery."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from isaaclab_teleop import CaptureTeleopSourceCfg, IsaacTeleopCfg
from isaaclab_teleop.control_events import ControlEvents
from isaaclab_teleop.play_source import CaptureTeleopSource, _desktop_pipeline

from isaaclab.utils import instantiate

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


@pytest.fixture
def capture(monkeypatch):
    from isaaclab_teleop import device_selection, play_source

    device = SimpleNamespace(
        __enter__=Mock(),
        __exit__=Mock(),
        advance=Mock(return_value=torch.tensor([0.3, -0.2])),
        request_start=Mock(),
        reset=Mock(),
        notify_reset=Mock(),
        last_control_events=ControlEvents(),
    )
    create = Mock(return_value=device)
    monkeypatch.setattr(device_selection, "create_teleop_device", create)
    monkeypatch.setattr(play_source, "_desktop_pipeline", Mock(return_value=object()))
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        scene=SimpleNamespace(env_prim_paths=[f"/World/envs/env_{i}" for i in range(4)]),
        sim=SimpleNamespace(visualizers=[object()], get_setting=lambda _: False),
    )
    return device, create, env


def test_selected_environment_input_and_lifetime(capture):
    """A source binds one batch row and disengages without touching simulation."""
    device, create, env = capture
    cfg = CaptureTeleopSourceCfg(device=IsaacTeleopCfg(), cloudxr_env_file=None)
    source = instantiate(cfg, env, env_ids=[2])
    assert create.call_args.kwargs["env_prim_path"] == "/World/envs/env_2"
    assert create.call_args.kwargs["visualizers"] is env.sim.visualizers
    assert create.call_args.kwargs["sim_device"] == "cpu"
    device.__enter__.assert_called_once()
    device.request_start.assert_called_once()
    sample = source.advance()
    torch.testing.assert_close(sample.actions, torch.tensor([0.3, -0.2]))
    assert sample.active and not sample.reset

    device.last_control_events = ControlEvents(is_active=False)
    assert not source.advance().active
    device.last_control_events = ControlEvents()
    assert not source.advance().active  # no new event does not re-engage
    device.last_control_events = ControlEvents(is_active=True)
    assert source.advance().active
    device.advance.return_value = None
    assert not source.advance().active

    source.close()
    source.close()
    device.__exit__.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        source.advance()


def test_host_reset_does_not_swallow_later_operator_reset(capture):
    """A skipped host-reset completion cannot consume the next operator reset."""
    device, _, env = capture
    source = CaptureTeleopSource(CaptureTeleopSourceCfg(device=IsaacTeleopCfg()), env, env_ids=[2])
    try:
        source.reset([])
        device.notify_reset.assert_not_called()
        source.reset(torch.tensor([2]))
        assert not source.advance().reset
        assert not source.advance().reset
        # Capture may replace the host-reset completion before the source polls it.
        # The first observed reset is then the operator's later command.
        device.last_control_events = ControlEvents(is_active=False, should_reset=True)
        sample = source.advance()
        assert sample.reset and not sample.active
        device.notify_reset.assert_called_once()
        device.reset.assert_not_called()
        assert not source.advance().reset  # repeated operator frame cannot reset the environment twice
        source.reset([2])
        device.last_control_events = ControlEvents(is_active=False)
        assert not source.advance().reset
        device.last_control_events = ControlEvents(is_active=False, should_reset=True)
        assert source.advance().reset
    finally:
        source.close()


def test_start_failure_releases_device(capture):
    """Startup still attempts cleanup, and a cleanup error cannot hide the original failure."""
    device, _, env = capture
    device.__enter__.side_effect = RuntimeError("runtime failed during startup")
    device.__exit__.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(RuntimeError, match="runtime failed"):
        CaptureTeleopSource(CaptureTeleopSourceCfg(device=IsaacTeleopCfg()), env, env_ids=[0])
    device.__exit__.assert_called_once()
    device.request_start.assert_not_called()


def test_incompatible_bindings_fail_before_start(capture, monkeypatch):
    from isaaclab_teleop import play_source

    device, create, env = capture
    cfg = CaptureTeleopSourceCfg(device=IsaacTeleopCfg())
    with pytest.raises(ValueError, match="one selected environment"):
        CaptureTeleopSource(cfg, env, env_ids=[0, 1])
    monkeypatch.setattr(play_source, "has_kit", lambda: True)
    with pytest.raises(ValueError, match="non-Kit"):
        CaptureTeleopSource(cfg, env, env_ids=[0])
    monkeypatch.setattr(play_source, "has_kit", lambda: False)
    env.sim.get_setting = lambda _: True
    with pytest.raises(ValueError, match="desktop input only"):
        CaptureTeleopSource(cfg, env, env_ids=[0])
    create.assert_not_called()
    device.__enter__.assert_not_called()


def test_pipeline_scope_allows_desktop_frames_and_rejects_xr(monkeypatch):
    """Opaque builders are checked by their actual leaves before starting input."""
    source_types = {name: type(name, (), {}) for name in ("KeyboardSource", "GamepadSource", "SpaceMouseSource")}
    value_input = type("ValueInput", (), {})
    monkeypatch.setitem(
        sys.modules,
        "isaacteleop.retargeting_engine.deviceio_source_nodes",
        SimpleNamespace(**source_types),
    )
    monkeypatch.setitem(
        sys.modules, "isaacteleop.retargeting_engine.interface", SimpleNamespace(ValueInput=value_input)
    )
    leaves = [source_types["KeyboardSource"](), value_input()]
    pipeline = SimpleNamespace(get_leaf_nodes=lambda: leaves)
    cfg = IsaacTeleopCfg(pipeline_builder=lambda: pipeline)
    assert _desktop_pipeline(cfg) is pipeline

    leaves.append(type("HandsSource", (), {})())
    with pytest.raises(ValueError, match="HandsSource"):
        _desktop_pipeline(cfg)
    leaves[:] = [value_input()]
    with pytest.raises(ValueError, match="no desktop source"):
        _desktop_pipeline(cfg)
