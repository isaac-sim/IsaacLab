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
        session_cleanup_failed=False,
        target_frame_prim_path=None,
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
    device.__enter__.assert_not_called()
    with pytest.raises(RuntimeError, match="start"):
        source.advance()
    source.start()
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


def test_host_and_operator_resets_are_independent(capture):
    """Only the controlled row resets input; each device reset event reaches playback."""
    device, _, env = capture
    source = CaptureTeleopSource(CaptureTeleopSourceCfg(device=IsaacTeleopCfg()), env, env_ids=[2])
    source.start()
    try:
        source.reset([])
        source.reset([0, 1])
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
        # The device owns returned-frame deduplication. Two distinct reset
        # frames can arrive consecutively without an intervening false pulse.
        assert source.advance().reset
        source.reset([2])
        device.last_control_events = ControlEvents(is_active=False)
        assert not source.advance().reset
        device.last_control_events = ControlEvents(is_active=False, should_reset=True)
        assert source.advance().reset
    finally:
        source.close()


def test_xr_borrows_presentation_handles_and_target_frame(capture, monkeypatch):
    """XR uses the owner's handles and physics transform without starting another runtime."""
    from isaaclab_teleop import play_source

    device, create, env = capture
    device.target_frame_prim_path = "/World/envs/env_2/robot/base"
    transform = object()
    session = SimpleNamespace(external_oxr_handles=Mock(), get_target_frame_T_world=Mock(return_value=transform))
    monkeypatch.setattr(play_source, "_desktop_pipeline", Mock(side_effect=AssertionError("desktop validation")))
    cfg = CaptureTeleopSourceCfg(device=IsaacTeleopCfg(), start_active=False)
    source = CaptureTeleopSource(cfg, env, env_ids=[2], xr_session=session)
    source.start()
    try:
        assert source.device is device
        assert create.call_args.kwargs["external_oxr_handles_provider"] is session.external_oxr_handles
        assert create.call_args.kwargs["cloudxr_env_file"] is None
        assert not create.call_args.kwargs["auto_launch_cloudxr"]
        assert not create.call_args.kwargs["use_kit_xr_bridge"]
        device.request_start.assert_not_called()
        source.advance()
        session.get_target_frame_T_world.assert_called_once_with("/World/envs/env_2/robot/base")
        device.advance.assert_called_once_with(transform)
    finally:
        source.close()


def test_failed_input_drain_can_be_retried(capture):
    """A close failure stays observable so playback retains the borrowed XR environment."""
    device, _, env = capture
    source = CaptureTeleopSource(CaptureTeleopSourceCfg(device=IsaacTeleopCfg()), env, env_ids=[0])
    source.start()
    device.__exit__.side_effect = [RuntimeError("input did not drain"), None]
    device.session_cleanup_failed = True
    with pytest.raises(RuntimeError, match="input did not drain"):
        source.close()
    assert source.session_cleanup_failed
    device.session_cleanup_failed = False
    source.close()
    source.close()
    assert device.__exit__.call_count == 2
    assert not source.session_cleanup_failed


def test_start_failure_retains_input_for_owner_cleanup(capture):
    """The owner retains the source even if native startup and its cleanup both fail."""
    device, _, env = capture
    device.__enter__.side_effect = RuntimeError("runtime failed during startup")
    device.__exit__.side_effect = [RuntimeError("cleanup failed"), None]
    source = CaptureTeleopSource(CaptureTeleopSourceCfg(device=IsaacTeleopCfg()), env, env_ids=[0])
    with pytest.raises(RuntimeError, match="runtime failed"):
        source.start()
    with pytest.raises(RuntimeError, match="cleanup failed"):
        source.close()
    source.close()
    assert device.__exit__.call_count == 2
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
    with pytest.raises(ValueError, match="task-configured XR presentation"):
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
