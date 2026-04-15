# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for MessageChannelTeleopStateManager, _classify_command, _extract_command,
and poll_control_events.

These tests exercise pure logic (no Omniverse/Isaac Sim stack required).
The state manager is tested by calling its ``_compute_execution_events``
method directly with fake pipeline I/O, mirroring how TeleopCore's
``teleop_control_pipeline`` mechanism invokes it.
"""

from __future__ import annotations

import dataclasses
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub out isaacteleop modules before any isaaclab_teleop imports so the
# tests can run in a plain Python environment without Omniverse.
# ---------------------------------------------------------------------------

_MODULES_TO_STUB = [
    "isaacteleop",
    "isaacteleop.deviceio",
    "isaacteleop.deviceio_trackers",
    "isaacteleop.retargeting_engine",
    "isaacteleop.retargeting_engine.deviceio_source_nodes",
    "isaacteleop.retargeting_engine.deviceio_source_nodes.deviceio_tensor_types",
    "isaacteleop.retargeting_engine.interface",
    "isaacteleop.retargeting_engine.interface.retargeter_core_types",
    "isaacteleop.retargeting_engine.interface.tensor_group_type",
    "isaacteleop.retargeting_engine_ui",
    "isaacteleop.schema",
    "isaacteleop.teleop_session_manager",
    "isaacteleop.teleop_session_manager.teleop_state_manager_retargeter",
    "isaacteleop.teleop_session_manager.teleop_state_manager_types",
]

_stubs: dict[str, ModuleType | MagicMock] = {}


def _install_stubs():
    for name in _MODULES_TO_STUB:
        if name not in sys.modules:
            _stubs[name] = MagicMock()
            sys.modules[name] = _stubs[name]

    # Provide real ExecutionState and ExecutionEvents so state logic works.
    from enum import Enum

    class ExecutionState(str, Enum):
        UNKNOWN = "unknown"
        STOPPED = "stopped"
        PAUSED = "paused"
        RUNNING = "running"

    @dataclasses.dataclass
    class ExecutionEvents:
        reset: bool = False
        execution_state: ExecutionState = ExecutionState.UNKNOWN

    ee_mod = sys.modules["isaacteleop.retargeting_engine.interface.execution_events"] = ModuleType(
        "isaacteleop.retargeting_engine.interface.execution_events"
    )
    ee_mod.ExecutionState = ExecutionState  # type: ignore[attr-defined]
    ee_mod.ExecutionEvents = ExecutionEvents  # type: ignore[attr-defined]

    # Make them available from the interface module too
    iface = sys.modules["isaacteleop.retargeting_engine.interface"]
    iface.ExecutionState = ExecutionState  # type: ignore[attr-defined]
    iface.ExecutionEvents = ExecutionEvents  # type: ignore[attr-defined]
    iface.RetargeterIOType = dict  # type: ignore[attr-defined]

    # Provide a minimal TeleopStateManager base so the subclass can instantiate
    class FakeTeleopStateManager:
        def __init__(self, name: str) -> None:
            self.name = name

    tsm_mod = sys.modules["isaacteleop.teleop_session_manager.teleop_state_manager_retargeter"]
    tsm_mod.TeleopStateManager = FakeTeleopStateManager  # type: ignore[attr-defined]

    # MessageChannelMessagesTrackedGroup stub
    dt_mod = sys.modules["isaacteleop.retargeting_engine.deviceio_source_nodes.deviceio_tensor_types"]
    dt_mod.MessageChannelMessagesTrackedGroup = MagicMock  # type: ignore[attr-defined]


_install_stubs()

from isaaclab_teleop.control_events import ControlEvents, poll_control_events  # noqa: E402
from isaaclab_teleop.message_channel_state_manager import (  # noqa: E402
    MessageChannelTeleopStateManager,
    _classify_command,
    _extract_command,
)

# Re-import after stubs so we can reference them in assertions.
from isaacteleop.retargeting_engine.interface.execution_events import (  # noqa: E402
    ExecutionEvents,
    ExecutionState,
)

# ---------------------------------------------------------------------------
# Test doubles for MessageChannelMessagesTrackedT
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _FakePayload:
    payload: bytes


@dataclasses.dataclass
class _FakeTracked:
    data: list[_FakePayload] | None = None


def _tracked(*payloads: bytes) -> _FakeTracked:
    """Build a lightweight stand-in for ``MessageChannelMessagesTrackedT``."""
    return _FakeTracked(data=[_FakePayload(p) for p in payloads])


def _empty_tracked() -> _FakeTracked:
    return _FakeTracked(data=[])


def _null_tracked() -> _FakeTracked:
    return _FakeTracked(data=None)


def _make_inputs(messages_tracked):
    """Build a fake RetargeterIO dict for the state manager."""
    tg = MagicMock()
    tg.__getitem__ = MagicMock(return_value=messages_tracked)
    return {MessageChannelTeleopStateManager.INPUT_MESSAGES: tg}


def _step(mgr, messages_tracked) -> ExecutionEvents:
    """Invoke the state manager's compute with fake inputs and return the events."""
    inputs = _make_inputs(messages_tracked)
    return mgr._compute_execution_events(inputs, context=None)


# ===========================================================================
# MessageChannelTeleopStateManager tests
# ===========================================================================


class TestInitialState:
    def test_defaults(self):
        mgr = MessageChannelTeleopStateManager(name="test")
        events = mgr.last_control_events
        assert events.is_active is None
        assert events.should_reset is False


class TestStartMessage:
    def test_start_sets_running(self):
        mgr = MessageChannelTeleopStateManager(name="test")
        result = _step(mgr, _tracked(b"start"))
        assert result.execution_state == ExecutionState.RUNNING
        assert mgr.last_control_events.is_active is True

    def test_start_does_not_set_reset(self):
        mgr = MessageChannelTeleopStateManager(name="test")
        result = _step(mgr, _tracked(b"start"))
        assert result.reset is False
        assert mgr.last_control_events.should_reset is False


class TestStopMessage:
    def test_stop_sets_paused(self):
        mgr = MessageChannelTeleopStateManager(name="test")
        result = _step(mgr, _tracked(b"stop"))
        assert result.execution_state == ExecutionState.PAUSED
        assert mgr.last_control_events.is_active is False


class TestResetMessage:
    def test_reset_sets_flags(self):
        mgr = MessageChannelTeleopStateManager(name="test")
        result = _step(mgr, _tracked(b"reset"))
        assert result.reset is True
        assert mgr.last_control_events.should_reset is True

    def test_reset_does_not_change_active_state(self):
        mgr = MessageChannelTeleopStateManager(name="test")
        _step(mgr, _tracked(b"start"))
        _step(mgr, _tracked(b"reset"))
        assert mgr.last_control_events.is_active is True


class TestResetPulseBehaviour:
    def test_should_reset_clears_on_next_step(self):
        mgr = MessageChannelTeleopStateManager(name="test")
        _step(mgr, _tracked(b"reset"))
        assert mgr.last_control_events.should_reset is True

        _step(mgr, _empty_tracked())
        assert mgr.last_control_events.should_reset is False


class TestWordBoundaryMatching:
    @pytest.mark.parametrize("payload", [b"teleop start", b"xr start session", b"start now"])
    def test_start_word(self, payload: bytes):
        mgr = MessageChannelTeleopStateManager(name="test")
        result = _step(mgr, _tracked(payload))
        assert result.execution_state == ExecutionState.RUNNING

    @pytest.mark.parametrize("payload", [b"teleop stop", b"stop teleop"])
    def test_stop_word(self, payload: bytes):
        mgr = MessageChannelTeleopStateManager(name="test")
        result = _step(mgr, _tracked(payload))
        assert result.execution_state == ExecutionState.PAUSED

    @pytest.mark.parametrize("payload", [b"teleop reset", b"env reset"])
    def test_reset_word(self, payload: bytes):
        mgr = MessageChannelTeleopStateManager(name="test")
        result = _step(mgr, _tracked(payload))
        assert result.reset is True


class TestAmbiguousPayloads:
    def test_stop_wins_over_start_when_both_present(self):
        mgr = MessageChannelTeleopStateManager(name="test")
        result = _step(mgr, _tracked(b"stop and start"))
        assert result.execution_state == ExecutionState.PAUSED

    def test_reset_wins_over_start_when_both_present(self):
        mgr = MessageChannelTeleopStateManager(name="test")
        result = _step(mgr, _tracked(b"reset and start"))
        assert result.reset is True


class TestEmptyAndNullBatches:
    def test_empty_data_list(self):
        mgr = MessageChannelTeleopStateManager(name="test")
        _step(mgr, _tracked(b"start"))
        _step(mgr, _empty_tracked())
        assert mgr.last_control_events.is_active is True

    def test_null_data(self):
        mgr = MessageChannelTeleopStateManager(name="test")
        _step(mgr, _null_tracked())
        assert mgr.last_control_events.is_active is None

    def test_none_input(self):
        mgr = MessageChannelTeleopStateManager(name="test")
        _step(mgr, None)
        assert mgr.last_control_events.is_active is None


class TestMultipleMessagesInBatch:
    def test_start_then_reset_in_one_batch(self):
        mgr = MessageChannelTeleopStateManager(name="test")
        result = _step(mgr, _tracked(b"start", b"reset"))
        assert mgr.last_control_events.is_active is True
        assert result.reset is True


class TestSequentialStartStop:
    def test_start_then_stop(self):
        mgr = MessageChannelTeleopStateManager(name="test")
        _step(mgr, _tracked(b"start"))
        assert mgr.last_control_events.is_active is True
        result = _step(mgr, _tracked(b"stop"))
        assert mgr.last_control_events.is_active is False
        assert result.execution_state == ExecutionState.PAUSED


class TestMalformedPayloads:
    def test_invalid_utf8(self):
        mgr = MessageChannelTeleopStateManager(name="test")
        _step(mgr, _tracked(b"\xff\xfe"))
        assert mgr.last_control_events.is_active is None
        assert mgr.last_control_events.should_reset is False

    def test_none_payload(self):
        mgr = MessageChannelTeleopStateManager(name="test")
        tracked = _FakeTracked(data=[_FakePayload(payload=None)])  # type: ignore[arg-type]
        _step(mgr, tracked)
        assert mgr.last_control_events.is_active is None


# ===========================================================================
# JSON format tests (Quest client sends JSON teleop_command messages)
# ===========================================================================


def _json_command(command: str) -> bytes:
    """Build a Quest-style JSON teleop_command payload."""
    import json

    return json.dumps({"type": "teleop_command", "message": {"command": command}}).encode("utf-8")


class TestJsonFormat:
    def test_json_start_teleop(self):
        mgr = MessageChannelTeleopStateManager(name="test")
        result = _step(mgr, _tracked(_json_command("start teleop")))
        assert result.execution_state == ExecutionState.RUNNING

    def test_json_stop_teleop(self):
        mgr = MessageChannelTeleopStateManager(name="test")
        result = _step(mgr, _tracked(_json_command("stop teleop")))
        assert result.execution_state == ExecutionState.PAUSED

    def test_json_reset_teleop(self):
        mgr = MessageChannelTeleopStateManager(name="test")
        result = _step(mgr, _tracked(_json_command("reset teleop")))
        assert result.reset is True

    def test_json_wrong_type_ignored(self):
        import json

        payload = json.dumps({"type": "other_event", "message": {"command": "start"}}).encode("utf-8")
        mgr = MessageChannelTeleopStateManager(name="test")
        _step(mgr, _tracked(payload))
        assert mgr.last_control_events.is_active is None

    def test_json_message_as_string(self):
        import json

        payload = json.dumps({"type": "teleop_command", "message": "start teleop"}).encode("utf-8")
        mgr = MessageChannelTeleopStateManager(name="test")
        result = _step(mgr, _tracked(payload))
        assert result.execution_state == ExecutionState.RUNNING


# ===========================================================================
# _extract_command unit tests
# ===========================================================================


class TestExtractCommand:
    def test_plain_text(self):
        assert _extract_command("start teleop") == "start teleop"

    def test_json_teleop_command(self):
        import json

        text = json.dumps({"type": "teleop_command", "message": {"command": "stop"}})
        assert _extract_command(text) == "stop"

    def test_json_wrong_type(self):
        import json

        text = json.dumps({"type": "other", "message": {"command": "start"}})
        assert _extract_command(text) is None

    def test_json_no_message_key(self):
        import json

        text = json.dumps({"type": "teleop_command"})
        assert _extract_command(text) is None

    def test_json_non_dict_value_returns_none(self):
        assert _extract_command("42") is None
        assert _extract_command("[1, 2, 3]") is None
        assert _extract_command("true") is None


# ===========================================================================
# _classify_command unit tests
# ===========================================================================


class TestClassifyCommand:
    def test_exact_words(self):
        assert _classify_command("start") == "start"
        assert _classify_command("stop") == "stop"
        assert _classify_command("reset") == "reset"

    def test_word_boundary_prevents_false_match(self):
        assert _classify_command("upstart") is None
        assert _classify_command("nonstop") is None
        assert _classify_command("unreset") is None

    def test_reset_beats_start(self):
        assert _classify_command("reset and start") == "reset"

    def test_stop_beats_start(self):
        assert _classify_command("stop and start") == "stop"

    def test_unrecognized_text(self):
        assert _classify_command("hello world") is None

    def test_case_insensitive(self):
        assert _classify_command("START") == "start"
        assert _classify_command("Stop Teleop") == "stop"
        assert _classify_command("RESET NOW") == "reset"


# ===========================================================================
# poll_control_events tests
# ===========================================================================


class TestPollControlEvents:
    def test_plain_object_returns_default(self):
        result = poll_control_events(object())
        assert result.is_active is None
        assert result.should_reset is False

    def test_device_with_control_events(self):
        class FakeDevice:
            @property
            def last_control_events(self):
                return ControlEvents(is_active=True, should_reset=True)

        result = poll_control_events(FakeDevice())
        assert result.is_active is True
        assert result.should_reset is True

    def test_device_with_none_events(self):
        class FakeDevice:
            last_control_events = None

        result = poll_control_events(FakeDevice())
        assert result.is_active is None
        assert result.should_reset is False

    def test_duck_typed_snapshot(self):
        class FakeSnapshot:
            is_active = False
            should_reset = True

        class FakeDevice:
            @property
            def last_control_events(self):
                return FakeSnapshot()

        result = poll_control_events(FakeDevice())
        assert result.is_active is False
        assert result.should_reset is True
