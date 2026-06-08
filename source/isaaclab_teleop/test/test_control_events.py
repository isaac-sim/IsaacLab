# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for TeleopMessageProcessor, _classify_command, _extract_command,
and poll_control_events.

These tests exercise pure logic (no Omniverse/Isaac Sim stack required).
The message processor is tested by calling its ``_compute_fn`` method
directly with fake pipeline I/O, mirroring how TeleopCore's
``teleop_control_pipeline`` mechanism invokes it.
"""

from __future__ import annotations

import dataclasses
import json
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

    iface = sys.modules["isaacteleop.retargeting_engine.interface"]
    iface.ExecutionState = ExecutionState  # type: ignore[attr-defined]
    iface.ExecutionEvents = ExecutionEvents  # type: ignore[attr-defined]
    iface.RetargeterIOType = dict  # type: ignore[attr-defined]

    class FakeBaseRetargeter:
        def __init__(self, name: str) -> None:
            self.name = name

    iface.BaseRetargeter = FakeBaseRetargeter  # type: ignore[attr-defined]

    tsm_types = sys.modules["isaacteleop.teleop_session_manager.teleop_state_manager_types"]
    tsm_types.bool_signal = MagicMock  # type: ignore[attr-defined]

    dt_mod = sys.modules["isaacteleop.retargeting_engine.deviceio_source_nodes.deviceio_tensor_types"]
    dt_mod.MessageChannelMessagesTrackedGroup = MagicMock  # type: ignore[attr-defined]


_install_stubs()

from isaaclab_teleop.control_events import ControlEvents, poll_control_events  # noqa: E402
from isaaclab_teleop.teleop_message_processor import (  # noqa: E402
    TeleopMessageProcessor,
    _classify_command,
    _extract_command,
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
    """Build a fake RetargeterIO dict for the processor."""
    tg = MagicMock()
    tg.__getitem__ = MagicMock(return_value=messages_tracked)
    return {TeleopMessageProcessor.INPUT_MESSAGES: tg}


class _FakeOutputSlot:
    """Captures ``outputs["key"][0] = value`` assignments."""

    def __init__(self):
        self.value = None

    def __setitem__(self, idx, val):
        self.value = val

    def __getitem__(self, idx):
        return self.value


def _make_outputs():
    """Build a fake outputs dict with capturable slots."""
    return {"run_toggle": _FakeOutputSlot(), "kill": _FakeOutputSlot(), "reset": _FakeOutputSlot()}


def _step(proc, messages_tracked) -> dict:
    """Run the processor's _compute_fn and return captured outputs."""
    inputs = _make_inputs(messages_tracked)
    outputs = _make_outputs()
    proc._compute_fn(inputs, outputs, context=None)
    return {k: v.value for k, v in outputs.items()}


# ===========================================================================
# TeleopMessageProcessor: basic command parsing
# ===========================================================================


class TestStartCommand:
    def test_start_sets_run_toggle(self):
        proc = TeleopMessageProcessor(name="test")
        result = _step(proc, _tracked(b"start"))
        assert result["run_toggle"] is True
        assert result["kill"] is False
        assert result["reset"] is False

    def test_start_does_not_set_reset(self):
        proc = TeleopMessageProcessor(name="test")
        result = _step(proc, _tracked(b"start"))
        assert result["reset"] is False


class TestStopCommand:
    def test_stop_from_stopped_is_noop(self):
        proc = TeleopMessageProcessor(name="test")
        result = _step(proc, _tracked(b"stop"))
        assert result["run_toggle"] is False
        assert result["kill"] is False


class TestResetCommand:
    def test_reset_sets_reset_flag(self):
        proc = TeleopMessageProcessor(name="test")
        result = _step(proc, _tracked(b"reset"))
        assert result["reset"] is True
        assert result["run_toggle"] is False
        assert result["kill"] is False


class TestResetPulseBehaviour:
    def test_reset_clears_on_next_step(self):
        proc = TeleopMessageProcessor(name="test")
        result = _step(proc, _tracked(b"reset"))
        assert result["reset"] is True

        result = _step(proc, _empty_tracked())
        assert result["reset"] is False


class TestKillAlwaysFalse:
    def test_kill_is_always_false(self):
        proc = TeleopMessageProcessor(name="test")
        for payload in [b"start", b"stop", b"reset", b"hello"]:
            result = _step(proc, _tracked(payload))
            assert result["kill"] is False


# ===========================================================================
# Shadow state and toggle sequences
# ===========================================================================


class TestStartFromStopped:
    """``start`` from STOPPED needs 2 toggle edges over 3 frames."""

    def test_full_sequence_reaches_running(self):
        proc = TeleopMessageProcessor(name="test")
        # Frame 0: "start" received, first toggle edge queued
        r0 = _step(proc, _tracked(b"start"))
        assert r0["run_toggle"] is True  # edge 1: STOPPED -> PAUSED

        # Frame 1: queue drains False (prev resets)
        r1 = _step(proc, _empty_tracked())
        assert r1["run_toggle"] is False

        # Frame 2: queue drains True (second edge)
        r2 = _step(proc, _empty_tracked())
        assert r2["run_toggle"] is True  # edge 2: PAUSED -> RUNNING

        # Frame 3: queue empty, back to idle
        r3 = _step(proc, _empty_tracked())
        assert r3["run_toggle"] is False

    def test_shadow_state_is_running_after_sequence(self):
        proc = TeleopMessageProcessor(name="test")
        _step(proc, _tracked(b"start"))
        _step(proc, _empty_tracked())
        _step(proc, _empty_tracked())
        assert proc._shadow_state == "running"


class TestStartFromPaused:
    """``start`` from PAUSED needs 1 toggle edge."""

    def test_single_edge_reaches_running(self):
        proc = TeleopMessageProcessor(name="test")
        # Drive to RUNNING: start sequence plays 3 frames
        _step(proc, _tracked(b"start"))
        _step(proc, _empty_tracked())
        _step(proc, _empty_tracked())
        assert proc._shadow_state == "running"

        # Stop to reach PAUSED (prev_toggle is True from start sequence,
        # so a False is prepended before the toggle edge)
        _step(proc, _tracked(b"stop"))  # drains False (prepended)
        r_stop_edge = _step(proc, _empty_tracked())  # drains True (edge)
        assert r_stop_edge["run_toggle"] is True
        assert proc._shadow_state == "paused"

        # Start from PAUSED: prev_toggle is True, so False prepended
        _step(proc, _tracked(b"start"))  # drains False (prepended)
        r_start_edge = _step(proc, _empty_tracked())  # drains True (edge)
        assert r_start_edge["run_toggle"] is True
        assert proc._shadow_state == "running"


class TestStartFromRunning:
    """``start`` when already RUNNING is a no-op."""

    def test_start_from_running_noop(self):
        proc = TeleopMessageProcessor(name="test")
        _step(proc, _tracked(b"start"))
        _step(proc, _empty_tracked())
        _step(proc, _empty_tracked())
        assert proc._shadow_state == "running"

        result = _step(proc, _tracked(b"start"))
        assert result["run_toggle"] is False


class TestStopFromRunning:
    """``stop`` from RUNNING uses one toggle edge to reach PAUSED."""

    def test_stop_pauses(self):
        proc = TeleopMessageProcessor(name="test")
        _step(proc, _tracked(b"start"))
        _step(proc, _empty_tracked())
        _step(proc, _empty_tracked())
        assert proc._shadow_state == "running"

        # prev_toggle is True, so stop prepends False before the edge
        r0 = _step(proc, _tracked(b"stop"))
        assert r0["run_toggle"] is False  # prepended False
        r1 = _step(proc, _empty_tracked())
        assert r1["run_toggle"] is True  # edge: RUNNING -> PAUSED
        assert proc._shadow_state == "paused"


class TestStopFromPaused:
    """``stop`` when already PAUSED is a no-op."""

    def test_stop_from_paused_noop(self):
        proc = TeleopMessageProcessor(name="test")
        _step(proc, _tracked(b"start"))
        _step(proc, _empty_tracked())
        _step(proc, _empty_tracked())
        # Stop to PAUSED
        _step(proc, _tracked(b"stop"))
        _step(proc, _empty_tracked())
        assert proc._shadow_state == "paused"

        result = _step(proc, _tracked(b"stop"))
        assert result["run_toggle"] is False


class TestCommandDuringToggleSequence:
    """Commands received while a toggle sequence is in progress are ignored."""

    def test_second_start_during_sequence_ignored(self):
        proc = TeleopMessageProcessor(name="test")
        _step(proc, _tracked(b"start"))  # starts the 3-frame sequence
        # Second start during the sequence should not restart it
        r1 = _step(proc, _tracked(b"start"))
        assert r1["run_toggle"] is False  # draining the False from queue

        r2 = _step(proc, _empty_tracked())
        assert r2["run_toggle"] is True  # second edge fires normally


# ===========================================================================
# inject_reset
# ===========================================================================


class TestInjectReset:
    def test_inject_reset_produces_pulse(self):
        proc = TeleopMessageProcessor(name="test")
        proc.inject_reset()
        result = _step(proc, _empty_tracked())
        assert result["reset"] is True

    def test_inject_reset_clears_after_one_step(self):
        proc = TeleopMessageProcessor(name="test")
        proc.inject_reset()
        _step(proc, _empty_tracked())
        result = _step(proc, _empty_tracked())
        assert result["reset"] is False

    def test_inject_reset_combines_with_message_reset(self):
        proc = TeleopMessageProcessor(name="test")
        proc.inject_reset()
        result = _step(proc, _tracked(b"reset"))
        assert result["reset"] is True

    def test_inject_reset_independent_of_toggle(self):
        proc = TeleopMessageProcessor(name="test")
        proc.inject_reset()
        result = _step(proc, _tracked(b"start"))
        assert result["run_toggle"] is True
        assert result["reset"] is True


# ===========================================================================
# Word boundary matching
# ===========================================================================


class TestWordBoundaryMatching:
    @pytest.mark.parametrize("payload", [b"teleop start", b"xr start session", b"start now"])
    def test_start_word(self, payload: bytes):
        proc = TeleopMessageProcessor(name="test")
        result = _step(proc, _tracked(payload))
        assert result["run_toggle"] is True

    @pytest.mark.parametrize("payload", [b"teleop reset", b"env reset"])
    def test_reset_word(self, payload: bytes):
        proc = TeleopMessageProcessor(name="test")
        result = _step(proc, _tracked(payload))
        assert result["reset"] is True


class TestAmbiguousPayloads:
    def test_reset_wins_over_start(self):
        proc = TeleopMessageProcessor(name="test")
        result = _step(proc, _tracked(b"reset and start"))
        assert result["reset"] is True
        assert result["run_toggle"] is False


# ===========================================================================
# Empty, null, and malformed batches
# ===========================================================================


class TestEmptyAndNullBatches:
    def test_empty_data_list(self):
        proc = TeleopMessageProcessor(name="test")
        result = _step(proc, _empty_tracked())
        assert result["run_toggle"] is False
        assert result["kill"] is False
        assert result["reset"] is False

    def test_null_data(self):
        proc = TeleopMessageProcessor(name="test")
        result = _step(proc, _null_tracked())
        assert result["run_toggle"] is False

    def test_none_input(self):
        proc = TeleopMessageProcessor(name="test")
        result = _step(proc, None)
        assert result["run_toggle"] is False


class TestMultipleMessagesInBatch:
    def test_start_then_reset_in_one_batch(self):
        proc = TeleopMessageProcessor(name="test")
        result = _step(proc, _tracked(b"start", b"reset"))
        assert result["run_toggle"] is True
        assert result["reset"] is True


class TestMalformedPayloads:
    def test_invalid_utf8(self):
        proc = TeleopMessageProcessor(name="test")
        result = _step(proc, _tracked(b"\xff\xfe"))
        assert result["run_toggle"] is False
        assert result["kill"] is False
        assert result["reset"] is False

    def test_none_payload(self):
        proc = TeleopMessageProcessor(name="test")
        tracked = _FakeTracked(data=[_FakePayload(payload=None)])  # type: ignore[arg-type]
        result = _step(proc, tracked)
        assert result["run_toggle"] is False


# ===========================================================================
# JSON format tests (Quest client sends JSON teleop_command messages)
# ===========================================================================


def _json_command(command: str) -> bytes:
    """Build a Quest-style JSON teleop_command payload."""
    return json.dumps({"type": "teleop_command", "message": {"command": command}}).encode("utf-8")


class TestJsonFormat:
    def test_json_start_teleop(self):
        proc = TeleopMessageProcessor(name="test")
        result = _step(proc, _tracked(_json_command("start teleop")))
        assert result["run_toggle"] is True

    def test_json_stop_teleop_from_stopped_noop(self):
        proc = TeleopMessageProcessor(name="test")
        result = _step(proc, _tracked(_json_command("stop teleop")))
        assert result["run_toggle"] is False

    def test_json_reset_teleop(self):
        proc = TeleopMessageProcessor(name="test")
        result = _step(proc, _tracked(_json_command("reset teleop")))
        assert result["reset"] is True

    def test_json_wrong_type_ignored(self):
        payload = json.dumps({"type": "other_event", "message": {"command": "start"}}).encode("utf-8")
        proc = TeleopMessageProcessor(name="test")
        result = _step(proc, _tracked(payload))
        assert result["run_toggle"] is False

    def test_json_message_as_string(self):
        payload = json.dumps({"type": "teleop_command", "message": "start teleop"}).encode("utf-8")
        proc = TeleopMessageProcessor(name="test")
        result = _step(proc, _tracked(payload))
        assert result["run_toggle"] is True


# ===========================================================================
# _extract_command unit tests
# ===========================================================================


class TestExtractCommand:
    def test_plain_text(self):
        assert _extract_command("start teleop") == "start teleop"

    def test_json_teleop_command(self):
        text = json.dumps({"type": "teleop_command", "message": {"command": "stop"}})
        assert _extract_command(text) == "stop"

    def test_json_wrong_type(self):
        text = json.dumps({"type": "other", "message": {"command": "start"}})
        assert _extract_command(text) is None

    def test_json_no_message_key(self):
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
