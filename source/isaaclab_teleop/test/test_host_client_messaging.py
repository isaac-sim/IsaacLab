# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for host -> client messaging over the teleop control channel.

These run against the real ``isaacteleop`` message-channel nodes rather than
stubs, so they cover the actual wiring: the ``MessageChannelSource`` built into
the control pipeline must be handed the same outbound queue that
:meth:`send_client_message` enqueues into, since that queue is the only path
host-originated messages take to the wire.

No Omniverse/Isaac Sim stack is required: the lifecycle is constructed with
``use_kit_xr_bridge=False`` so it never touches Kit, and no session is created.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

_PURGED_ROOTS = ("isaacteleop", "isaaclab_teleop")

_UNSET = object()


@pytest.fixture(autouse=True)
def teleop(monkeypatch):
    """Yield ``isaaclab_teleop`` modules bound to the *real* ``isaacteleop``.

    Sibling test modules install ``MagicMock`` stubs for ``isaacteleop`` into
    ``sys.modules`` at import time and never remove them.  Purging those alone is
    not enough: ``session_lifecycle`` binds names such as
    ``TeleopMessageProcessor`` at module scope, so a module imported while the
    stubs were active keeps holding mocks.  Both package trees are therefore
    purged and re-imported here, which is what makes these assertions test the
    real graph wiring rather than mock behaviour.

    Everything is restored afterwards so the stub-based tests are unaffected.
    """
    import sys
    from types import SimpleNamespace

    saved = {name: module for name, module in sys.modules.items() if name.split(".")[0] in _PURGED_ROOTS}
    for name in saved:
        del sys.modules[name]

    try:
        from isaaclab_teleop.control_events import TELEOP_CONTROL_CHANNEL_UUID
        from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg
        from isaaclab_teleop.session_lifecycle import TeleopSessionLifecycle
    except ImportError:
        for name in [n for n in sys.modules if n.split(".")[0] in _PURGED_ROOTS]:
            del sys.modules[name]
        sys.modules.update(saved)
        pytest.skip("isaacteleop is required for message-channel wiring tests")

    try:
        yield SimpleNamespace(
            TELEOP_CONTROL_CHANNEL_UUID=TELEOP_CONTROL_CHANNEL_UUID,
            IsaacTeleopCfg=IsaacTeleopCfg,
            TeleopSessionLifecycle=TeleopSessionLifecycle,
        )
    finally:
        for name in [n for n in sys.modules if n.split(".")[0] in _PURGED_ROOTS]:
            del sys.modules[name]
        sys.modules.update(saved)


def _make_lifecycle(teleop, control_channel_uuid: bytes | None = _UNSET):
    """Build a lifecycle that never touches Kit or CloudXR."""
    if control_channel_uuid is _UNSET:
        control_channel_uuid = teleop.TELEOP_CONTROL_CHANNEL_UUID
    cfg = teleop.IsaacTeleopCfg(
        pipeline_builder=lambda: MagicMock(),
        control_channel_uuid=control_channel_uuid,
    )
    return teleop.TeleopSessionLifecycle(cfg, cloudxr_env_file=None, use_kit_xr_bridge=False)


class TestConstructionWithoutAKitApp:
    """The lifecycle must construct when carb is present but no Kit app runs.

    Regression coverage for a crash that only appeared in CI: locally
    ``omni.kit.app`` is not importable at all, so the import guard swallowed the
    whole block, while a full Isaac Sim image imports it fine and
    ``get_eventdispatcher()`` returns ``None`` because no app is running.
    Calling ``observe_event()`` on that ``None`` aborted construction.
    """

    def _fake_kit_modules(self, monkeypatch, dispatcher):
        """Make ``omni.kit.app`` and ``carb.eventdispatcher`` importable."""
        import sys
        from types import ModuleType

        carb = ModuleType("carb")
        carb_ed = ModuleType("carb.eventdispatcher")
        carb_ed.get_eventdispatcher = lambda: dispatcher  # type: ignore[attr-defined]
        carb.eventdispatcher = carb_ed  # type: ignore[attr-defined]

        omni = ModuleType("omni")
        omni_kit = ModuleType("omni.kit")
        omni_kit_app = ModuleType("omni.kit.app")
        omni_kit_app.GLOBAL_EVENT_PRE_SHUTDOWN = "pre_shutdown"  # type: ignore[attr-defined]
        omni.kit = omni_kit  # type: ignore[attr-defined]
        omni_kit.app = omni_kit_app  # type: ignore[attr-defined]

        for name, module in (
            ("carb", carb),
            ("carb.eventdispatcher", carb_ed),
            ("omni", omni),
            ("omni.kit", omni_kit),
            ("omni.kit.app", omni_kit_app),
        ):
            monkeypatch.setitem(sys.modules, name, module)

    def test_constructs_when_the_dispatcher_is_none(self, teleop, monkeypatch):
        self._fake_kit_modules(monkeypatch, dispatcher=None)
        lifecycle = _make_lifecycle(teleop)
        assert lifecycle is not None

    def test_subscribes_when_a_dispatcher_exists(self, teleop, monkeypatch):
        # The happy path must keep working: a real Kit app still gets the
        # pre-shutdown observer that tears the session down before XRCore.
        dispatcher = MagicMock()
        self._fake_kit_modules(monkeypatch, dispatcher=dispatcher)

        _make_lifecycle(teleop)

        dispatcher.observe_event.assert_called_once()
        assert dispatcher.observe_event.call_args.kwargs["order"] == -100


class TestControlPipelineWiring:
    """The outbound half of the control channel."""

    def test_control_pipeline_builds(self, teleop):
        lifecycle = _make_lifecycle(teleop)
        pipeline, processor = lifecycle._build_control_pipeline(teleop.TELEOP_CONTROL_CHANNEL_UUID)
        assert pipeline is not None
        assert processor is not None

    def test_source_receives_the_lifecycle_outbound_queue(self, teleop):
        # The source only puts bytes on the wire from the queue it was handed.
        # If the lifecycle enqueued into a different deque every host message
        # would be silently dropped -- this is the crux of the outbound path.
        from isaacteleop.retargeting_engine.deviceio_source_nodes import MessageChannelSource

        lifecycle = _make_lifecycle(teleop)
        pipeline, _ = lifecycle._build_control_pipeline(teleop.TELEOP_CONTROL_CHANNEL_UUID)

        sources = [n for n in pipeline.get_leaf_nodes() if isinstance(n, MessageChannelSource)]
        assert len(sources) == 1
        assert sources[0]._outbound_queue is lifecycle._pending_client_messages

    def test_source_name_matches_the_recorded_replay_channel(self, teleop):
        # Replay matches the recorded channel by node name; renaming this breaks
        # MCAP replay of teleop control.
        from isaacteleop.retargeting_engine.deviceio_source_nodes import MessageChannelSource

        lifecycle = _make_lifecycle(teleop)
        pipeline, _ = lifecycle._build_control_pipeline(teleop.TELEOP_CONTROL_CHANNEL_UUID)

        (source,) = [n for n in pipeline.get_leaf_nodes() if isinstance(n, MessageChannelSource)]
        assert source.name == "_teleop_control_source"

    def test_message_channel_sink_is_not_a_device_sink(self, teleop):
        # Regression guard. The outbound path deliberately does NOT build a
        # MessageChannelSink and does NOT put one in
        # TeleopSessionConfig(sinks=...): unlike HapticSink it is a plain
        # BaseRetargeter, so _validate_sinks rejects it and the session dies at
        # start with "sinks entries must be an IDeviceIOSink". Delivery goes
        # through the source's outbound queue instead. If upstream ever makes
        # this an IDeviceIOSink, revisit that decision.
        from isaacteleop.retargeting_engine.deviceio_source_nodes import IDeviceIOSink, MessageChannelSink
        from isaacteleop.retargeting_engine.deviceio_source_nodes.haptic_sink import HapticSink

        assert not issubclass(MessageChannelSink, IDeviceIOSink)
        assert issubclass(HapticSink, IDeviceIOSink)

    def test_queued_message_is_shaped_the_way_the_source_drains_it(self, teleop):
        # poll_tracker iterates batch.data and reads .payload off each entry.
        lifecycle = _make_lifecycle(teleop)
        lifecycle.send_client_message({"type": "system_notice"})

        (batch,) = lifecycle._pending_client_messages
        assert [m.payload for m in batch.data] == [b'{"type": "system_notice"}']


class TestSendClientMessage:
    """Queueing behaviour of the public send API."""

    def test_message_is_queued_as_encoded_json(self, teleop):
        lifecycle = _make_lifecycle(teleop)
        lifecycle.send_client_message({"type": "system_notice", "message": {"level": "warning"}})

        (batch,) = lifecycle._pending_client_messages
        assert json.loads(batch.data[0].payload.decode())["type"] == "system_notice"

    def test_no_control_channel_drops_the_message(self, teleop):
        lifecycle = _make_lifecycle(teleop, control_channel_uuid=None)
        lifecycle.send_client_message({"type": "system_notice"})
        assert not lifecycle._pending_client_messages

    def test_non_serializable_message_is_dropped_without_raising(self, teleop):
        # A bad payload must not propagate into the teleop step loop.
        lifecycle = _make_lifecycle(teleop)
        lifecycle.send_client_message({"bad": object()})
        assert not lifecycle._pending_client_messages

    def test_oversized_message_is_dropped(self, teleop):
        lifecycle = _make_lifecycle(teleop)
        lifecycle.send_client_message({"type": "x", "blob": "a" * (64 * 1024)})
        assert not lifecycle._pending_client_messages

    def test_queue_is_bounded_and_drops_oldest(self, teleop):
        lifecycle = _make_lifecycle(teleop)
        for index in range(100):
            lifecycle.send_client_message({"index": index})

        assert len(lifecycle._pending_client_messages) == 32
        assert json.loads(lifecycle._pending_client_messages[-1].data[0].payload.decode())["index"] == 99


class TestQueueLifetimeAcrossStart:
    """Messages queued before ``start()`` must survive it.

    ``start()`` runs from ``IsaacTeleopDevice.__enter__``, which the teleop
    scripts reach hundreds of lines after building the device, so there is a
    wide window in which a caller can queue a message. Clearing the queue in
    ``start()`` dropped those silently, contradicting what
    :meth:`send_client_message` documents.
    """

    def _start_without_a_session(self, lifecycle, monkeypatch):
        """Run ``start()`` far enough to exercise the queue, with no XR stack."""
        from isaaclab_teleop import system_check

        # Neutralize everything start() does after the queue handling: the
        # pipeline build and session creation need Kit/OpenXR.
        monkeypatch.setattr(
            system_check, "check_system_requirements", lambda device=None: system_check.SystemCheckResult()
        )
        monkeypatch.setattr(type(lifecycle), "_build_combined_pipeline", lambda self, pipeline: MagicMock())
        monkeypatch.setattr(type(lifecycle), "_try_start_session", lambda self: False)
        lifecycle.start()

    def test_message_queued_before_start_survives(self, teleop, monkeypatch):
        lifecycle = _make_lifecycle(teleop)
        lifecycle.send_client_message({"type": "custom", "message": {"n": 1}})

        self._start_without_a_session(lifecycle, monkeypatch)

        (batch,) = lifecycle._pending_client_messages
        assert json.loads(batch.data[0].payload.decode())["type"] == "custom"

    def test_start_still_queues_its_own_notice(self, teleop, monkeypatch):
        # The path that already works must keep working: a failing check queues
        # a notice during start() even with nothing queued beforehand.
        from isaaclab_teleop import system_check

        lifecycle = _make_lifecycle(teleop)
        monkeypatch.setattr(
            system_check,
            "check_system_requirements",
            lambda device=None: system_check.SystemCheckResult(
                items=(system_check.SystemCheckItem("CPU governor", False, "powersave", "performance"),)
            ),
        )
        monkeypatch.setattr(type(lifecycle), "_build_combined_pipeline", lambda self, pipeline: MagicMock())
        monkeypatch.setattr(type(lifecycle), "_try_start_session", lambda self: False)

        lifecycle.start()

        (batch,) = lifecycle._pending_client_messages
        assert json.loads(batch.data[0].payload.decode())["type"] == "system_notice"

    def test_caller_message_and_startup_notice_coexist(self, teleop, monkeypatch):
        from isaaclab_teleop import system_check

        lifecycle = _make_lifecycle(teleop)
        lifecycle.send_client_message({"type": "custom"})
        monkeypatch.setattr(
            system_check,
            "check_system_requirements",
            lambda device=None: system_check.SystemCheckResult(
                items=(system_check.SystemCheckItem("CPU governor", False, "powersave", "performance"),)
            ),
        )
        monkeypatch.setattr(type(lifecycle), "_build_combined_pipeline", lambda self, pipeline: MagicMock())
        monkeypatch.setattr(type(lifecycle), "_try_start_session", lambda self: False)

        lifecycle.start()

        types = [json.loads(b.data[0].payload.decode())["type"] for b in lifecycle._pending_client_messages]
        assert types == ["custom", "system_notice"]

    def test_stop_clears_undelivered_messages(self, teleop, monkeypatch):
        # stop() remains the session-boundary cleanup, so a notice from one
        # session cannot leak into the next.
        lifecycle = _make_lifecycle(teleop)
        self._start_without_a_session(lifecycle, monkeypatch)
        lifecycle.send_client_message({"type": "custom"})

        lifecycle.stop()

        assert not lifecycle._pending_client_messages


class TestSystemCheckOnStart:
    """The startup check feeding the client notice."""

    def test_failing_check_queues_a_notice(self, teleop, monkeypatch):
        from isaaclab_teleop import system_check

        lifecycle = _make_lifecycle(teleop)
        monkeypatch.setattr(
            system_check,
            "check_system_requirements",
            lambda device=None: system_check.SystemCheckResult(
                items=(system_check.SystemCheckItem("CPU governor", False, "powersave", "performance"),)
            ),
        )

        lifecycle._run_system_check()

        (batch,) = lifecycle._pending_client_messages
        assert json.loads(batch.data[0].payload.decode())["type"] == "system_notice"

    def test_passing_check_queues_nothing(self, teleop, monkeypatch):
        from isaaclab_teleop import system_check

        lifecycle = _make_lifecycle(teleop)
        monkeypatch.setattr(
            system_check,
            "check_system_requirements",
            lambda device=None: system_check.SystemCheckResult(
                items=(system_check.SystemCheckItem("CPU governor", True, "performance", "performance"),)
            ),
        )

        lifecycle._run_system_check()
        assert not lifecycle._pending_client_messages

    def test_check_raising_does_not_break_startup(self, teleop, monkeypatch):
        from isaaclab_teleop import system_check

        lifecycle = _make_lifecycle(teleop)

        def _boom(device=None):
            raise RuntimeError("probe exploded")

        monkeypatch.setattr(system_check, "check_system_requirements", _boom)

        lifecycle._run_system_check()
        assert not lifecycle._pending_client_messages
