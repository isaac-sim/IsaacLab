# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Host-reset notifications through Capture's real graph and asynchronous runner."""

import threading
import time

import numpy as np
import pytest

pytest.importorskip("isaacteleop")

from isaaclab_teleop import IsaacTeleopCfg
from isaaclab_teleop.session_lifecycle import TeleopSessionLifecycle
from isaacteleop.retargeting_engine.interface import (
    BaseRetargeter,
    ComputeContext,
    ExecutionEvents,
    ExecutionState,
    TensorGroup,
    TensorGroupType,
)
from isaacteleop.retargeting_engine.interface.retargeter_core_types import ExecutionCache
from isaacteleop.retargeting_engine.tensor_types import NDArrayType
from isaacteleop.teleop_session_manager import RetargetingExecutionConfig, TeleopSession, TeleopSessionConfig
from isaacteleop.teleop_session_manager.async_retarget_runner import AsyncRetargetRunner, StepRequest

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


class _ResetProbe(BaseRetargeter):
    """Record the reset and execution state delivered to a real retargeter."""

    def __init__(self):
        self.events = []
        self.fail = False
        super().__init__("reset_probe")

    def input_spec(self):
        return {}

    def output_spec(self):
        return {"action": TensorGroupType("action", [NDArrayType("reset_seen", (1,))])}

    def _compute_fn(self, inputs, outputs, context):
        self.events.append((context.execution_events.reset, context.execution_events.execution_state))
        if self.fail:
            self.fail = False
            raise RuntimeError("retargeter failed")
        outputs["action"][0] = np.array([context.execution_events.reset], dtype=np.float32)


@pytest.fixture
def reset_pipeline():
    probe = _ResetProbe()
    lifecycle = TeleopSessionLifecycle(
        IsaacTeleopCfg(pipeline_builder=lambda: probe, sim_device="cpu"), use_kit_xr_bridge=False
    )
    pipeline = lifecycle._build_combined_pipeline(probe)
    controller_inputs = {}
    for name, group_type in lifecycle._button_controllers.input_spec().items():
        group = TensorGroup(group_type)
        group[0] = None
        controller_inputs[name] = group
    inputs = {probe.name: {}, lifecycle._button_controllers.name: controller_inputs}
    return lifecycle, pipeline, probe, inputs


def test_host_notification_preserves_operator_events_and_shared_cache(reset_pipeline):
    """Both normal and sink/shared-cache execution preserve the original control context."""
    lifecycle, pipeline, probe, inputs = reset_pipeline
    context = ComputeContext(execution_events=ExecutionEvents(execution_state=ExecutionState.RUNNING))
    lifecycle.notify_reset()
    pipeline.execute_pipeline(inputs, context)
    assert probe.events == [(True, ExecutionState.RUNNING)]
    assert not context.execution_events.reset

    # A simultaneous operator reset and pause remain visible to the host.
    operator_context = ComputeContext(
        execution_events=ExecutionEvents(reset=True, execution_state=ExecutionState.PAUSED)
    )
    cache = ExecutionCache(inputs, operator_context)
    lifecycle.notify_reset()
    pipeline.execute_pipeline_with_cache(cache)
    assert probe.events[-1] == (True, ExecutionState.PAUSED)
    assert cache.context is operator_context
    assert operator_context.execution_events.reset
    assert operator_context.execution_events.execution_state == ExecutionState.PAUSED

    # Revisiting an already computed frame must not consume a new host notification.
    lifecycle.notify_reset()
    pipeline.execute_pipeline_with_cache(cache)
    assert len(probe.events) == 2
    pipeline.execute_pipeline(inputs, context)
    pipeline.execute_pipeline(inputs, context)
    assert probe.events[-2:] == [(True, ExecutionState.RUNNING), (False, ExecutionState.RUNNING)]


def test_failed_execution_restores_context_and_retries_notification(reset_pipeline):
    """A graph failure cannot leak a host reset into control state or discard the notification."""
    lifecycle, pipeline, probe, inputs = reset_pipeline
    context = ComputeContext(execution_events=ExecutionEvents(execution_state=ExecutionState.RUNNING))
    cache = ExecutionCache(inputs, context)
    lifecycle.notify_reset()
    probe.fail = True
    with pytest.raises(RuntimeError, match="retargeter failed"):
        pipeline.execute_pipeline_with_cache(cache)
    assert cache.context is context
    assert not context.execution_events.reset

    pipeline.execute_pipeline(inputs, context)
    pipeline.execute_pipeline(inputs, context)
    assert probe.events == [
        (True, ExecutionState.RUNNING),
        (True, ExecutionState.RUNNING),
        (False, ExecutionState.RUNNING),
    ]


def test_async_notification_survives_dropped_requests_without_echo(reset_pipeline):
    """Latest-frame reuse and dropped submissions do not turn a host reset into an operator event."""
    lifecycle, pipeline, probe, inputs = reset_pipeline
    started = threading.Event()
    release = threading.Event()

    def execute(request):
        if request.frame_id == 1:
            started.set()
            assert release.wait(5.0), "test did not release the first worker frame"
        context = ComputeContext(execution_events=request.execution_events)
        return pipeline.execute_pipeline(inputs, context), context

    def request(frame_id, *, reset=False, state=ExecutionState.RUNNING):
        return StepRequest(frame_id, None, None, ExecutionEvents(reset=reset, execution_state=state), time.monotonic())

    runner = AsyncRetargetRunner(execute, RetargetingExecutionConfig(mode="pipelined"))
    runner.start()
    try:
        runner.submit(request(1))
        assert started.wait(5.0), "worker did not begin the first frame"
        lifecycle.notify_reset()
        runner.submit(request(2))
        assert runner.submit(request(3)) == 1
        release.set()
        latest = runner.wait_for_frame(3, timeout_s=5.0)
        assert latest is not None and latest.frame_id == 3
        assert probe.events == [(True, ExecutionState.RUNNING), (False, ExecutionState.RUNNING)]
        assert not latest.context.execution_events.reset
        repeated = runner.latest()
        assert repeated is not None and repeated.frame_id == latest.frame_id
        assert not repeated.context.execution_events.reset

        # The host reset completion was superseded; a later operator reset still reaches the host.
        runner.submit(request(4, reset=True, state=ExecutionState.PAUSED))
        latest = runner.wait_for_frame(4, timeout_s=5.0)
        assert latest is not None and latest.context.execution_events.reset
        assert latest.context.execution_events.execution_state == ExecutionState.PAUSED
        assert probe.events[-1] == (True, ExecutionState.PAUSED)
    finally:
        release.set()
        assert runner.stop(timeout_s=5.0)


def test_host_reset_withholds_old_actions_until_post_notification_result(reset_pipeline, monkeypatch):
    """Real Capture latest-frame reuse cannot drive a new episode with a pre-reset action."""
    lifecycle, pipeline, probe, inputs = reset_pipeline
    session = TeleopSession(
        TeleopSessionConfig(
            app_name="host_reset_test",
            pipeline=pipeline,
            retargeting_execution=RetargetingExecutionConfig(mode="pipelined"),
        )
    )
    entered = {frame: threading.Event() for frame in (1, 3, 5)}
    release = {frame: threading.Event() for frame in entered}

    def wait(frame):
        entered[frame].set()
        assert release[frame].wait(5.0), f"test did not release frame {frame}"

    def execute(request):
        # Replace only the native input phase; execute the real graph and let
        # Capture own submission IDs, dropped requests, and completed outputs.
        if request.frame_id == 5:
            wait(5)
        context = ComputeContext(
            execution_events=ExecutionEvents(reset=request.frame_id == 1, execution_state=ExecutionState.RUNNING)
        )
        result = pipeline.execute_pipeline(inputs, context)
        if request.frame_id in (1, 3):
            wait(request.frame_id)
        return result, context

    monkeypatch.setattr(session, "_execute_step_request", execute)
    monkeypatch.setattr(lifecycle, "_attach_keyboards", lambda: None)
    monkeypatch.setattr(lifecycle, "_build_external_inputs", lambda *args: {probe.name: {}})
    lifecycle._session, lifecycle._pipeline = session, pipeline
    lifecycle._message_processor = object()  # Enable the normal returned-control-event path.
    try:
        lifecycle.notify_reset()  # No submitted frame yet: the synchronous seed is already fresh.
        assert lifecycle.step().tolist() == [1.0]
        lifecycle.step()  # Submit frame 1, held after graph execution but before publication.
        assert entered[1].wait(5.0)
        lifecycle.notify_reset()
        assert lifecycle.step() is None  # Submit frame 2; latest output still belongs to the old episode.
        assert lifecycle.step() is None  # Frame 3 supersedes pending frame 2.
        release[1].set()
        assert entered[3].wait(5.0)
        assert lifecycle.step() is None  # Reset was consumed, but its result has not been published.
        assert lifecycle.last_control_events.should_reset  # Preserve the operator event from old frame 1.

        lifecycle.notify_reset()  # Another reset while frame 3's rebased result is still in flight.
        assert lifecycle.step() is None
        release[3].set()
        assert entered[5].wait(5.0)
        assert lifecycle.step() is None  # Completed frame 3 predates the second notification.
        assert not lifecycle.last_control_events.should_reset

        release[5].set()
        assert session._async_runner.wait_for_frame(5, timeout_s=5.0) is not None
        assert lifecycle.step() is not None
        assert probe.events[:4] == [(True, ExecutionState.RUNNING)] * 4
        assert not lifecycle.last_control_events.should_reset
    finally:
        for event in release.values():
            event.set()
        if session._async_runner is not None:
            assert session._async_runner.stop(timeout_s=5.0)
        lifecycle._session = None
