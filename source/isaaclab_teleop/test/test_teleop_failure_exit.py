# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Execute the real public entry without native startup to prove failure/cleanup contracts."""

import contextlib
import logging
import signal
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest


@pytest.fixture
def runtime_namespace(monkeypatch):
    """Use the shared cleanup helpers and restore the signal hook replaced by each test."""
    from isaaclab_teleop.entrypoints import _ovxr_runtime

    monkeypatch.setattr(_ovxr_runtime, "restore_ovxr_shutdown_handlers", Mock())
    return vars(_ovxr_runtime)


@pytest.fixture
def entry_namespace(monkeypatch, runtime_namespace):
    """Use the imported entrypoint and restore all globals replaced by each test."""
    from isaaclab_teleop.entrypoints import teleop_se3_agent as agent

    namespace = vars(agent)
    for name in (
        "args_cli",
        "gym",
        "_resolve_cloudxr_env",
        "has_kit",
        "create_teleop_device",
        "torch",
    ):
        monkeypatch.setitem(namespace, name, namespace.get(name))
    return namespace


@pytest.mark.parametrize(
    "stage,use_capture", [("environment", True), ("device", True), ("haptics", True), ("input", True), ("input", False)]
)
def test_live_entry_failure_closes_resources_and_reports_cause(
    monkeypatch, caplog, entry_namespace, stage, use_capture
):
    """Startup and live-loop failures are logged or propagated, and both release the resources."""
    import isaaclab_teleop

    failure = RuntimeError(f"{stage} failed")
    camera = SimpleNamespace(close=Mock(), bind=lambda _env: contextlib.nullcontext())
    env = SimpleNamespace(
        sim=SimpleNamespace(device="cpu", has_gui=False, visualizers=(), is_running=lambda: True),
        scene=SimpleNamespace(env_prim_paths=["/World/envs/env_0"]),
        reset=Mock(),
        close=Mock(),
    )
    env.unwrapped = env
    interface = MagicMock()
    interface.advance.side_effect = failure
    monkeypatch.setattr(isaaclab_teleop.XrCameraFeedSession, "prepare", Mock(return_value=camera))
    monkeypatch.setattr(
        isaaclab_teleop,
        "create_haptic_feedback_driver",
        Mock(side_effect=failure if stage == "haptics" else None, return_value=None),
    )
    entry_namespace.update(
        {
            "args_cli": SimpleNamespace(
                task="test-task",
                device="cpu",
                xr=False,
                using_ovxr=False,
                disable_external_cameras=True,
                auto_launch_cloudxr=False,
                enable_debug_visualization=False,
                cloudxr_env="none",
            ),
            "gym": SimpleNamespace(
                make=Mock(
                    side_effect=failure if stage == "environment" else None, return_value=SimpleNamespace(unwrapped=env)
                )
            ),
            "_resolve_cloudxr_env": lambda *_args: None,
            "has_kit": lambda: False,
            "create_teleop_device": Mock(side_effect=failure if stage == "device" else None, return_value=interface),
            "torch": SimpleNamespace(inference_mode=contextlib.nullcontext),
        }
    )
    cfg = SimpleNamespace(
        isaac_teleop=SimpleNamespace(
            teleoperation_active_default=False,
            require_initial_bimanual_hand_tracking=False,
            action_conditioner=None,
            action_validator=None,
        )
    )
    if not use_capture:
        cfg = SimpleNamespace()  # Native inputs must not read Capture-only action policy.
    if stage != "input":
        # Startup failures propagate after cleanup, so the command exits non-zero.
        with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError) as caught:
            entry_namespace["run_teleoperation"](cfg, getattr(cfg, "isaac_teleop", None), use_isaac_teleop=use_capture)
        assert caught.value is failure
        if stage != "haptics":
            assert f"Failed to create {'environment' if stage == 'environment' else 'teleop device'}" in caplog.text
    else:
        # A live-loop failure keeps the upstream log-and-return exit.
        with caplog.at_level(logging.ERROR):
            entry_namespace["run_teleoperation"](cfg, getattr(cfg, "isaac_teleop", None), use_isaac_teleop=use_capture)
        assert f"{stage} failed" in caplog.text
    camera.close.assert_called_once_with()
    assert env.close.call_count == int(stage != "environment")


@pytest.mark.parametrize("ovxr", [True, False])
def test_input_cleanup_failure_leaves_the_xr_session_open(monkeypatch, entry_namespace, runtime_namespace, ovxr):
    """The undrained session stays open; the environment closes, and its STOP keeps the session's pair alive."""
    from isaaclab_teleop.entrypoints import teleop_se3_agent as agent

    from isaaclab.app import launch_simulation
    from isaaclab.physics import PhysicsCfg
    from isaaclab.sim import SimulationContext

    env, camera, xr = Mock(), Mock(), Mock()
    interface = SimpleNamespace(session_cleanup_failed=True)
    presentation = None
    if ovxr:
        presentation = entry_namespace["OvxrPresentation"](xr)
        presentation.input_device = interface
    previous, current = object(), object()
    state = {"context": previous}
    clear = Mock()
    monkeypatch.setattr(SimulationContext, "instance", lambda: state["context"])
    monkeypatch.setattr(SimulationContext, "clear_instance", clear)

    def run(*_args, **_kwargs):
        state["context"] = current
        runtime_namespace["close_teleop_resources"](env, camera, presentation, {}, interface)

    monkeypatch.setattr(agent, "_resolve_env_cfg", lambda: (None, None, True, ()))
    monkeypatch.setattr(agent, "args_cli", SimpleNamespace(using_ovxr=ovxr), raising=False)
    monkeypatch.setattr(agent, "run_teleoperation", run)
    monkeypatch.setattr(agent, "launch_simulation", lambda _cfg, _args: launch_simulation(PhysicsCfg()))
    with pytest.raises(RuntimeError, match="drain"):
        agent.main()
    if ovxr:
        xr.request_shutdown.assert_called_once_with()
        xr.close.assert_not_called()
    camera.close.assert_not_called()
    env.close.assert_called_once_with()
    clear.assert_called_once_with()


@pytest.mark.parametrize(
    ("cleanup_stage", "has_primary_error"),
    [("xr", True), ("environment", True), ("camera", False), ("camera", True), ("signals", False)],
)
def test_cleanup_preserves_first_error_and_closes_the_environment_last(
    entry_namespace, runtime_namespace, cleanup_stage, has_primary_error
):
    """Cleanup errors stay visible without replacing the initial cause; the environment still closes after XR."""
    events = []
    cleanup_error = RuntimeError("cleanup failed")
    primary_error = RuntimeError("original failure")

    def close(name):
        events.append(name)
        if name == cleanup_stage:
            raise cleanup_error

    xr = SimpleNamespace(close=lambda: close("xr"))
    camera = SimpleNamespace(close=lambda: close("camera"))
    env = SimpleNamespace(close=lambda: close("environment"))
    runtime_namespace["restore_ovxr_shutdown_handlers"] = lambda _: close("signals")

    def exercise():
        try:
            if has_primary_error:
                raise primary_error
        finally:
            runtime_namespace["close_teleop_resources"](env, camera, entry_namespace["OvxrPresentation"](xr), {})

    with pytest.raises(RuntimeError) as caught:
        exercise()
    assert caught.value is (primary_error if has_primary_error else cleanup_error)
    assert events == ["xr", "camera", "environment", "signals"]
    if has_primary_error:
        assert any("cleanup failed" in note for note in primary_error.__notes__)


def test_shutdown_signal_stays_owned_until_xr_drain_finishes(entry_namespace, runtime_namespace):
    """Until the environment is closed, a second Ctrl-C requests shutdown instead of interrupting cleanup."""
    events = []
    original = signal.signal(signal.SIGINT, lambda *_args: events.append("shutdown"))

    def restore(handlers):
        for signum, handler in handlers.items():
            signal.signal(signum, handler)

    def close_xr():
        signal.raise_signal(signal.SIGINT)
        events.append("xr")

    runtime_namespace["restore_ovxr_shutdown_handlers"] = restore

    def close_environment():
        signal.raise_signal(signal.SIGINT)
        events.append("environment")

    try:
        runtime_namespace["close_teleop_resources"](
            SimpleNamespace(close=close_environment),
            SimpleNamespace(close=lambda: events.append("camera")),
            entry_namespace["OvxrPresentation"](SimpleNamespace(close=close_xr)),
            {signal.SIGINT: original},
        )
        assert events == ["shutdown", "xr", "camera", "shutdown", "environment"]
        assert signal.getsignal(signal.SIGINT) is original
    finally:
        signal.signal(signal.SIGINT, original)


@pytest.mark.parametrize("new_context,drain_failed", [(True, False), (False, False), (True, True)])
def test_partial_construction_cleans_only_the_new_context(
    monkeypatch, entry_namespace, runtime_namespace, new_context, drain_failed
):
    """Failed env construction is cleaned after the XR drain, even a failed one; a preexisting context survives."""
    from isaaclab.app import launch_simulation
    from isaaclab.physics import PhysicsCfg
    from isaaclab.sim import SimulationContext

    previous = object()
    state = {"context": previous}
    clear = Mock(side_effect=lambda: state.update(context=None))
    monkeypatch.setattr(SimulationContext, "instance", lambda: state["context"])
    monkeypatch.setattr(SimulationContext, "clear_instance", clear)
    camera = SimpleNamespace(close=Mock())
    xr = SimpleNamespace(close=Mock(side_effect=RuntimeError("drain failed") if drain_failed else None))
    failure = RuntimeError("environment construction failed")
    with pytest.raises(RuntimeError) as caught:
        with launch_simulation(PhysicsCfg()):
            if new_context:
                state["context"] = object()
            try:
                raise failure
            finally:
                runtime_namespace["close_teleop_resources"](
                    None, camera, entry_namespace["OvxrPresentation"](xr), {}, previous_context=previous
                )
    assert caught.value is failure
    assert clear.call_count == int(new_context)
    if not new_context:
        assert state["context"] is previous
