# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Execute the real public entry without native startup to prove failure/cleanup contracts."""

import ast
import contextlib
import logging
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest


@pytest.fixture
def entry_namespace():
    """Load whole entry/cleanup functions, excluding module-level native app startup."""
    script = Path(__file__).resolve().parents[3] / "scripts/environments/teleoperation/teleop_se3_agent.py"
    tree = ast.parse(script.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"main", "_close_teleop_resources"}
    ]
    namespace = {
        "ManagerBasedRLEnvCfg": object,
        "_restore_ovxr_shutdown_handlers": Mock(),
        "sys": sys,
        "logger": logging.getLogger(__name__),
    }
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(script), "exec"), namespace
    )
    return namespace


@pytest.mark.parametrize("stage", ["environment", "device", "haptics", "keyboard", "input"])
def test_live_entry_failure_closes_resources_and_preserves_cause(monkeypatch, entry_namespace, stage):
    """Every startup phase and the live loop must retain its failure after cleanup."""
    import isaaclab_teleop

    failure = RuntimeError(f"{stage} failed")
    camera = SimpleNamespace(close=Mock(), bind=lambda _env: contextlib.nullcontext())
    env = SimpleNamespace(
        sim=SimpleNamespace(device="cpu", has_gui=False, is_running=lambda: True), reset=Mock(), close=Mock()
    )
    interface = MagicMock()
    interface.advance.side_effect = failure
    monkeypatch.setattr(isaaclab_teleop.XrCameraFeedSession, "prepare", Mock(return_value=camera))
    monkeypatch.setattr(
        isaaclab_teleop,
        "create_isaac_teleop_device",
        Mock(side_effect=failure if stage == "device" else None, return_value=interface),
    )
    entry_namespace.update(
        {
            "args_cli": SimpleNamespace(
                task="test-task",
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
            "_rtx_rendering_requested": lambda _args: False,
            "_resolve_cloudxr_env": lambda *_args: None,
            "_make_haptic_io": Mock(side_effect=failure if stage == "haptics" else None, return_value=(Mock(), Mock())),
            "_make_control_keyboard": Mock(side_effect=failure if stage == "keyboard" else None),
            "_install_ovxr_shutdown_handlers": Mock(return_value={}),
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
    with pytest.raises(RuntimeError) as caught:
        entry_namespace["main"](cfg, False, True)
    assert caught.value is failure
    camera.close.assert_called_once_with()
    assert env.close.call_count == int(stage != "environment")


def test_input_cleanup_failure_prevents_borrowed_owner_release(entry_namespace):
    env, camera, xr = Mock(), Mock(), Mock()
    interface = SimpleNamespace(session_cleanup_failed=True)
    with pytest.raises(RuntimeError, match="borrowed XR/environment drain is unproved"):
        entry_namespace["_close_teleop_resources"](env, camera, xr, {}, interface)
    xr.close.assert_not_called()
    camera.close.assert_not_called()
    env.close.assert_not_called()


@pytest.mark.parametrize(
    ("cleanup_stage", "has_primary_error"),
    [("xr", True), ("environment", True), ("camera", False), ("signals", False)],
)
def test_cleanup_preserves_first_error_and_never_frees_undrained_stage(
    entry_namespace, cleanup_stage, has_primary_error
):
    """Cleanup errors must stay visible without replacing the initial cause or risking UAF."""
    events = []
    cleanup_error = RuntimeError("cleanup failed")
    primary_error = RuntimeError("original failure")

    def close(name):
        events.append(name)
        if name == cleanup_stage:
            raise cleanup_error

    xr = SimpleNamespace(close=lambda: close("xr"), async_worker_owns_environment=False)
    camera = SimpleNamespace(close=lambda: close("camera"))
    env = SimpleNamespace(close=lambda: close("environment"))
    entry_namespace["_restore_ovxr_shutdown_handlers"] = lambda _: close("signals")

    def exercise():
        try:
            if has_primary_error:
                raise primary_error
        finally:
            entry_namespace["_close_teleop_resources"](env, camera, xr, {})

    with pytest.raises(RuntimeError) as caught:
        exercise()
    assert caught.value is (primary_error if has_primary_error else cleanup_error)
    assert events == (
        ["xr", "camera", "signals"] if cleanup_stage == "xr" else ["xr", "camera", "environment", "signals"]
    )
    if has_primary_error:
        assert any("cleanup failed" in note for note in primary_error.__notes__)


def test_shutdown_signal_stays_owned_until_xr_drain_finishes(entry_namespace):
    """A second Ctrl-C requests shutdown again instead of interrupting native cleanup."""
    events = []
    original = signal.signal(signal.SIGINT, lambda *_args: events.append("shutdown"))

    def restore(handlers):
        for signum, handler in handlers.items():
            signal.signal(signum, handler)

    def close_xr():
        signal.raise_signal(signal.SIGINT)
        events.append("xr")

    entry_namespace["_restore_ovxr_shutdown_handlers"] = restore
    try:
        entry_namespace["_close_teleop_resources"](
            SimpleNamespace(close=lambda: events.append("environment")),
            SimpleNamespace(close=lambda: events.append("camera")),
            SimpleNamespace(close=close_xr, async_worker_owns_environment=False),
            {signal.SIGINT: original},
        )
        assert events == ["shutdown", "xr", "camera", "environment"]
        assert signal.getsignal(signal.SIGINT) is original
    finally:
        signal.signal(signal.SIGINT, original)


@pytest.mark.parametrize("new_context,drain_failed", [(True, False), (False, False), (True, True)])
def test_partial_construction_cleans_only_new_unowned_context(monkeypatch, entry_namespace, new_context, drain_failed):
    """Failed env construction is cleaned, but preexisting/undrained contexts survive."""
    from isaaclab.app import launch_simulation
    from isaaclab.physics import PhysicsCfg
    from isaaclab.sim import SimulationContext

    previous = object()
    state = {"context": previous}
    clear = Mock(side_effect=lambda: state.update(context=None))
    monkeypatch.setattr(SimulationContext, "instance", lambda: state["context"])
    monkeypatch.setattr(SimulationContext, "clear_instance", clear)
    camera = SimpleNamespace(close=Mock())
    xr = SimpleNamespace(
        close=Mock(side_effect=RuntimeError("drain failed") if drain_failed else None),
        async_worker_owns_environment=False,
    )
    failure = RuntimeError("environment construction failed")
    with pytest.raises(RuntimeError) as caught:
        with launch_simulation(PhysicsCfg(), cleanup_context=False):
            if new_context:
                state["context"] = object()
            try:
                raise failure
            finally:
                entry_namespace["_close_teleop_resources"](None, camera, xr, {}, previous_context=previous)
    assert caught.value is failure
    assert clear.call_count == int(new_context and not drain_failed)
    if not new_context:
        assert state["context"] is previous
