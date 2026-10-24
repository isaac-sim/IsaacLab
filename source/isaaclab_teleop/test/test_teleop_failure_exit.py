# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Execute the real public entry without native startup to prove failure/cleanup contracts."""

import ast
import contextlib
import logging
import runpy
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
        entry_namespace["main"](cfg, False, True, [])
    assert caught.value is failure
    camera.close.assert_called_once_with()
    assert env.close.call_count == int(stage != "environment")


@pytest.mark.parametrize("device,camera_count", [(None, 0), ("cuda:0", 0), ("cuda:1", 3)])
def test_geometry_publication_cli_reaches_shared_xr_renderer(monkeypatch, device, camera_count):
    """The real parser and entry route defaults/overrides to one stereo/PiP renderer config."""
    import isaaclab_teleop
    from isaaclab_ov.renderers import OVRTXRendererCfg
    from isaaclab_teleop.isaac_teleop_cfg import XrCameraFeedCfg, XrCameraFeedLayoutCfg

    from isaaclab.sensors import CameraCfg

    script = Path(__file__).resolve().parents[3] / "scripts/environments/teleoperation/teleop_se3_agent.py"
    argv = [str(script), "--task", "Test", "--ovxr", "--physics", "ovphysx"]
    if device is not None:
        argv += ["--geometry_publication_device", device]
        argv += ["--ovrtx_active_cuda_gpus", "1"]
    names = [f"camera_{index}" for index in range(camera_count)]
    if names:
        argv += ["--ovxr_pip_head_locked", "--ovxr_pip_camera", *names]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1")
    namespace = runpy.run_path(str(script))["main"].__globals__
    config = SimpleNamespace(
        scene=SimpleNamespace(num_envs=1, **{name: CameraCfg(prim_path=f"/World/{name}") for name in names}),
        sim=SimpleNamespace(device="cpu", physics=object()),
        terminations=SimpleNamespace(time_out=1),
        isaac_teleop=SimpleNamespace(
            xr_camera_feeds=[XrCameraFeedCfg(camera_name=name) for name in names],
            xr_camera_feed_layout=XrCameraFeedLayoutCfg(placement="head_locked"),
        ),
    )
    namespace.update(
        ManagerBasedRLEnvCfg=SimpleNamespace,
        resolve_task_config=lambda *_args, overrides: (config, None),
        validate_launch=lambda *_args: ((), ()),
    )
    monkeypatch.setattr(isaaclab_teleop, "remove_camera_configs", lambda cfg, **_kwargs: cfg)
    camera_session = SimpleNamespace(close=Mock())
    monkeypatch.setattr(isaaclab_teleop.XrCameraFeedSession, "prepare", Mock(return_value=camera_session))
    captured = []
    stopped = RuntimeError("Stop before native startup")

    def prepare(_sim, cfg, renderer_cfg):
        captured.append((cfg, renderer_cfg))
        raise stopped

    monkeypatch.setattr(isaaclab_teleop.OvLibrariesXrSession, "prepare", prepare)
    resolved = namespace["_resolve_env_cfg"]()
    with pytest.raises(RuntimeError) as caught:
        namespace["main"](*resolved)
    assert caught.value is stopped
    assert len(captured) == 1
    session, renderer = captured[0]
    assert renderer.geometry_publication_device == device
    assert renderer.active_cuda_gpus == ("1" if device is not None else None)
    assert renderer.log_file_path == OVRTXRendererCfg().log_file_path
    assert len(session.auxiliary_cameras) == camera_count
    assert all(camera.camera_cfg.renderer_cfg is renderer for camera in session.auxiliary_cameras)
    camera_session.close.assert_called_once_with()


@pytest.mark.parametrize(
    "option,value", [("--geometry_publication_device", "cuda:0"), ("--ovrtx_active_cuda_gpus", "0")]
)
def test_render_device_cli_requires_ovxr(monkeypatch, option, value):
    script = Path(__file__).resolve().parents[3] / "scripts/environments/teleoperation/teleop_se3_agent.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--task", "Test", option, value])
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(script))
    assert caught.value.code == 2


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
