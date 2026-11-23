# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for recording command configuration."""

import math
import sys

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "backend_flags", [["--physics", "isaacsim_physx"], ["--physics", "physx", "--xr"], ["--physics", "newton_mjwarp"]]
)
def test_recording_defers_command_resampling(monkeypatch, tmp_path, backend_flags):
    """Recording defers command resampling beyond a realistic demonstration."""
    from isaaclab_teleop.entrypoints import record_demos

    monkeypatch.setattr(sys, "argv", ["record_demos.py"])
    record_demos._parse_args(["--task", "Isaac-Reach-Franka", *backend_flags, "--teleop_device", "spacemouse"])
    cfg, _, _, _ = record_demos.create_environment_config(str(tmp_path), "reach")

    resampling_time_range = cfg.commands.ee_pose.resampling_time_range
    assert all(math.isfinite(bound) for bound in resampling_time_range)
    assert min(resampling_time_range) > 24 * 60 * 60


@pytest.mark.parametrize(
    ("task", "backend"),
    [
        ("IsaacContrib-PickPlace-GR1T2-Abs", "newton_mjwarp"),
        ("IsaacContrib-PickPlace-GR1T2-Abs", "ovphysx"),
        ("IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs", "newton_mjwarp"),
    ],
)
def test_recording_runs_task_action_policy_on_kitless_physics(monkeypatch, tmp_path, task, backend):
    """Kitless preparation selects the task's action policy, which the recording loop applies."""
    from isaaclab_teleop.entrypoints import record_demos

    monkeypatch.setattr(sys, "argv", ["record_demos.py"])
    record_demos._parse_args(["--task", task, "--physics", backend, "--renderer", "ovrtx"])
    cfg, _, teleop_cfg, _ = record_demos.create_environment_config(str(tmp_path), "policy")
    assert teleop_cfg is cfg.isaac_teleop
    if "GR1T2" in task:
        assert teleop_cfg.require_initial_bimanual_hand_tracking
    else:
        assert teleop_cfg.action_conditioner is not None and teleop_cfg.action_validator is not None


def test_recording_prepares_so101_calibrated_reset(monkeypatch, tmp_path):
    from isaaclab_teleop.entrypoints import record_demos

    monkeypatch.setattr(sys, "argv", ["record_demos.py"])
    record_demos._parse_args(["--task", "IsaacContrib-Stack-Cube-SO101-IK-Abs-v0", "--physics", "isaacsim_physx"])
    cfg, _, _, _ = record_demos.create_environment_config(str(tmp_path), "stack")
    assert cfg.events.randomize_joint_state.params["std"] == 0.0


def test_recording_reset_notifies_teleop_after_environment_reset(monkeypatch):
    """An operator RESET already consumed by the control channel still resets retargeters."""
    from types import SimpleNamespace

    from isaaclab_teleop.entrypoints import record_demos

    calls = []
    monkeypatch.setattr(record_demos, "args_cli", SimpleNamespace(reset_sim_buffer_each_episode=False), raising=False)
    env = SimpleNamespace(
        recorder_manager=SimpleNamespace(reset=lambda: calls.append("recorder-reset")),
        reset=lambda: calls.append("env-reset"),
    )
    device = SimpleNamespace(
        reset=lambda: calls.append("device-reset"), notify_reset=lambda: calls.append("device-notify-reset")
    )

    assert record_demos.handle_reset(env, 3, SimpleNamespace(show_demo=lambda _text: None), "label", device) == 0
    assert calls == ["recorder-reset", "env-reset", "device-reset", "device-notify-reset"]


@pytest.mark.parametrize(
    ("terminated", "truncated", "notified"),
    [
        ((True, False), (False, False), True),
        ((False, False), (False, True), False),
        ((False, False), (True, False), True),
    ],
)
def test_recording_notifies_teleop_when_environment_zero_resets_itself(terminated, truncated, notified):
    """A termination inside ``env.step`` restarts teleop input only for the teleoperated environment."""
    from types import SimpleNamespace

    import torch
    from isaaclab_teleop.entrypoints import record_demos

    calls = []
    device = SimpleNamespace(notify_reset=lambda: calls.append("notify"))
    step_result = (None, None, torch.tensor(terminated), torch.tensor(truncated), {})
    assert record_demos.notify_teleop_of_auto_reset(step_result, device) is notified
    assert calls == (["notify"] if notified else [])
    # Devices without the notification keep their state, as before.
    assert record_demos.notify_teleop_of_auto_reset(step_result, SimpleNamespace()) is False


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--xr", "--ovxr"], "mutually exclusive"),
        (["--ovxr_step_hz", "20"], "requires --ovxr"),
        (["--ovxr", "--ovxr_step_hz", "nan"], "finite and non-negative"),
        (["--ovxr_desktop_mirror"], "--ovxr_desktop_mirror requires --ovxr"),
        (["--ovxr_camera_feed", "head_cam"], "--ovxr_camera_feed requires --ovxr"),
        (["--ovxr", "--ovxr_camera_feed", "head_cam", "presets=rgb"], "takes camera names, not overrides"),
        (["--ovxr", "--ovrtx_active_cuda_gpus", "cuda:0"], "active_cuda_gpus must list CUDA indices"),
        (["--ovxr", "--ovrtx_active_cuda_gpus", "0,0"], "more than once"),
        (["--ovrtx_active_cuda_gpus", "0"], "--ovrtx_active_cuda_gpus requires --ovxr"),
        (["--ovxr", "--ovxr_session_mode", "vr", "--ovxr_step_hz", "20", "--ovrtx_active_cuda_gpus", "1"], None),
        (["--ovxr", "--ovxr_desktop_mirror", "--ovxr_camera_feed", "head_cam"], None),
    ],
)
def test_recording_parses_ovxr_options_like_live(monkeypatch, capsys, arguments, message):
    """The recorder accepts and rejects the live agent's OVXR options with the same results."""
    from isaaclab_teleop.entrypoints import record_demos, teleop_se3_agent

    parsed = []
    for module in (record_demos, teleop_se3_agent):
        monkeypatch.setattr(sys, "argv", ["entrypoint.py"])
        monkeypatch.setattr(module, "args_cli", None, raising=False)
        if message is not None:
            with pytest.raises(SystemExit):
                module._parse_args(["--task", "Test", *arguments])
            assert message in capsys.readouterr().err
            continue
        args = module._parse_args(["--task", "Test", *arguments])
        names = (
            "using_ovxr",
            "ovxr_session_mode",
            "ovxr_step_hz",
            "cloudxr_env",
            "auto_launch_cloudxr",
            "ovxr_desktop_mirror",
            "ovxr_desktop_mirror_fps",
            "ovxr_camera_feed",
            "ovxr_camera_feed_fps",
            "ovrtx_active_cuda_gpus",
        )
        parsed.append({name: getattr(args, name) for name in names})
    if message is None:
        from isaaclab_teleop import CLOUDXR_JS_ENV

        assert parsed[0] == parsed[1]
        assert parsed[0]["using_ovxr"] and parsed[0]["cloudxr_env"] == CLOUDXR_JS_ENV


def test_recording_ovxr_checks_launch_before_preparing_and_owns_context_cleanup(monkeypatch, tmp_path):
    """OVXR checks precede generic validation, the task hook and simulation launch, as in live teleoperation."""
    import contextlib

    from isaaclab_teleop.entrypoints import record_demos

    import isaaclab.app

    monkeypatch.setattr(sys, "argv", ["record_demos.py"])
    monkeypatch.setattr(record_demos, "args_cli", None, raising=False)
    record_demos._parse_args([
        "--task", "IsaacContrib-Stack-Cube-SO101-IK-Abs-v0", "--physics", "ovphysx", "--ovxr",
        "--dataset_file", str(tmp_path / "stack.hdf5"),
    ])  # fmt: skip
    calls = []

    def trace(name, function):
        def traced(*args, **kwargs):
            calls.append(name)
            return function(*args, **kwargs)

        return traced

    resolve = record_demos.resolve_task_config

    def resolve_traced(*args, **kwargs):
        cfg, runner = resolve(*args, **kwargs)
        cfg.isaac_teleop.prepare_environment = trace("prepare_environment", cfg.isaac_teleop.prepare_environment)
        return cfg, runner

    @contextlib.contextmanager
    def launch(_cfg, _args):
        calls.append("launch")
        yield

    monkeypatch.setattr(record_demos, "resolve_task_config", resolve_traced)
    monkeypatch.setattr(record_demos, "prepare_ovxr_launch", trace("ovxr-launch", record_demos.prepare_ovxr_launch))
    monkeypatch.setattr(isaaclab.app, "validate_launch", trace("validate_launch", isaaclab.app.validate_launch))
    reject = trace("ovxr-conflicts", record_demos.reject_ovxr_runtime_conflicts)
    monkeypatch.setattr(record_demos, "reject_ovxr_runtime_conflicts", reject)
    monkeypatch.setattr(record_demos, "launch_simulation", launch)
    monkeypatch.setattr(record_demos, "record_demos", lambda cfg, *_args, **_kwargs: calls.append(("record", cfg)))
    record_demos.main()

    assert calls[:-1] == ["ovxr-launch", "validate_launch", "ovxr-conflicts", "prepare_environment", "launch"]
    cfg = calls[-1][1]
    assert cfg.events.randomize_joint_state.params["std"] == 0.0


@pytest.mark.parametrize(
    "task", ["IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs", "IsaacContrib-PickPlace-GR1T2-Abs"]
)
@pytest.mark.parametrize("pip_camera", [False, True])
def test_recording_ovxr_presentation_options_reach_session_construction(monkeypatch, tmp_path, task, pip_camera):
    """The recorder passes the live agent's PiP, mirror and GPU options to the OVXR session, keeping only PiP cameras.

    The GPU selection reaches the one renderer configuration shared by the stereo session and its PiP camera.
    """
    import contextlib
    from types import SimpleNamespace

    # These adapter tests require the OVRTX runtime and the optional Capture OVXR session module.
    pytest.importorskip("ovrtx")
    pytest.importorskip("isaaccapture.ovxr_session_manager")
    import isaaclab_teleop
    from isaaclab_teleop.entrypoints import record_demos

    from isaaclab.sim import SimulationContext

    monkeypatch.setattr(sys, "argv", ["record_demos.py"])
    monkeypatch.setattr(record_demos, "args_cli", None, raising=False)
    record_demos._parse_args([
        "--task", task, "--physics", "ovphysx", "--ovxr",
        "--cloudxr_env", "none", "--dataset_file", str(tmp_path / "demo.hdf5"),
        "--ovxr_camera_feed_fps", "15", "--ovxr_desktop_mirror", "--ovxr_desktop_mirror_fps", "20",
        "--ovrtx_active_cuda_gpus", "1",
        *(["--ovxr_camera_feed", "robot_pov_cam"] if pip_camera else []),
    ])  # fmt: skip
    launched = []
    prepared = []

    class SessionReached(RuntimeError):
        pass

    @contextlib.contextmanager
    def launch(cfg, _args):
        launched.append(cfg)
        yield

    def prepare(_sim_cfg, xr_cfg, renderer_cfg=None):
        prepared.append((xr_cfg, renderer_cfg))
        raise SessionReached

    monkeypatch.setattr(record_demos, "launch_simulation", launch)
    monkeypatch.setattr(record_demos, "has_kit", lambda: False)
    monkeypatch.setattr(
        isaaclab_teleop.XrCameraFeedSession, "prepare", lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None)
    )
    monkeypatch.setattr(isaaclab_teleop.OvLibrariesXrSession, "prepare", prepare)
    monkeypatch.setattr(SimulationContext, "instance", lambda: None)
    with pytest.raises(SessionReached):
        record_demos.main()

    [cfg] = launched
    assert (getattr(cfg.scene, "robot_pov_cam", None) is not None) is pip_camera
    [(xr_cfg, renderer_cfg)] = prepared
    assert xr_cfg.desktop_mirror is not None and xr_cfg.desktop_mirror.max_fps == 20.0
    assert renderer_cfg.active_cuda_gpus == "1"
    assert [(camera.name, camera.max_fps) for camera in xr_cfg.camera_feeds] == (
        [("robot_pov_cam", 15.0)] if pip_camera else []
    )
    # G1 declares a head-locked layout and GR1T2 keeps the default one; OVXR shows both head-locked.
    assert all(
        camera.space == "head_locked" and camera.position == pytest.approx((0.0, -0.15, -0.8))
        for camera in xr_cfg.camera_feeds
    )
    assert all(camera.camera_cfg.renderer_cfg == renderer_cfg for camera in xr_cfg.camera_feeds)


def test_instruction_display_logs_new_messages_without_kit_labels(caplog):
    """Without a Kit window, instructions are logged once per change instead of dropped."""
    import logging
    from types import SimpleNamespace

    from isaaclab_teleop.entrypoints._instruction_display import InstructionDisplay

    display = InstructionDisplay(xr=False)
    with caplog.at_level(logging.INFO, logger="isaaclab_teleop.entrypoints._instruction_display"):
        display.show_subtask("Current objective: stack")
        display.show_subtask("Current objective: stack")
        display.show_demo("Recorded 1 successful demonstrations.")
        assert [record.getMessage() for record in caplog.records] == [
            "Current objective: stack",
            "Recorded 1 successful demonstrations.",
        ]
        caplog.clear()
        subtask_label, demo_label = SimpleNamespace(text=""), SimpleNamespace(text="")
        display.set_labels(subtask_label, demo_label)
        display.show_subtask("Next objective: place")
        assert subtask_label.text == "Next objective: place" and not caplog.records
