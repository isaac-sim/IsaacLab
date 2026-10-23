# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hardware-free checks of G1 operator controls and control timing."""

import contextlib
import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def deploy():
    pytest.importorskip("termios")
    path = Path(__file__).parents[2] / "scripts/deploy/g1_deploy.py"
    spec = importlib.util.spec_from_file_location("g1_deploy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def terminal(monkeypatch):
    pty = pytest.importorskip("pty")
    master, slave = pty.openpty()
    with os.fdopen(slave, "r") as stream:
        monkeypatch.setattr(sys, "stdin", stream)
        try:
            yield master, stream
        finally:
            os.close(master)


def test_teleop_single_keys_heading_and_terminal_restore(deploy, terminal):
    import termios

    master, stream = terminal
    saved = termios.tcgetattr(stream.fileno())
    quat = np.array([0.0, 0.0, 0.0, 1.0], np.float32)
    with pytest.raises(RuntimeError, match="stop"), deploy.Teleop() as teleop:
        teleop.heading = 0.6
        os.write(master, b"w")
        np.testing.assert_allclose(teleop.command(quat), [0.1, 0.0, 0.3], atol=1e-7)
        os.write(master, b" ")
        np.testing.assert_allclose(teleop.command(quat), [0.0, 0.0, 0.3], atol=1e-7)
        os.write(master, b"r")
        np.testing.assert_allclose(teleop.command(quat), [0.0, 0.0, 0.0], atol=1e-7)
        raise RuntimeError("stop")
    restored = termios.tcgetattr(stream.fileno())
    flags = termios.ICANON | termios.ECHO | termios.ISIG
    assert restored[3] & flags == saved[3] & flags


def test_control_wait_discards_missed_periods(deploy, monkeypatch):
    now = [10.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(deploy.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(deploy.time, "sleep", sleep)
    link = deploy.G1Link.__new__(deploy.G1Link)
    link._next = 0.0
    link.wait(0.02)
    for _ in range(3):
        link.wait(0.02)
    np.testing.assert_allclose(slept[-3:], [0.02, 0.02, 0.02], atol=1e-12)
    assert now[0] == pytest.approx(10.06)


def test_bring_up_waits_for_exact_confirmation_and_keeps_publishing(deploy, terminal, monkeypatch):
    master, _ = terminal
    now, writes = [10.0], []
    pending = [(13.01, b"wrong\n"), (13.11, b"go2\n")]

    def sleep(seconds):
        now[0] += seconds
        if pending and now[0] >= pending[0][0]:
            os.write(master, pending.pop(0)[1])

    monkeypatch.setattr(deploy.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(deploy.time, "sleep", sleep)
    monkeypatch.setattr(deploy, "confirm", lambda *args: None)
    link = SimpleNamespace(
        read=lambda: {"joint_pos": np.zeros(29), "gravity_z": -1.0},
        check_abort=lambda: None,
        write=lambda target, kp, kd: writes.append((now[0], target.copy())),
        wait=lambda period: deploy.G1Link.wait(link, period),
        _next=0.0,
    )
    deploy.bring_up(link, np.ones(29), np.ones(29), np.ones(29), ramp_s=3.0)
    assert not pending
    assert now[0] >= 16.11
    np.testing.assert_allclose(np.diff([stamp for stamp, _ in writes[:150]]), 0.02, atol=1e-12)
    np.testing.assert_allclose([target[0] for _, target in writes[:150]], np.linspace(1 / 150, 1, 150))
    np.testing.assert_allclose([target for _, target in writes[150:]], 1.0)
    assert max(np.diff([stamp for stamp, _ in writes])) <= 0.021


@pytest.fixture
def hardware(deploy, monkeypatch):
    state = {
        "joint_pos": np.zeros(29),
        "joint_vel": np.zeros(29),
        "root_quat_xyzw": np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]),
        "ang_vel": np.zeros(3),
        "gravity_z": -1.0,
        "mode_machine": 5,
    }
    damped, targets = [], []
    link = SimpleNamespace(
        read=lambda: state,
        check_abort=lambda: None,
        wait=lambda period: None,
        write=lambda target, kp, kd: targets.append(target.copy()),
        damp=lambda kd: damped.append(kd.copy()),
    )
    policy = SimpleNamespace(default_pose=np.zeros(29), kp=np.ones(29), kd=np.ones(29))
    monkeypatch.setattr(deploy, "GraphPolicy", lambda *args: policy)
    monkeypatch.setattr(deploy, "G1Link", lambda *args: link)
    monkeypatch.setattr(
        deploy,
        "DepthStream",
        lambda *args: SimpleNamespace(
            wait=lambda: None,
            latest=lambda: np.ones((38, 64)),
        ),
    )
    monkeypatch.setattr(deploy, "take_lowcmd", lambda *args: None)
    monkeypatch.setattr(deploy, "confirm", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["g1_deploy.py", "--pipeline", "test.yaml", "--interface", "lo"])
    return policy, damped, targets


@pytest.mark.parametrize("failure", ["OperatorAbort", "KeyboardInterrupt", "RuntimeError", "TimeoutError"])
def test_startup_failures_damp_before_exiting(deploy, hardware, monkeypatch, failure):
    policy, damped, _ = hardware
    errors = {
        "OperatorAbort": deploy.OperatorAbort,
        "KeyboardInterrupt": KeyboardInterrupt,
        "RuntimeError": RuntimeError,
        "TimeoutError": TimeoutError,
    }

    def bring_up(*args, **kwargs):
        raise errors[failure]("startup stopped")

    monkeypatch.setattr(deploy, "bring_up", bring_up)
    with contextlib.suppress(*errors.values()):
        deploy.main()
    assert len(damped) == 1
    np.testing.assert_array_equal(damped[0], policy.kd)


def test_policy_starts_at_measured_heading_and_quit_sends_no_extra_target(deploy, hardware, terminal, monkeypatch):
    policy, damped, targets = hardware
    master, _ = terminal
    commands = []

    def step(q, dq, quat, gyro, command, depth):
        commands.append(command.copy())
        os.write(master, b"x\n")
        return np.ones(29), policy.kp, policy.kd

    policy.step = step
    monkeypatch.setattr(deploy, "bring_up", lambda *args, **kwargs: None)
    assert deploy.main() == 0
    np.testing.assert_allclose(commands, [[0.0, 0.0, 0.0]], atol=1e-7)
    assert len(targets) == len(damped) == 1


def test_graph_uses_named_ports_resets_feedback_and_corrects_imu(deploy, tmp_path):
    pytest.importorskip("leapp")
    import onnx
    import torch
    import yaml
    from onnx import TensorProto, helper, numpy_helper
    from safetensors.torch import save_file

    names = list(reversed(deploy.MOTOR_NAMES)) + [f"finger_{i}" for i in range(14)]
    n = len(names)

    def port(name, shape, connection=None):
        spec = {"name": name, "shape": shape, "dtype": "float32", "type": "tensor"}
        if connection:
            spec["isaaclab_connection"] = connection
        if shape == [1, n]:
            spec["element_names"] = [names]
        return spec

    inputs = [
        port("q", [1, n], "state:robot:joint_pos"),
        port("dq", [1, n], "state:robot:joint_vel"),
        port("quat", [1, 4], "state:robot:root_quat_w"),
        port("gyro", [1, 3], "state:robot:root_ang_vel_b"),
        port("cmd", [1, 3], "command:base_velocity"),
        port("depth", [1, 38, 64, 1], "state:depth_camera:output:distance_to_image_plane"),
        port("carry", [1, n]),
    ]
    outputs = [
        port("targets", [1, n], "write:robot:set_joint_position_target_index"),
        port("kp", [1, n], "write:robot:write_joint_stiffness_to_sim_index"),
        port("kd", [1, n], "write:robot:write_joint_damping_to_sim_index"),
    ]
    # The target exposes both corrected sensor components and the previous target.
    nodes = [
        helper.make_node("Gather", ["gyro", "zero"], ["gyro_x"], axis=1),
        helper.make_node("Gather", ["quat", "three"], ["quat_w"], axis=1),
        helper.make_node("Mul", ["gyro_x", "two_f"], ["gyro_term"]),
        helper.make_node("Mul", ["quat_w", "three_f"], ["quat_term"]),
        helper.make_node("Add", ["gyro_term", "quat_term"], ["sensor_term"]),
        helper.make_node("Add", ["q", "carry"], ["history_term"]),
        helper.make_node("Add", ["history_term", "sensor_term"], ["targets"]),
        helper.make_node("Identity", ["gains"], ["kp"]),
        helper.make_node("Identity", ["gains"], ["kd"]),
    ]
    values = {
        "zero": np.array([0]),
        "three": np.array([3]),
        "two_f": np.array(2, np.float32),
        "three_f": np.array(3, np.float32),
        "gains": np.arange(1, n + 1, dtype=np.float32)[None],
    }
    graph = helper.make_graph(
        nodes,
        "sensor_feedback",
        [helper.make_tensor_value_info(p["name"], TensorProto.FLOAT, p["shape"]) for p in inputs],
        [helper.make_tensor_value_info(p["name"], TensorProto.FLOAT, p["shape"]) for p in outputs],
        [numpy_helper.from_array(value, name) for name, value in values.items()],
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), tmp_path / "model.onnx"
    )
    data = (tmp_path / "model.onnx").read_bytes()
    save_file({"custom_node/carry": torch.zeros(1, n)}, str(tmp_path / "initial.safetensors"))
    description = {
        "models": {
            "custom_node": {
                "inputs": inputs,
                "outputs": outputs,
                "parameters": {
                    "backend": "onnx",
                    "model_path": "model.onnx",
                    "md5sum": hashlib.md5(data).hexdigest(),
                    "sha256sum": hashlib.sha256(data).hexdigest(),
                },
            }
        },
        "pipeline": {
            "inputs": {"custom_node": [p["name"] for p in inputs[:-1]]},
            "outputs": {"custom_node": [p["name"] for p in outputs]},
            "data_flow": {},
            "feedback_flow": {"custom_node/targets": ["custom_node/carry"]},
            "initial_values": "initial.safetensors",
        },
        "system information": {"leapp version": "0.7.2"},
    }
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(yaml.safe_dump(description))
    policy = deploy.GraphPolicy(str(pipeline), 180, imu_tilt=(90.0, 0.0))
    q = np.linspace(0, 0.28, 29, dtype=np.float32)
    sensors = (
        q,
        np.zeros(29),
        np.array([0, np.sqrt(0.5), 0, np.sqrt(0.5)]),
        np.array([0, 0, 1]),
        np.zeros(3),
        np.ones((38, 64)),
    )
    # A level pelvis with this IMU offset has identity orientation and pelvis gyro +X.
    for multiplier in (1, 2):
        target, kp, kd = policy.step(*sensors)
        np.testing.assert_allclose(target, multiplier * (q + 5), atol=2e-6)
        np.testing.assert_array_equal(kp, np.arange(29, 0, -1))
        np.testing.assert_array_equal(kd, kp)
