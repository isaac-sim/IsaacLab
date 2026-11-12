# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline deployment contracts. No robot, SSH, systemd or simulation is used."""

import importlib.util
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from isaaclab_tasks.contrib.microduck_obstacle.mdp.actions import MicroDuckWalkingAction, MicroDuckWalkingActionCfg

HERE = Path(__file__).resolve().parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nav_pad = load_script("nav_pad")
proxy = load_script("padproxy_on_duck")


def test_gamepad_disconnect_and_nonfinite_request_stop(monkeypatch):
    """Neither loss of the forwarding socket nor an invalid manual request leaves motion active."""
    messages = []
    monkeypatch.setattr(proxy, "nav_send", lambda method, params=None: messages.append(method))
    proxy.state.update(req=(0.15, 0, 0), nav_driving=True)
    assert not proxy.from_gamepad(json.dumps({"method": "robot.move", "params": {"vx": float("nan")}}))
    assert messages == ["robot.stop"]
    proxy.state.update(req=(0.15, 0, 0), nav_driving=True)
    client, incoming = socket.socketpair()
    outgoing, upstream = socket.socketpair()
    try:
        thread = threading.Thread(target=proxy.pump, args=(incoming, outgoing, proxy.from_gamepad))
        thread.start()
        client.close()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert messages == ["robot.stop", "robot.stop"]
        assert not proxy.state["nav_driving"] and proxy.state["req"] == (0, 0, 0)
    finally:
        client.close()
        upstream.close()


def test_command_mapping_matches_native_training():
    """Random residuals, slew limits, stops and stale frames use the real training owner."""
    rng = np.random.default_rng(42)
    requested = rng.uniform([0, -0.05, -0.7], [0.35, 0.05, 0.7], (128, 3)).astype(np.float32)
    requested[0] = 0
    actions = rng.uniform(-2, 2, (128, 3)).astype(np.float32)
    previous = rng.uniform([0, -0.02, -0.4], [0.25, 0.02, 0.4], (128, 3)).astype(np.float32)
    ages = np.zeros(128, np.float32)
    ages[1] = 0.26
    delivered = np.ones(128, np.float32)
    delivered[2] = -1
    term = object.__new__(MicroDuckWalkingAction)
    term.cfg = MicroDuckWalkingActionCfg()
    term._env = SimpleNamespace(
        device="cpu",
        scene={
            "head_tof": SimpleNamespace(
                delivered_sample_time=torch.from_numpy(delivered), sample_age=torch.from_numpy(ages)
            )
        },
        command_manager=SimpleNamespace(get_command=lambda _: torch.from_numpy(requested)),
    )
    term._raw_actions = torch.from_numpy(previous.copy())
    term.previous_velocity_commands = torch.zeros_like(term._raw_actions)
    term._velocity_limits = torch.tensor(term.cfg.velocity_limits)
    term._command_step = torch.tensor(term.cfg.acceleration_limits) * 0.02
    term.process_actions(torch.from_numpy(actions))
    actual = np.stack(
        [
            nav_pad.command_from_action(a, r, p, d >= 0, age, 0.02)
            for a, r, p, d, age in zip(actions, requested, previous, delivered, ages)
        ]
    )
    np.testing.assert_allclose(actual, term.velocity_commands.numpy(), atol=1e-7)
    with pytest.raises(ValueError, match="Non-finite"):
        nav_pad.command_from_action(np.array([np.nan, 0, 0]), requested[1], previous[1], True, 0, 0.02)


def test_observation_history_final_command_and_stale_stop(tmp_path):
    """A real CPU ONNX graph exposes observation ordering without inventing an inference API."""
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Gather", ["obs", "indices"], ["action"], axis=1)],
            "echo_requested_velocity",
            [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 1343])],
            [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 3])],
            [numpy_helper.from_array(np.array([37, 38, 39], np.int64), "indices")],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=9,
    )
    path = tmp_path / "policy.onnx"
    onnx.save(model, path)
    nav = nav_pad.Navigator(path)
    nav.on_tof({"mm": [1000] * 63 + [float("nan")], "status": [5] * 64, "rx": 10.0})
    state = {"q": [0] * 15, "dq": list(range(15)), "gyro": [0, 0, 0], "grav": [0, 0, -1], "rx": 10.0}
    args = SimpleNamespace(stick_turn_override=True, override_deadband=0.1, stop_dist=0.2)
    request = np.array([0.15, 0.0, 0.4], np.float32)
    sent, drives, _, _, _, result = nav_pad.tick(args, nav, state, request, {"policy": True})
    np.testing.assert_allclose(result[0], request)
    np.testing.assert_array_equal(nav.prev_applied, sent)
    assert drives and sent[2] == request[2]
    assert nav.tof[0][-1] == nav.tof[1][-1] == 0
    assert len(nav.tof_hist) == 10 and nav.tof_hist[0][2] == 1.0
    np.testing.assert_array_equal(nav_pad.policy_joints(list(range(15))), list(range(9)) + list(range(10, 15)))

    # The additional brake must be represented in the next observation's command history.
    nav.on_tof({"mm": [150] * 64, "status": [5] * 64, "rx": 10.02})
    state["rx"] = 10.02
    sent, _, _, safety, _, _ = nav_pad.tick(args, nav, state, request, {"policy": True})
    assert safety
    np.testing.assert_array_equal(sent[:2], 0)
    np.testing.assert_array_equal(nav.prev_applied, sent)
    # A yaw override must never undo stale-telemetry stopping.
    state["received_at"] = time.monotonic() - 1
    sent, *_ = nav_pad.tick(args, nav, state, request, {"policy": True})
    np.testing.assert_array_equal(sent, 0)
    state.pop("received_at")
    nav.tof_received_at -= 1
    sent, *_ = nav_pad.tick(args, nav, state, request, {"policy": True})
    np.testing.assert_array_equal(sent, 0)
    nav_pad.tick(args, nav, state, -request, {"policy": False})
    np.testing.assert_array_equal(nav.prev_applied, -request)
    with pytest.raises(ValueError, match="15 finite"):
        nav_pad.policy_joints([0] * 14)


@pytest.mark.parametrize("ending", ["stale_request", "nonfinite", "eof"])
def test_robot_proxy_stops_active_navigation(tmp_path, ending):
    """Exercise the real stdin/watchdog loop against a temporary fake robotd socket."""
    path = str(tmp_path / "robotd.sock")
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(path)
        server.listen()
        server.settimeout(5)
        code = f"""
import importlib.util, time
spec = importlib.util.spec_from_file_location('proxy', {str(HERE / "padproxy_on_duck.py")!r})
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
p.ROBOTD = {path!r}
p.serve = lambda: None
p.tap = lambda: None
p.REQUEST_FRESH_S = 0.1 if {ending!r} == 'stale_request' else 10
p.WATCHDOG_S = 0.2
p.state.update(req=(0.15, 0, 0), req_t=time.monotonic())
p.main()
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            conn, _ = server.accept()
            with conn, conn.makefile("r") as reader:
                conn.settimeout(5)
                proc.stdin.write("0.1 0 0\n")
                proc.stdin.flush()
                assert json.loads(reader.readline())["method"] == "robot.move"
                if ending == "nonfinite":
                    proc.stdin.write("nan 0 0\n")
                    proc.stdin.flush()
                elif ending == "eof":
                    proc.stdin.close()
                assert json.loads(reader.readline())["method"] == "robot.stop"
        finally:
            if not proc.stdin.closed:
                proc.stdin.close()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        assert proc.returncode == 0, proc.stderr.read()
