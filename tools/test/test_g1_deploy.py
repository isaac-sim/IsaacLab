# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 LEAPP port mapping and command freshness without ROS or robot hardware."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.kitless


@pytest.fixture
def deploy():
    spec = importlib.util.spec_from_file_location(
        "g1_deploy", Path(__file__).resolve().parents[2] / "scripts/deploy/g1_deploy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(tmp_path):
    names = [f"joint_{i}" for i in range(29)]
    inputs = [
        ("state:robot:joint_pos", [1, 29], names),
        ("state:robot:joint_vel", [1, 29], names),
        ("state:robot:root_quat_w", [1, 4], ["qx", "qy", "qz", "qw"]),
        ("state:robot:root_ang_vel_b", [1, 3], ["x", "y", "z"]),
    ]
    outputs = [
        ("write:robot:set_joint_position_target_index", [1, 29], names),
        ("write:robot:write_joint_stiffness_to_sim_index", [1, 29], names),
        ("write:robot:write_joint_damping_to_sim_index", [1, 29], names),
    ]

    def create(motion):
        commands = (
            [("command:motion", [1, 10, 62], [])]
            if motion
            else [
                ("command:base_velocity", [1, 3], []),
                ("state:depth_camera:output:distance_to_image_plane", [1, 38, 64, 1], []),
            ]
        )
        model = {"parameters": {"model_path": "policy.onnx"}}
        for direction, entries in (("inputs", inputs + commands), ("outputs", outputs)):
            model[direction] = [
                {
                    "name": f"{direction}_{i}",
                    "isaaclab_connection": connection,
                    "shape": shape,
                    "dtype": "float32",
                    "element_names": [labels],
                }
                for i, (connection, shape, labels) in enumerate(entries)
            ]
        config = {
            "models": {"policy": model},
            "pipeline": {
                direction: {"policy": [port["name"] for port in model[direction]]}
                for direction in ("inputs", "outputs")
            },
        }
        # This test covers manifest preparation, before the runtime loads the model.
        (tmp_path / "policy.onnx").touch()
        path = tmp_path / "bundle.yaml"
        path.write_text(yaml.safe_dump(config))
        return path, config

    return create


@pytest.mark.parametrize("motion", [False, True])
def test_prepare_pipeline_preserves_model_ports_and_maps_command_sources(deploy, bundle, tmp_path, motion):
    path, original = bundle(motion)
    output, ports = deploy.prepare_pipeline(str(path), tmp_path)
    config = yaml.safe_load(output.read_text())
    assert config["pipeline"] == original["pipeline"]
    assert config["models"]["policy"]["parameters"]["model_path"] == str(tmp_path / "policy.onnx")
    command = ports["command:motion" if motion else "command:base_velocity"]
    assert (command["kind"], command["source"]) == ("tensor" if motion else "command/body/velocity", command["name"])
    assert ("state:depth_camera:output:distance_to_image_plane" in ports) is not motion
    assert yaml.safe_load(path.read_text()) == original


@pytest.mark.parametrize("field,value", [("shape", [2, 640]), ("shape", [1, -1]), ("dtype", "float64")])
def test_prepare_pipeline_rejects_invalid_motion_tensors(deploy, bundle, tmp_path, field, value):
    path, config = bundle(True)
    config["models"]["policy"]["inputs"][-1][field] = value
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="Motion commands"):
        deploy.prepare_pipeline(str(path), tmp_path)


def test_motion_tensor_requires_named_finite_fresh_data_and_times_out(deploy, monkeypatch):
    now = [10.0]
    monkeypatch.setattr(deploy.time, "monotonic", lambda: now[0])
    faults, gates = [], []
    watchdog = deploy.InputWatchdog.__new__(deploy.InputWatchdog)
    watchdog.node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(now[0] * 1e9))),
        get_logger=lambda: SimpleNamespace(info=lambda _: None),
    )
    watchdog.required = ("joints", "imu", "motion")
    watchdog.seen = {"joints": 10.0, "imu": 10.0}
    watchdog.started, watchdog.last_output = 10.0, 10.0
    watchdog.fault, watchdog.enabled = None, False
    watchdog.stop = faults.append
    watchdog.gate = SimpleNamespace(publish=gates.append)
    watchdog._bool = SimpleNamespace
    msg = SimpleNamespace(
        names=["motion"],
        header=SimpleNamespace(stamp=SimpleNamespace(sec=10, nanosec=0)),
        tensors=[
            SimpleNamespace(
                shape=[2],
                dtype_code=2,
                dtype_bits=32,
                dtype_lanes=1,
                byte_offset=0,
                data=np.zeros(2, dtype=np.float32).tobytes(),
            )
        ],
    )
    watchdog.tick()
    assert not gates
    watchdog.on_tensor(msg, "motion", "motion", [2])
    watchdog.tick()
    assert gates[-1].data and not faults
    msg.names = ["wrong"]
    watchdog.on_tensor(msg, "motion", "motion", [2])
    assert faults.pop() == "Invalid motion tensor name"
    msg.names = ["motion"]
    msg.tensors[0].data = np.array([np.nan, 0], dtype=np.float32).tobytes()
    watchdog.on_tensor(msg, "motion", "motion", [2])
    assert faults.pop() == "Non-finite motion"
    now[0] = 10.3
    watchdog.seen.update(joints=10.3, imu=10.3)
    watchdog.tick()
    assert faults.pop() == "Robot or policy command input timed out"
