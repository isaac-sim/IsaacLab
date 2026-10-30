# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a G1 LEAPP bundle using Isaac ROS Deploy and the official Unitree ros2_control driver.

Source the Isaac ROS environment before running. No Isaac Lab installation is needed at runtime.
The exported ONNX models and feedback state remain unchanged; only ROS port metadata is adapted.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

INPUTS = {
    "state:robot:joint_pos": ("state/joint/position", "/joint_states"),
    "state:robot:joint_vel": ("state/joint/velocity", "/joint_states"),
    "state:robot:root_quat_w": ("state/body/rotation", "/imu_sensor_broadcaster/imu"),
    "state:robot:root_ang_vel_b": ("state/body/angular_velocity", "/imu_sensor_broadcaster/imu"),
    "command:base_velocity": ("command/body/velocity", "/cmd_vel"),
    "state:depth_camera:output:distance_to_image_plane": ("tensor", "/g1_policy/depth_tensor"),
}
OUTPUTS = {
    "write:robot:set_joint_position_target_index": "target/joint/position",
    "write:robot:write_joint_stiffness_to_sim_index": "kp",
    "write:robot:write_joint_damping_to_sim_index": "kd",
}
MAX_AGE = 0.2
"""Maximum input/output age in seconds; the graph runs at 50 Hz."""


def prepare_pipeline(pipeline: str, directory: Path) -> tuple[Path, dict]:
    """Map LEAPP ports to official converters without changing model weights or graph edges."""
    source = Path(pipeline).expanduser().resolve(strict=True)
    config = yaml.safe_load(source.read_text())
    ports = {}
    for model_name, model in config["models"].items():
        model_path = (source.parent / model["parameters"]["model_path"]).resolve(strict=True)
        if model_path.suffix != ".onnx":
            raise ValueError("Isaac ROS Deploy requires an ONNX LEAPP bundle")
        model["parameters"]["model_path"] = str(model_path)
        for direction in ("inputs", "outputs"):
            for spec in model[direction]:
                if spec["name"] not in config["pipeline"][direction].get(model_name, []):
                    continue
                connection = spec.get("isaaclab_connection")
                if connection in ports or connection not in (INPUTS if direction == "inputs" else OUTPUTS):
                    raise ValueError(f"Unsupported or duplicate external connection: {connection}")
                ports[connection] = spec
                if direction == "inputs":
                    spec["kind"] = INPUTS[connection][0]
                    # Each converted value needs its own source even when ROS topics are shared.
                    spec["source"] = spec["name"]
                    if connection == "command:base_velocity":
                        # The official Twist converter exposes six axes; this policy uses three.
                        spec["element_names"] = [[], ["lin_vel_x", "lin_vel_y", "ang_vel_z"]]
                else:
                    spec["kind"] = OUTPUTS[connection]
    if set(ports) != INPUTS.keys() | OUTPUTS.keys():
        raise ValueError("Expected the G1 depth policy's six inputs and three joint outputs")
    names = ports["state:robot:joint_pos"]["element_names"][-1]
    for connection in ("state:robot:joint_pos", "state:robot:joint_vel", *OUTPUTS):
        spec = ports[connection]
        if spec.get("element_names", [[]])[-1] != names or spec["shape"] != [1, len(names)]:
            raise ValueError(f"Inconsistent joint layout: {connection}")
    for connection, shape in zip(list(INPUTS)[2:], ([1, 4], [1, 3], [1, 3], [1, 38, 64, 1])):
        if ports[connection]["shape"] != shape:
            raise ValueError(f"Unsupported tensor shape: {connection}")
    if len(set(names)) != len(names):
        raise ValueError("Joint names must be unique")
    if len({spec["name"] for spec in ports.values()}) != len(ports):
        raise ValueError("External tensor names must be unique")
    initial = config["pipeline"].get("initial_values")
    if initial:
        config["pipeline"]["initial_values"] = str((source.parent / initial).resolve(strict=True))
    destination = directory / "g1_ros.yaml"
    destination.write_text(yaml.safe_dump(config, sort_keys=False))
    return destination, ports


def prepare_hardware(directory: Path, interface: str, names: list[str]) -> tuple[Path, Path]:
    """Use the official G1 description and controllers with this policy's 29 body joints."""
    import xacro
    from ament_index_python.packages import get_package_share_directory

    description = Path(get_package_share_directory("unitree_g1_description"))
    robot = xacro.process_file(
        str(description / "urdf/g1_real_hardware.urdf.xacro"),
        mappings={"network_interface": interface, "with_hands": "false"},
    ).toxml()
    root = ET.fromstring(robot)
    joints = [joint.attrib["name"] for joint in root.findall("ros2_control/joint")]
    known_joints = {joint.attrib["name"] for joint in root.findall("joint")}
    if len(joints) != 29 or not set(joints).issubset(names) or not set(names).issubset(known_joints):
        raise ValueError("The bundle must match the official G1 body and optional hand joints")
    bringup = Path(get_package_share_directory("unitree_g1_bringup"))
    official = yaml.safe_load((bringup / "config/controller_manager.yaml").read_text())
    controllers = [
        "joint_state_broadcaster",
        "imu_sensor_broadcaster",
        "freeze_controller",
        "safety_controller",
        "forward_joint_command_controller",
    ]
    manager = official["controller_manager"]["ros__parameters"]
    config = {
        "controller_manager": {
            "ros__parameters": {
                "update_rate": 200,
                "enforce_command_limits": False,
                "enable_pal_statistics_publisher": False,
                "enable_pal_statistics": False,
                **{name: manager[name] for name in controllers},
            }
        }
    }
    for name in controllers:
        source_name = "whole_body_forward_joint_command_controller" if name == controllers[-1] else name
        config[name] = official[source_name]
        if "joints" in config[name]["ros__parameters"]:
            config[name]["ros__parameters"]["joints"] = list(joints)
    # Uncontrolled fingers retain the zero state used by the exported articulation.
    extra_joints = [n for n in names if n not in joints]
    if extra_joints:
        config["joint_state_broadcaster"]["ros__parameters"]["extra_joints"] = extra_joints
    config["forward_joint_command_controller"]["ros__parameters"]["joint_command_topic"] = "/g1_policy/checked_commands"
    config["freeze_controller"]["ros__parameters"].update(kp={".*": 0.0}, kd={".*": 3.0})
    config["safety_controller"]["ros__parameters"].update(
        blend_ratio=0.0,
        max_blend_ratio_speed=0.3,
    )
    config["safety_controller"]["ros__parameters"].pop("gravity_compensation_joints", None)
    urdf, parameters = directory / "g1.urdf", directory / "controllers.yaml"
    urdf.write_text(robot)
    parameters.write_text(yaml.safe_dump(config, sort_keys=False))
    return urdf, parameters


def run_runtime(manifest: dict) -> int:
    """Compose official hardware, controllers, image conversion and inference nodes."""
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription, LaunchService
    from launch.actions import EmitEvent, GroupAction, IncludeLaunchDescription, RegisterEventHandler
    from launch.event_handlers import OnProcessExit
    from launch.events import Shutdown
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    from launch_ros.actions import ComposableNodeContainer, Node, SetParameter
    from launch_ros.descriptions import ComposableNode

    description = {"robot_description": Path(manifest["urdf"]).read_text()}
    params = manifest["controllers"]
    manager = Node(package="controller_manager", executable="ros2_control_node", parameters=[description, params])
    load_stop = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["freeze_controller", "--inactive", "--param-file", params],
    )
    active = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "imu_sensor_broadcaster",
            "safety_controller",
            "forward_joint_command_controller",
            "--activate-as-group",
            "--param-file",
            params,
        ],
    )

    def after_stop_loaded(event, context):
        if context.is_shutdown:
            return []
        return [active] if event.returncode == 0 else [EmitEvent(event=Shutdown(reason="Controller setup failed"))]

    inference = Path(get_package_share_directory("isaac_ros_deploy_bringup")) / "launch/inference_graph.launch.py"
    service = LaunchService()
    failed_processes = []

    def record_exit(event, context):
        if event.returncode != 0:
            failed_processes.append(event.process_name)

    service.include_launch_description(
        LaunchDescription(
            [
                RegisterEventHandler(OnProcessExit(on_exit=record_exit)),
                RegisterEventHandler(
                    OnProcessExit(
                        target_action=manager,
                        on_exit=lambda event, context: []
                        if context.is_shutdown
                        else [EmitEvent(event=Shutdown(reason="Controller manager exited"))],
                    )
                ),
                manager,
                Node(package="robot_state_publisher", executable="robot_state_publisher", parameters=[description]),
                RegisterEventHandler(OnProcessExit(target_action=load_stop, on_exit=after_stop_loaded)),
                RegisterEventHandler(
                    OnProcessExit(
                        target_action=active,
                        on_exit=lambda event, context: []
                        if event.returncode == 0 or context.is_shutdown
                        else [EmitEvent(event=Shutdown(reason="Controller activation failed"))],
                    )
                ),
                load_stop,
                GroupAction(
                    [
                        SetParameter(name="enable_topic", value="/g1_policy/enable"),
                        SetParameter(name="enabled_on_start", value=False),
                        SetParameter(name="reset_feedback_on_enable", value=True),
                        IncludeLaunchDescription(
                            PythonLaunchDescriptionSource(str(inference)),
                            launch_arguments={
                                "config_path": manifest["pipeline"],
                                "namespace": "g1_policy",
                                "publish_rate": "50.0",
                                "source_to_topic": manifest["source_to_topic"],
                            }.items(),
                        ),
                    ]
                ),
                ComposableNodeContainer(
                    package="rclcpp_components",
                    executable="component_container",
                    name="depth_tensor_container",
                    namespace="",
                    composable_node_descriptions=[
                        ComposableNode(
                            package="isaac_ros_tensor_proc",
                            plugin="nvidia::isaac_ros::dnn_inference::ImageToTensorNode",
                            name="depth_to_tensor",
                            parameters=[{"scale": False, "tensor_name": manifest["depth_name"]}],
                            remappings=[("image", "/g1_policy/depth/image"), ("tensor", "/g1_policy/depth_tensor")],
                        )
                    ],
                ),
            ]
        )
    )
    result = service.run()
    return int(result != 0 or bool(failed_processes))


class InputWatchdog:
    """Latch faults and request the official damping controller; never send motor commands.

    InputBuilder warns on stale inputs but continues inference. The forward controller also
    retains its last command, so a missing camera or stopped graph needs an explicit stop.
    """

    def __init__(self, names: list[str], imu_topic: str, depth_name: str):
        import rclpy
        from controller_manager_msgs.srv import SwitchController
        from geometry_msgs.msg import Twist
        from isaac_ros_deploy_interfaces.msg import JointCommand
        from isaac_ros_tensor_msgs.msg import TensorList
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Imu, JointState
        from std_msgs.msg import Bool, Empty

        self.node = rclpy.create_node("g1_policy_watchdog")
        self.names, self.seen, self.enabled, self.fault = names, {}, False, None
        self.depth_name = depth_name
        self.started, self.last_output = time.monotonic(), None
        self.pending, self.stopped = None, False
        self._switch_type, self._bool = SwitchController, Bool
        self.switch = self.node.create_client(SwitchController, "/controller_manager/switch_controller")
        self.gate = self.node.create_publisher(Bool, "/g1_policy/enable", 1)
        self.commands = self.node.create_publisher(JointCommand, "/g1_policy/checked_commands", 1)
        self.node.create_subscription(JointState, "/joint_states", self.on_joints, qos_profile_sensor_data)
        self.node.create_subscription(Imu, imu_topic, self.on_imu, qos_profile_sensor_data)
        self.node.create_subscription(Twist, "/cmd_vel", self.on_velocity, 1)
        self.node.create_subscription(TensorList, "/g1_policy/depth_tensor", self.on_depth, 1)
        self.node.create_subscription(JointCommand, "/g1_policy/joint_commands", self.on_output, 1)
        self.node.create_subscription(Empty, "/g1_policy/stop", lambda _: self.stop("Operator stop"), 1)
        self.node.create_timer(0.02, self.tick)

    def receive(self, source: str, values, stamp=None):
        if not np.isfinite(values).all():
            self.stop(f"Non-finite {source}")
            return False
        if stamp is not None:
            age = (self.node.get_clock().now().nanoseconds - stamp.sec * 10**9 - stamp.nanosec) / 1e9
            if not -0.05 <= age <= MAX_AGE:
                if self.enabled:
                    self.stop(f"Stale or future-dated {source}")
                return False
        self.seen[source] = time.monotonic()
        return True

    def on_joints(self, msg):
        if (
            not set(self.names).issubset(msg.name)
            or len(set(msg.name)) != len(msg.name)
            or len(msg.position) != len(msg.name)
            or len(msg.velocity) != len(msg.name)
        ):
            self.stop("Invalid joint state")
            return
        self.receive("joints", [*msg.position, *msg.velocity], msg.header.stamp)

    def on_imu(self, msg):
        q, v = msg.orientation, msg.angular_velocity
        self.receive("imu", [q.x, q.y, q.z, q.w, v.x, v.y, v.z], msg.header.stamp)
        norm = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
        if abs(norm - 1.0) > 0.05 or 1.0 - 2.0 * (q.x * q.x + q.y * q.y) < (0.7 if self.enabled else 0.9):
            self.stop("Robot is not upright or IMU orientation is invalid")

    def on_velocity(self, msg):
        self.receive(
            "velocity", [msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.x, msg.angular.y, msg.angular.z]
        )

    def on_depth(self, msg):
        if list(msg.names) != [self.depth_name] or len(msg.tensors) != 1:
            self.stop("Invalid depth tensor name")
            return
        tensor = msg.tensors[0]
        if (
            list(tensor.shape) != [38, 64, 1]
            or (tensor.dtype_code, tensor.dtype_bits, tensor.dtype_lanes) != (2, 32, 1)
            or tensor.byte_offset != 0
            or len(tensor.data) != 38 * 64 * 4
        ):
            self.stop("Invalid depth tensor layout")
            return
        self.receive("depth", np.frombuffer(bytes(tensor.data), dtype=np.float32), msg.header.stamp)

    def on_output(self, msg):
        if self.fault or not self.enabled:
            return
        if list(msg.names) != self.names or any(len(v) != len(self.names) for v in (msg.position, msg.kp, msg.kd)):
            self.stop("Invalid policy joint command layout")
            return
        if any(len(v) not in (0, len(self.names)) for v in (msg.velocity, msg.effort)) or min(*msg.kp, *msg.kd) < 0:
            self.stop("Invalid policy velocity, effort or gains")
            return
        if not self.receive("output", [*msg.position, *msg.velocity, *msg.effort, *msg.kp, *msg.kd], msg.header.stamp):
            return
        self.tick()
        if not self.fault:
            if self.last_output is None:
                self.node.get_logger().info(
                    "Fresh policy output ready. Operator may now raise safety_controller blend_ratio."
                )
            self.last_output = time.monotonic()
            self.commands.publish(msg)

    def stop(self, reason: str):
        if self.fault is None:
            self.fault = reason
            self.node.get_logger().error(f"{reason}; requesting damping. Restart to re-arm.")
        self.gate.publish(self._bool(data=False))
        if self.pending is None and not self.stopped and self.switch.service_is_ready():
            request = self._switch_type.Request()
            request.activate_controllers = ["freeze_controller"]
            request.deactivate_controllers = ["forward_joint_command_controller", "safety_controller"]
            request.strictness = request.BEST_EFFORT
            request.timeout.sec = 2
            self.pending = self.switch.call_async(request)
        if self.pending is not None and self.pending.done():
            try:
                self.stopped = self.pending.result().ok
            except Exception as error:
                self.node.get_logger().error(f"Controller switch failed: {error}")
            self.pending = None

    def tick(self):
        if self.fault:
            self.stop(self.fault)
            return
        now = time.monotonic()
        ready = all(now - self.seen.get(key, -np.inf) <= MAX_AGE for key in ("joints", "imu", "velocity", "depth"))
        if self.enabled and not ready:
            self.stop("Robot, velocity command or depth input timed out")
        elif self.enabled and now - (self.last_output or self.started) > (MAX_AGE if self.last_output else 60.0):
            self.stop("Inference output timed out")
        elif ready:
            if not self.enabled:
                self.enabled, self.started = True, now
                self.node.get_logger().info("Inputs ready. Inference enabled; safety blend remains at zero.")
            self.gate.publish(self._bool(data=True))
        elif now - self.started > 120.0:
            self.stop("Startup inputs did not become ready")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pipeline", help="Path to the LEAPP ONNX pipeline YAML.")
    parser.add_argument("--interface", default="eth0", help="Robot Ethernet interface, passed to the official driver.")
    parser.add_argument(
        "--imu_topic", default="/imu_sensor_broadcaster/imu", help="Absolute pelvis-frame ROS IMU topic."
    )
    parser.add_argument("--runtime", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.runtime:
        return run_runtime(json.loads(Path(args.runtime).read_text()))
    if not args.pipeline:
        parser.error("--pipeline is required")
    if not args.imu_topic.startswith("/"):
        parser.error("--imu_topic must be an absolute ROS topic name")

    import rclpy
    from rclpy.signals import SignalHandlerOptions

    shutdown_requested = False

    def interrupted(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)

    with tempfile.TemporaryDirectory(prefix="g1_ros_deploy_") as temporary:
        directory = Path(temporary)
        pipeline, ports = prepare_pipeline(args.pipeline, directory)
        names = ports["state:robot:joint_pos"]["element_names"][-1]
        urdf, controllers = prepare_hardware(directory, args.interface, names)
        manifest = directory / "runtime.json"
        manifest.write_text(
            json.dumps(
                {
                    "pipeline": str(pipeline),
                    "urdf": str(urdf),
                    "controllers": str(controllers),
                    "depth_name": ports["state:depth_camera:output:distance_to_image_plane"]["name"],
                    "source_to_topic": ",".join(
                        f"{ports[connection]['name']}:{args.imu_topic if kind.startswith('state/body/') else topic}"
                        for connection, (kind, topic) in INPUTS.items()
                    ),
                }
            )
        )
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        watchdog = InputWatchdog(
            names, args.imu_topic, ports["state:depth_camera:output:distance_to_image_plane"]["name"]
        )
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--runtime", str(manifest)],
            start_new_session=True,
            env={**os.environ, "TMPDIR": temporary},
        )
        try:
            while process.poll() is None and not watchdog.fault and not shutdown_requested:
                rclpy.spin_once(watchdog.node, timeout_sec=0.02)
            if process.poll() is not None:
                raise RuntimeError(f"Isaac ROS runtime exited: {process.returncode}")
        finally:
            watchdog.stop("Deployment shutdown")
            deadline = time.monotonic() + 3.0
            while not watchdog.stopped and time.monotonic() < deadline:
                rclpy.spin_once(watchdog.node, timeout_sec=0.02)
            if not watchdog.stopped:
                watchdog.node.get_logger().error(
                    "Damping switch was not acknowledged; use the robot's independent stop."
                )
            else:
                deadline = time.monotonic() + 1.5
                while time.monotonic() < deadline:
                    rclpy.spin_once(watchdog.node, timeout_sec=0.02)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5.0)
            watchdog.node.destroy_node()
            rclpy.shutdown()
    return int(process.returncode != 0 or watchdog.fault not in ("Operator stop", "Deployment shutdown"))


if __name__ == "__main__":
    raise SystemExit(main())
