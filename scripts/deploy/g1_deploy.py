# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a G1 LEAPP bundle using Isaac ROS Deploy and the official Unitree ros2_control driver.

Source the Isaac ROS environment before running.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

INPUTS = {
    "state:robot:joint_pos": ("state/joint/position", "/joint_states"),
    "state:robot:joint_vel": ("state/joint/velocity", "/joint_states"),
    "state:robot:root_quat_w": ("state/body/rotation", "/imu_sensor_broadcaster/imu"),
    "state:robot:root_ang_vel_b": ("state/body/angular_velocity", "/imu_sensor_broadcaster/imu"),
    "command:base_velocity": ("command/body/velocity", "/cmd_vel"),
    "state:depth_camera:output:distance_to_image_plane": ("tensor", "/g1_policy/depth_tensor"),
    "command:motion": ("tensor", "/g1_policy/motion"),
}
OUTPUTS = {
    "write:robot:set_joint_position_target_index": "target/joint/position",
    "write:robot:write_joint_stiffness_to_sim_index": "kp",
    "write:robot:write_joint_damping_to_sim_index": "kd",
}


def prepare_pipeline(pipeline: str, directory: Path) -> tuple[Path, dict]:
    """Configure ROS message converters for the LEAPP input and output ports."""
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
                # Isaac ROS converters use element_names; current LEAPP bundles label axes.
                if "axes" in spec:
                    spec["element_names"] = [axis.get("names", []) for axis in spec["axes"]]
                ports[connection] = spec
                if direction == "inputs":
                    spec["kind"] = INPUTS[connection][0]
                    # Each converted value needs its own source even when ROS topics are shared.
                    spec["source"] = spec["name"]
                    if connection == "command:base_velocity":
                        spec["element_names"] = [[], ["lin_vel_x", "lin_vel_y", "ang_vel_z"]]
                else:
                    spec["kind"] = OUTPUTS[connection]
    motion = "command:motion" in ports
    expected_inputs = set(INPUTS) - (
        {"command:base_velocity", "state:depth_camera:output:distance_to_image_plane"} if motion else {"command:motion"}
    )
    if set(ports) != expected_inputs | OUTPUTS.keys():
        raise ValueError(
            "Expected G1 joint/IMU inputs, depth and velocity or a motion command, and three joint outputs"
        )
    names = ports["state:robot:joint_pos"]["element_names"][-1]
    for connection in ("state:robot:joint_pos", "state:robot:joint_vel", *OUTPUTS):
        spec = ports[connection]
        if spec.get("element_names", [[]])[-1] != names or spec["shape"] != [1, len(names)]:
            raise ValueError(f"Inconsistent joint layout: {connection}")
    for connection, shape in zip(list(INPUTS)[2:6], ([1, 4], [1, 3], [1, 3], [1, 38, 64, 1])):
        if connection in ports and ports[connection]["shape"] != shape:
            raise ValueError(f"Unsupported tensor shape: {connection}")
    if motion:
        spec = ports["command:motion"]
        shape = spec["shape"]
        if len(shape) < 2 or shape[0] != 1 or any(type(size) is not int or size <= 0 for size in shape):
            raise ValueError("Motion commands require a fixed shape with batch size one")
        if spec["dtype"] != "float32":
            raise ValueError("Motion commands must use float32")
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
    config["forward_joint_command_controller"]["ros__parameters"]["joint_command_topic"] = "/g1_policy/joint_commands"
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
    from launch.actions import EmitEvent, ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler
    from launch.conditions import IfCondition
    from launch.event_handlers import OnProcessExit
    from launch.events import Shutdown
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    from launch_ros.actions import ComposableNodeContainer, Node
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

    publisher = ExecuteProcess(
        condition=IfCondition("true" if manifest.get("motion") else "false"),
        cmd=[
            sys.executable,
            str(Path(__file__).resolve()),
            "--publish-motion",
            "--pipeline",
            manifest["pipeline"],
            "--motion",
            manifest.get("motion") or "",
            "--imu_topic",
            manifest["imu_topic"],
        ],
    )

    def publisher_exited(event, context):
        if context.is_shutdown:
            return []
        failed_processes.append(event.process_name)
        return [EmitEvent(event=Shutdown(reason="Motion reference publisher exited"))]

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
                RegisterEventHandler(OnProcessExit(target_action=publisher, on_exit=publisher_exited)),
                publisher,
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(str(inference)),
                    launch_arguments={
                        "config_path": manifest["pipeline"],
                        "namespace": "g1_policy",
                        "publish_rate": "50.0",
                        "source_to_topic": manifest["source_to_topic"],
                    }.items(),
                ),
                ComposableNodeContainer(
                    condition=IfCondition("true" if manifest["depth_name"] else "false"),
                    package="rclcpp_components",
                    executable="component_container",
                    name="depth_tensor_container",
                    namespace="",
                    composable_node_descriptions=[
                        ComposableNode(
                            package="isaac_ros_tensor_proc",
                            plugin="nvidia::isaac_ros::dnn_inference::ImageToTensorNode",
                            name="depth_to_tensor",
                            parameters=[{"scale": False, "tensor_name": manifest["depth_name"] or ""}],
                            remappings=[("image", "/g1_policy/depth/image"), ("tensor", "/g1_policy/depth_tensor")],
                        )
                    ],
                ),
            ]
        )
    )
    result = service.run()
    return int(result != 0 or bool(failed_processes))


class MotionSequence:
    """Single-clip reference clock with standing/motion transitions and heading alignment."""

    def __init__(self, path, joint_names, transition_s=1.0):
        import numpy as np
        from scipy.spatial.transform import Rotation

        with np.load(path, allow_pickle=False) as data:
            if data["joint_names"].tolist() != joint_names:
                raise ValueError("Reference joint order must match the LEAPP bundle")
            self.q, self.dq, quat = (
                data[name].astype(np.float32) for name in ("joint_pos", "joint_vel", "root_quat_w")
            )
            self.fps = float(data["fps"])
            self.frames, self.stride = int(data["goal_frames"]), int(data["goal_stride"])
        if (
            self.q.ndim != 2
            or self.q.shape[1] != len(joint_names)
            or self.dq.shape != self.q.shape
            or quat.shape != (len(self.q), 4)
            or not len(self.q)
            or self.fps != 50.0
            or self.frames != 10
            or self.stride != 5
            or transition_s <= 0
            or not all(np.isfinite(x).all() for x in (self.q, self.dq, quat))
            or not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-4)
        ):
            raise ValueError("Invalid 50 Hz reference motion")
        self.rotation = Rotation.from_quat(np.roll(quat, -1, axis=-1))
        self.alignment = Rotation.identity()
        self.transition_s = transition_s
        self.started = self.returned = None
        self.return_frame, self.return_blend = 0, 0.0
        self.played = False

    def start(self, now, quat_xyzw):
        import numpy as np
        from scipy.spatial.transform import Rotation

        if self.started is not None:
            raise ValueError("A motion is already active")
        if self.played:
            raise ValueError("This reference has already played; reposition the robot before restarting deployment")
        rotation = Rotation.from_quat(quat_xyzw).as_matrix()
        reference = self.rotation[0].as_matrix()
        angle = np.arctan2(rotation[1, 0], rotation[0, 0]) - np.arctan2(reference[1, 0], reference[0, 0])
        self.alignment = Rotation.from_euler("z", angle)
        self.started, self.returned = now, None
        self.played = True

    def stop(self, now):
        frame, blend = self._position(now)
        if self.started is not None and self.returned is None:
            self.return_frame, self.return_blend, self.returned = frame, blend, now

    def _position(self, now):
        if self.started is None:
            return self.return_frame, 0.0
        if self.returned is not None:
            weight = max(0.0, 1 - (now - self.returned) / self.transition_s)
            if weight == 0:
                self.started = None
            return self.return_frame, self.return_blend * weight
        elapsed = max(0.0, now - self.started)
        frame = max(0, int((elapsed - self.transition_s) * self.fps))
        if frame >= len(self.q):
            self.return_frame, self.return_blend = len(self.q) - 1, 1.0
            self.returned = self.started + self.transition_s + len(self.q) / self.fps
            return self._position(now)
        return frame, min(1.0, elapsed / self.transition_s)

    def sample(self, now):
        import numpy as np

        frame, blend = self._position(now)
        indices = np.minimum(frame + np.arange(self.frames) * self.stride, len(self.q) - 1)
        quat = np.roll((self.alignment * self.rotation[indices]).as_quat(), 1, axis=-1)
        return np.concatenate(
            (self.q[indices], self.dq[indices], quat, np.full((self.frames, 1), blend)), axis=-1
        ).astype(np.float32)


def publish_motion(args):
    """Publish the bundle's reference tensor; ROS services start motion or return to standing."""
    import numpy as np
    import rclpy
    from isaac_ros_tensor_msgs.msg import TensorList
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu
    from std_srvs.srv import Trigger
    from tensor_msgs.msg import ExperimentalTensor

    with tempfile.TemporaryDirectory() as temporary:
        _, ports = prepare_pipeline(args.pipeline, Path(temporary))
    spec = ports.get("command:motion")
    if spec is None or spec["shape"] != [1, 10, 63]:
        raise ValueError("The reference publisher requires a model-based motion bundle")
    sequence = MotionSequence(args.motion, ports["state:robot:joint_pos"]["element_names"][-1])
    rclpy.init()
    node = rclpy.create_node("motion_reference", namespace="g1_policy")
    publisher = node.create_publisher(TensorList, "/g1_policy/motion", 1)
    imu = [None]

    def receive_imu(msg):
        q = msg.orientation
        value = np.asarray([q.x, q.y, q.z, q.w])
        captured = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        now = node.get_clock().now().nanoseconds * 1e-9
        if (
            captured > 0
            and -0.05 <= now - captured <= 0.2
            and (imu[0] is None or captured >= imu[0][1])
            and np.isfinite(value).all()
            and np.isclose(np.linalg.norm(value), 1, atol=1e-3)
        ):
            imu[0] = (value, captured)

    def start(request, response):
        now = node.get_clock().now().nanoseconds * 1e-9
        try:
            if imu[0] is None or not -0.05 <= now - imu[0][1] <= 0.2:
                raise ValueError("A recent pelvis IMU measurement is required")
            sequence.start(now, imu[0][0])
            response.success = True
            response.message = "Playing one motion, then returning to the stationary reference"
        except ValueError as error:
            response.message = str(error)
        return response

    def stand(request, response):
        sequence.stop(node.get_clock().now().nanoseconds * 1e-9)
        response.success, response.message = True, "Returning to standing"
        return response

    def tick():
        stamp = node.get_clock().now()
        value = sequence.sample(stamp.nanoseconds * 1e-9)
        tensor = ExperimentalTensor(
            dtype_code=2, dtype_bits=32, dtype_lanes=1, shape=list(value.shape), data=value.tobytes()
        )
        message = TensorList(names=[spec["name"]], tensors=[tensor])
        message.header.stamp = stamp.to_msg()
        publisher.publish(message)

    node.create_subscription(Imu, args.imu_topic, receive_imu, qos_profile_sensor_data)
    node.create_service(Trigger, "start_motion", start)
    node.create_service(Trigger, "stand", stand)
    node.create_timer(1 / sequence.fps, tick)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pipeline", required=True, help="Path to the LEAPP ONNX pipeline YAML.")
    parser.add_argument("--interface", default="eth0", help="Robot Ethernet interface, passed to the official driver.")
    parser.add_argument(
        "--imu_topic", default="/imu_sensor_broadcaster/imu", help="Absolute pelvis-frame ROS IMU topic."
    )
    parser.add_argument("--motion", help="Exported motion.npz; defaults to the file beside the pipeline YAML.")
    parser.add_argument("--publish-motion", action="store_true", help="Run only the ROS reference publisher.")
    args = parser.parse_args()
    if not args.imu_topic.startswith("/"):
        parser.error("--imu_topic must be an absolute ROS topic name")

    if args.motion is None:
        candidate = Path(args.pipeline).resolve().parent / "motion.npz"
        args.motion = str(candidate) if candidate.is_file() else None
    if args.publish_motion:
        if not args.motion:
            parser.error("--publish-motion requires a reference motion")
        return publish_motion(args)

    with tempfile.TemporaryDirectory(prefix="g1_ros_deploy_") as temporary:
        directory = Path(temporary)
        pipeline, ports = prepare_pipeline(args.pipeline, directory)
        names = ports["state:robot:joint_pos"]["element_names"][-1]
        if "command:motion" in ports:
            if not args.motion:
                parser.error("A motion bundle requires --motion or motion.npz beside the pipeline YAML")
            if ports["command:motion"]["shape"] != [1, 10, 63]:
                parser.error("The reference publisher requires motion shape [1, 10, 63]")
            MotionSequence(args.motion, names)
        else:
            args.motion = None
        urdf, controllers = prepare_hardware(directory, args.interface, names)
        return run_runtime(
            {
                "pipeline": str(pipeline),
                "motion": args.motion,
                "imu_topic": args.imu_topic,
                "urdf": str(urdf),
                "controllers": str(controllers),
                "depth_name": ports.get("state:depth_camera:output:distance_to_image_plane", {}).get("name"),
                "source_to_topic": ",".join(
                    f"{ports[connection]['name']}:{args.imu_topic if kind.startswith('state/body/') else topic}"
                    for connection, (kind, topic) in INPUTS.items()
                    if connection in ports
                ),
            }
        )


if __name__ == "__main__":
    raise SystemExit(main())
