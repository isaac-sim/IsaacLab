# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare ROS RealSense depth images for the G1 visual policy.

The official realsense2_camera driver owns capture. This node only applies the policy's
field of view, valid-pixel averaging, range and orientation, publishing 32FC1 metres.
Isaac ROS ImageToTensorNode performs the conversion to the inference tensor.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

DEPTH_SHAPE = (38, 64)
TRAINED_HFOV = 87.0
TRAINED_VFOV = 58.8


def crop_to_fov(width: int, height: int, fx: float, fy: float) -> tuple[int, int, int, int]:
    """Return the centered crop matching the policy's field of view."""
    if min(width, height, fx, fy) <= 0 or not np.isfinite([fx, fy]).all():
        raise ValueError("Camera dimensions and focal lengths must be finite and positive")
    cropped_width = min(width, round(2 * fx * math.tan(math.radians(TRAINED_HFOV) / 2)))
    cropped_height = min(height, round(2 * fy * math.tan(math.radians(TRAINED_VFOV) / 2)))
    return (width - cropped_width) // 2, (height - cropped_height) // 2, cropped_width, cropped_height


def downsample_masked(frame_m: np.ndarray) -> np.ndarray:
    """Area-average valid depths into the policy's 38-by-64 image.

    Args:
        frame_m: Cropped depth image in metres, at least as large as the output.
            Non-finite or non-positive pixels indicate missing returns.

    Returns:
        Float32 metres; bins with no valid samples contain zero. Image orientation
        is unchanged; the publisher rotates after downsampling.
    """
    height, width = frame_m.shape
    if height < DEPTH_SHAPE[0] or width < DEPTH_SHAPE[1]:
        raise ValueError(f"Cropped image {width}x{height} is smaller than the policy's 64x38 input")
    valid = np.isfinite(frame_m) & (frame_m > 0.0)
    values = np.where(valid, frame_m, 0.0).astype(np.float64)
    counts = valid.astype(np.float64)
    y_edges = np.arange(DEPTH_SHAPE[0]) * height // DEPTH_SHAPE[0]
    x_edges = np.arange(DEPTH_SHAPE[1]) * width // DEPTH_SHAPE[1]
    values = np.add.reduceat(np.add.reduceat(values, y_edges, axis=0), x_edges, axis=1)
    counts = np.add.reduceat(np.add.reduceat(counts, y_edges, axis=0), x_edges, axis=1)
    return np.divide(values, counts, out=np.zeros_like(values), where=counts > 0).astype(np.float32)


def prepare_depth(frame_m: np.ndarray, crop: tuple[int, int, int, int], rotation: int) -> np.ndarray:
    """Preserve the policy's millimetre quantization and three-metre missing-return convention."""
    x, y, width, height = crop
    frame = downsample_masked(frame_m[y : y + height, x : x + width])
    frame = np.clip(np.rint(frame * 1000.0), 0, 65535).astype(np.float32) / 1000.0
    frame = np.where(frame > 0.0, np.minimum(frame, 3.0), 3.0)
    return np.ascontiguousarray(frame[::-1, ::-1] if rotation == 180 else frame)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--depth-topic", default="/camera/camera/depth/image_rect_raw")
    parser.add_argument("--camera-info-topic", default="/camera/camera/depth/camera_info")
    parser.add_argument(
        "--depth-rotation",
        type=int,
        choices=(0, 180),
        default=180,
        help="Compensate the exported camera orientation (default: 180 degrees).",
    )
    args = parser.parse_args()

    import rclpy
    from cv_bridge import CvBridge
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    rclpy.init()
    node = rclpy.create_node("g1_depth_preprocessor")
    bridge = CvBridge()
    publisher = node.create_publisher(Image, "/g1_policy/depth/image", 1)
    camera_info = None

    def on_info(msg):
        nonlocal camera_info
        camera_info = msg

    def on_depth(msg):
        if camera_info is None:
            return
        try:
            if (msg.width, msg.height, msg.header.frame_id) != (
                camera_info.width,
                camera_info.height,
                camera_info.header.frame_id,
            ):
                raise ValueError("Depth image and CameraInfo do not match")
            if msg.encoding not in ("16UC1", "32FC1"):
                raise ValueError(f"Unsupported depth encoding: {msg.encoding}")
            raw = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.float32)
            if msg.encoding == "16UC1":
                raw *= 0.001  # The ROS depth-image convention is uint16 millimetres or float32 metres.
            crop = crop_to_fov(msg.width, msg.height, camera_info.k[0], camera_info.k[4])
            output = bridge.cv2_to_imgmsg(prepare_depth(raw, crop, args.depth_rotation), encoding="32FC1")
            output.header = msg.header  # Preserve capture time so downstream freshness checks remain meaningful.
            publisher.publish(output)
        except (ValueError, RuntimeError) as error:
            node.get_logger().error(str(error), throttle_duration_sec=2.0)

    node.create_subscription(CameraInfo, args.camera_info_topic, on_info, qos_profile_sensor_data)
    node.create_subscription(Image, args.depth_topic, on_depth, qos_profile_sensor_data)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
