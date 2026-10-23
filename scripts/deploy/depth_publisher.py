# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish the G1 chest RealSense camera to the workstation running g1_deploy.py.

Run this standalone script on the G1 computer connected to the camera (PC2). It requires
NumPy and pyrealsense2, without Isaac Lab or any other files from this repository.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import socket
import struct
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pyrealsense2 as rs

logger = logging.getLogger(__name__)

DEPTH_SHAPE = (38, 64)
TRAINED_HFOV = 87.0
TRAINED_VFOV = 58.8
HEADER = struct.Struct("<IIQHH")
"""Magic, sequence, capture time [ns], height, width; followed by uint16 millimetres."""


def crop_to_fov(intrinsics: rs.intrinsics) -> tuple[int, int, int, int]:
    """Return the centered crop matching the policy's field of view.

    Args:
        intrinsics: Native camera dimensions and focal lengths in pixels.

    Returns:
        Crop origin and size ``(x, y, width, height)`` in pixels. Axes narrower than
        the trained field of view retain their full extent.
    """
    width = min(intrinsics.width, round(2 * intrinsics.fx * math.tan(math.radians(TRAINED_HFOV) / 2)))
    height = min(intrinsics.height, round(2 * intrinsics.fy * math.tan(math.radians(TRAINED_VFOV) / 2)))
    return (intrinsics.width - width) // 2, (intrinsics.height - height) // 2, width, height


def downsample_masked(frame_m: np.ndarray) -> np.ndarray:
    """Area-average valid depths into the policy's 38-by-64 image.

    Args:
        frame_m: Cropped depth image in metres, at least as large as the output.
            Non-finite or non-positive pixels indicate missing returns.

    Returns:
        Float32 metres; bins with no valid samples contain zero. Image orientation
        is unchanged because the deployment control loop handles rotation.
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


def pack_frame(sequence: int, frame_m: np.ndarray, capture_time_ns: int) -> bytes:
    """Serialize a depth frame for the deployment receiver.

    Args:
        sequence: Monotonically increasing frame number, wrapped to uint32 on the wire.
        frame_m: Processed depth in metres, with missing returns encoded as zero.
        capture_time_ns: Camera capture timestamp in nanoseconds.

    Returns:
        Header followed by row-major, little-endian uint16 millimetres.
    """
    millimetres = np.clip(np.rint(frame_m * 1000.0), 0, 65535).astype("<u2")
    return HEADER.pack(0x47314450, sequence & 0xFFFFFFFF, capture_time_ns, *frame_m.shape) + millimetres.tobytes()


def start_stream(width: int, height: int, fps: int) -> tuple[rs.pipeline, rs.pipeline_profile, int]:
    """Start streaming, retrying with a camera reset and then a lower frame rate.

    Args:
        width: Native stream width in pixels.
        height: Native stream height in pixels.
        fps: Requested frame rate in Hz. The final attempt uses at most 60 Hz.

    Returns:
        Started pipeline, profile, and actual requested frame rate.

    Raises:
        RuntimeError: If all three attempts fail to deliver a frame.
    """
    import pyrealsense2 as rs

    device = None
    last_error = None
    for attempt in range(3):
        if attempt == 1 and device is not None:
            logger.warning("Resetting the camera before retrying")
            device.hardware_reset()
            time.sleep(5.0)
        if attempt == 2 and fps > 60:
            logger.warning("Retrying the camera at 60 Hz")
            fps = 60
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        try:
            profile = pipe.start(cfg)
            device = profile.get_device()
            pipe.wait_for_frames(timeout_ms=3000)
            return pipe, profile, fps
        except RuntimeError as error:
            last_error = error
            logger.warning(f"Camera attempt {attempt + 1}/3 failed: {error}")
            with contextlib.suppress(RuntimeError):
                pipe.stop()
    raise RuntimeError("The camera did not deliver a frame after three attempts") from last_error


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", required=True, help="Workstation IP reachable from the G1.")
    parser.add_argument("--port", type=int, default=5601, help="Workstation UDP port; matches --depth_port.")
    parser.add_argument("--width", type=int, default=848, help="Native depth width in pixels.")
    parser.add_argument("--height", type=int, default=480, help="Native depth height in pixels.")
    parser.add_argument("--fps", type=int, default=90, help="Native frame rate; default 90 Hz for the 50 Hz policy.")
    args = parser.parse_args()

    import pyrealsense2 as rs

    pipe, profile, fps = start_stream(args.width, args.height, args.fps)
    try:
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        x, y, width, height = crop_to_fov(intrinsics)
        hfov = math.degrees(2 * math.atan2(width, 2 * intrinsics.fx))
        vfov = math.degrees(2 * math.atan2(height, 2 * intrinsics.fy))
        logger.info(f"Camera {intrinsics.width}x{intrinsics.height} at {fps} Hz; crop {width}x{height} at ({x}, {y})")
        logger.info(f"Cropped field of view {hfov:.2f} x {vfov:.2f} deg; publishing 64x38 to {args.host}:{args.port}")
        if abs(hfov - TRAINED_HFOV) > 1.0 or abs(vfov - TRAINED_VFOV) > 1.0:
            logger.warning("Cropped field of view differs from the trained camera by more than one degree")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sequence, sent, report_time = 0, 0, time.monotonic()
            while True:
                depth = pipe.wait_for_frames(timeout_ms=2000).get_depth_frame()
                if not depth:
                    continue
                raw = np.asanyarray(depth.get_data())
                frame = downsample_masked(raw[y : y + height, x : x + width].astype(np.float32) * scale)
                sock.sendto(pack_frame(sequence, frame, int(depth.get_timestamp() * 1e6)), (args.host, args.port))
                sequence += 1
                sent += 1
                now = time.monotonic()
                if now - report_time >= 5.0:
                    logger.info(f"Publishing {sent / (now - report_time):.0f} Hz; {sequence} frames sent")
                    sent, report_time = 0, now
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        pipe.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
