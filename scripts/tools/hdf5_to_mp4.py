# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to convert HDF5 demonstration files to MP4 videos.

This script converts camera frames stored in HDF5 demonstration files to MP4 videos.
It supports multiple camera modalities including RGB, segmentation, and normal maps.
The output videos are saved in the specified directory with appropriate naming.

required arguments:
    --input_file         Path to the input HDF5 file.
    --output_dir         Directory to save the output MP4 files.

optional arguments:
    --input_keys         List of input keys to process from the HDF5 file. (default: ["table_cam", "wrist_cam", "table_cam_segmentation", "table_cam_normals", "table_cam_shaded_segmentation"])
    --video_height       Height of the output video in pixels. (default: 704)
    --video_width        Width of the output video in pixels. (default: 1280)
    --framerate          Frames per second for the output video. (default: 30)
"""

# Standard library imports
import argparse
import h5py
import numpy as np

# Third-party imports
import os

import cv2

# Constants
DEFAULT_VIDEO_HEIGHT = 704
DEFAULT_VIDEO_WIDTH = 1280
DEFAULT_INPUT_KEYS = [
    "table_cam",
    "wrist_cam",
    "table_cam_segmentation",
    "table_cam_normals",
    "table_cam_shaded_segmentation",
    "table_cam_depth",
]
DEFAULT_FRAMERATE = 30
LIGHT_SOURCE = np.array([0.0, 0.0, 1.0])
MIN_DEPTH = 0.0
MAX_DEPTH = 1.5


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert HDF5 demonstration files to MP4 videos.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input HDF5 file containing demonstration data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory path where the output MP4 files will be saved.",
    )

    parser.add_argument(
        "--input_keys",
        type=str,
        nargs="+",
        default=DEFAULT_INPUT_KEYS,
        help="List of input keys to process.",
    )
    parser.add_argument(
        "--video_height",
        type=int,
        default=DEFAULT_VIDEO_HEIGHT,
        help="Height of the output video in pixels.",
    )
    parser.add_argument(
        "--video_width",
        type=int,
        default=DEFAULT_VIDEO_WIDTH,
        help="Width of the output video in pixels.",
    )
    parser.add_argument(
        "--framerate",
        type=int,
        default=DEFAULT_FRAMERATE,
        help="Frames per second for the output video.",
    )

    args = parser.parse_args()

    return args


def write_demo_to_mp4(
    hdf5_file,
    demo_id,
    frames_path,
    input_key,
    output_dir,
    video_height,
    video_width,
    framerate=DEFAULT_FRAMERATE,
):
    """Convert frames from an HDF5 file to an MP4 video.

    Args:
        hdf5_file (str): Path to the HDF5 file containing the frames.
        demo_id (int): ID of the demonstration to convert.
        frames_path (str): Path to the frames data in the HDF5 file.
        input_key (str): Name of the input key to convert.
        output_dir (str): Directory to save the output MP4 file.
        video_height (int): Height of the output video in pixels.
        video_width (int): Width of the output video in pixels.
        framerate (int, optional): Frames per second for the output video. Defaults to 30.
    """
    with h5py.File(hdf5_file, "r") as f:
        # Get frames based on input key type
        if "shaded_segmentation" in input_key:
            temp_key = input_key.replace("shaded_segmentation", "segmentation")
            frames = f[f"data/demo_{demo_id}/obs/{temp_key}"]
        else:
            frames = f[frames_path + "/" + input_key]

        # Setup video writer
        output_path = os.path.join(output_dir, f"demo_{demo_id}_{input_key}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if "depth" in input_key:
            video = cv2.VideoWriter(output_path, fourcc, framerate, (video_width, video_height), isColor=False)
        else:
            video = cv2.VideoWriter(output_path, fourcc, framerate, (video_width, video_height))

        # Process and write frames
        for ix, frame in enumerate(frames):
            # Convert normal maps to uint8 if needed
            if "normals" in input_key:
                frame = (frame * 255.0).astype(np.uint8)

            # Process shaded segmentation frames
            elif "shaded_segmentation" in input_key:
                seg = frame[..., :-1]
                normals_key = input_key.replace("shaded_segmentation", "normals")
                normals = f[f"data/demo_{demo_id}/obs/{normals_key}"][ix]
                shade = 0.5 + (normals * LIGHT_SOURCE[None, None, :]).sum(axis=-1) * 0.5
                shaded_seg = (shade[..., None] * seg).astype(np.uint8)
                frame = np.concatenate((shaded_seg, frame[..., -1:]), axis=-1)

            # Convert RGB to BGR
            if "depth" not in input_key:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame = (frame[..., 0] - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
                frame = np.where(frame < 0.01, 1.0, frame)
                frame = 1.0 - frame
                frame = (frame * 255.0).astype(np.uint8)

            # Resize to video resolution
            frame = cv2.resize(frame, (video_width, video_height), interpolation=cv2.INTER_CUBIC)
            video.write(frame)

        video.release()


def get_num_demos(hdf5_file):
    """Get the number of demonstrations in the HDF5 file.

    Args:
        hdf5_file (str): Path to the HDF5 file.

    Returns:
        int: Number of demonstrations found in the file.
    """
    with h5py.File(hdf5_file, "r") as f:
        return len(f["data"].keys())


def main():
    """Main function to convert all demonstrations to MP4 videos."""
    # Parse command line arguments
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Get number of demonstrations from the file
    num_demos = get_num_demos(args.input_file)
    print(f"Found {num_demos} demonstrations in {args.input_file}")

    # Convert each demonstration
    for i in range(num_demos):
        frames_path = f"data/demo_{str(i)}/obs"
        for input_key in args.input_keys:
            write_demo_to_mp4(
                args.input_file,
                i,
                frames_path,
                input_key,
                args.output_dir,
                args.video_height,
                args.video_width,
                args.framerate,
            )


if __name__ == "__main__":
    main()
