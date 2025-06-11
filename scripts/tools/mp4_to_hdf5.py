# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to create a new dataset by combining existing HDF5 demonstrations with visually augmented MP4 videos.

This script takes an existing HDF5 dataset containing demonstrations and a directory of MP4 videos
that are visually augmented versions of the original demonstration videos (e.g., with different lighting,
color schemes, or visual effects). It creates a new HDF5 dataset that preserves all the original
demonstration data (actions, robot state, etc.) but replaces the video frames with the augmented versions.

required arguments:
    --input_file         Path to the input HDF5 file containing original demonstrations.
    --output_file        Path to save the new HDF5 file with augmented videos.
    --videos_dir         Directory containing the visually augmented MP4 videos.
"""

# Standard library imports
import argparse
import glob
import h5py
import numpy as np

# Third-party imports
import os

import cv2


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create a new dataset with visually augmented videos.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input HDF5 file containing original demonstrations.",
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        required=True,
        help="Directory containing the visually augmented MP4 videos.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the new HDF5 file with augmented videos.",
    )

    args = parser.parse_args()

    return args


def get_frames_from_mp4(video_path, target_height=None, target_width=None):
    """Extract frames from an MP4 video file.

    Args:
        video_path (str): Path to the MP4 video file.
        target_height (int, optional): Target height for resizing frames. If None, no resizing is done.
        target_width (int, optional): Target width for resizing frames. If None, no resizing is done.

    Returns:
        np.ndarray: Array of frames from the video in RGB format.
    """
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get video properties
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read all frames into a numpy array
    frames = []
    for _ in range(frame_count):
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if target_height is not None and target_width is not None:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)

    # Convert to numpy array
    frames = np.array(frames).astype(np.uint8)

    # Release the video object
    video.release()

    return frames


def process_video_and_demo(f_in, f_out, video_path, orig_demo_id, new_demo_id):
    """Process a single video and create a new demo with augmented video frames.

    Args:
        f_in (h5py.File): Input HDF5 file.
        f_out (h5py.File): Output HDF5 file.
        video_path (str): Path to the augmented video file.
        orig_demo_id (int): ID of the original demo to copy.
        new_demo_id (int): ID for the new demo.
    """
    # Get original demo data
    actions = f_in[f"data/demo_{str(orig_demo_id)}/actions"]
    eef_pos = f_in[f"data/demo_{str(orig_demo_id)}/obs/eef_pos"]
    eef_quat = f_in[f"data/demo_{str(orig_demo_id)}/obs/eef_quat"]
    gripper_pos = f_in[f"data/demo_{str(orig_demo_id)}/obs/gripper_pos"]
    wrist_cam = f_in[f"data/demo_{str(orig_demo_id)}/obs/wrist_cam"]

    # Get original video resolution
    orig_video = f_in[f"data/demo_{str(orig_demo_id)}/obs/table_cam"]
    target_height, target_width = orig_video.shape[1:3]

    # Extract frames from video with original resolution
    frames = get_frames_from_mp4(video_path, target_height, target_width)

    # Create new datasets
    f_out.create_dataset(f"data/demo_{str(new_demo_id)}/actions", data=actions, compression="gzip")
    f_out.create_dataset(f"data/demo_{str(new_demo_id)}/obs/eef_pos", data=eef_pos, compression="gzip")
    f_out.create_dataset(f"data/demo_{str(new_demo_id)}/obs/eef_quat", data=eef_quat, compression="gzip")
    f_out.create_dataset(f"data/demo_{str(new_demo_id)}/obs/gripper_pos", data=gripper_pos, compression="gzip")
    f_out.create_dataset(
        f"data/demo_{str(new_demo_id)}/obs/table_cam", data=frames.astype(np.uint8), compression="gzip"
    )
    f_out.create_dataset(f"data/demo_{str(new_demo_id)}/obs/wrist_cam", data=wrist_cam, compression="gzip")

    # Copy attributes
    f_out[f"data/demo_{str(new_demo_id)}"].attrs["num_samples"] = f_in[f"data/demo_{str(orig_demo_id)}"].attrs[
        "num_samples"
    ]


def main():
    """Main function to create a new dataset with augmented videos."""
    # Parse command line arguments
    args = parse_args()

    # Get list of MP4 videos
    search_path = os.path.join(args.videos_dir, "*.mp4")
    video_paths = glob.glob(search_path)
    video_paths.sort()
    print(f"Found {len(video_paths)} MP4 videos in {args.videos_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with h5py.File(args.input_file, "r") as f_in, h5py.File(args.output_file, "w") as f_out:
        # Copy all data from input to output
        f_in.copy("data", f_out)

        # Get the largest demo ID to start new demos from
        demo_ids = [int(key.split("_")[1]) for key in f_in["data"].keys()]
        next_demo_id = max(demo_ids) + 1  # noqa: SIM113
        print(f"Starting new demos from ID: {next_demo_id}")

        # Process each video and create new demo
        for video_path in video_paths:
            # Extract original demo ID from video filename
            video_filename = os.path.basename(video_path)
            orig_demo_id = int(video_filename.split("_")[1])

            process_video_and_demo(f_in, f_out, video_path, orig_demo_id, next_demo_id)
            next_demo_id += 1

    print(f"Augmented data saved to {args.output_file}")


if __name__ == "__main__":
    main()
