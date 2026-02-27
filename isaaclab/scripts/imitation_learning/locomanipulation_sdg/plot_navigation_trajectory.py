# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to visualize navigation datasets.

Loads a navigation dataset and generates plots showing paths, poses and obstacles.

Args:
    dataset: Path to the HDF5 dataset file containing recorded demonstrations.
    output_dir: Directory path where visualization plots will be saved.
    figure_size: Size of the generated figures (width, height).
    demo_filter: If provided, only visualize specific demo(s). Can be a single demo name or comma-separated list.
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt


def main():
    """Main function to process dataset and generate visualizations."""
    # add argparse arguments
    parser = argparse.ArgumentParser(
        description="Visualize navigation dataset from locomanipulation sdg demonstrations."
    )
    parser.add_argument(
        "--input_file", type=str, help="Path to the HDF5 dataset file containing recorded demonstrations."
    )
    parser.add_argument("--output_dir", type=str, help="Directory path where visualization plots will be saved.")
    parser.add_argument(
        "--figure_size",
        type=int,
        nargs=2,
        default=[20, 20],
        help="Size of the generated figures (width, height). Default: [20, 20]",
    )
    parser.add_argument(
        "--demo_filter",
        type=str,
        default=None,
        help="If provided, only visualize specific demo(s). Can be a single demo name or comma-separated list.",
    )

    # parse the arguments
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Dataset file not found: {args.input_file}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    dataset = h5py.File(args.input_file, "r")

    demos = list(dataset["data"].keys())

    # Filter demos if specified
    if args.demo_filter:
        filter_demos = [d.strip() for d in args.demo_filter.split(",")]
        demos = [d for d in demos if d in filter_demos]
        if not demos:
            print(f"Warning: No demos found matching filter '{args.demo_filter}'")
            return

    print(f"Visualizing {len(demos)} demonstrations...")

    for i, demo in enumerate(demos):
        print(f"Processing demo {i + 1}/{len(demos)}: {demo}")

        replay_data = dataset["data"][demo]["locomanipulation_sdg_output_data"]
        path = replay_data["base_path"]
        base_pose = replay_data["base_pose"]
        object_pose = replay_data["object_pose"]
        start_pose = replay_data["start_fixture_pose"]
        end_pose = replay_data["end_fixture_pose"]
        obstacle_poses = replay_data["obstacle_fixture_poses"]

        plt.figure(figsize=args.figure_size)
        plt.plot(path[0, :, 0], path[0, :, 1], "r-", label="Target Path", linewidth=2)
        plt.plot(base_pose[:, 0], base_pose[:, 1], "g--", label="Base Pose", linewidth=2)
        plt.plot(object_pose[:, 0], object_pose[:, 1], "b--", label="Object Pose", linewidth=2)
        plt.plot(obstacle_poses[0, :, 0], obstacle_poses[0, :, 1], "ro", label="Obstacles", markersize=8)

        # Add start and end markers
        plt.plot(start_pose[0, 0], start_pose[0, 1], "gs", label="Start", markersize=12)
        plt.plot(end_pose[0, 0], end_pose[0, 1], "rs", label="End", markersize=12)

        plt.legend(loc="upper right", ncol=1, fontsize=12)
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.title(f"Navigation Visualization - {demo}", fontsize=16)
        plt.xlabel("X Position (m)", fontsize=14)
        plt.ylabel("Y Position (m)", fontsize=14)

        output_path = os.path.join(args.output_dir, f"{demo}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()  # Close the figure to free memory

    dataset.close()
    print(f"Visualization complete! Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
