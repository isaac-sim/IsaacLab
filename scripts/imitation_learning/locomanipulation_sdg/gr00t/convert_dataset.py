# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert locomanipulation SDG HDF5 datasets to LeRobot (GNx) format with parquet, videos, and meta."""

import argparse
import glob
import json
import math
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import tqdm
from scipy.spatial.transform import RigidTransform, Rotation


def pose_to_transform(pose: np.ndarray) -> RigidTransform:
    """Convert a 7D pose array (position + quaternion) to a RigidTransform.

    Args:
        pose: Pose array with shape (..., 7). First 3 elements are translation (x, y, z),
            last 4 are quaternion in scalar-first (w, x, y, z) order.

    Returns:
        RigidTransform representing the pose.
    """
    translation = pose[..., :3]
    rotation = Rotation.from_quat(pose[..., 3:], scalar_first=True)
    return RigidTransform.from_components(translation, rotation)


def pose_from_transform(transform: RigidTransform) -> np.ndarray:
    """Convert a RigidTransform to a 7D pose array.

    Args:
        transform: The rigid transform to convert.

    Returns:
        Pose array with shape (..., 7): translation (3) and quaternion (4) in scalar-first order.
    """
    translation, rotation = transform.as_components()
    quat = rotation.as_quat(scalar_first=True)
    return np.concatenate([translation, quat], axis=-1)


def get_total_object_displacement(demo) -> float:
    """Compute the horizontal distance the object moved between start and end of a demo.

    Args:
        demo: HDF5 group or dict for one episode containing "locomanipulation_sdg_output_data"
            with "object_pose" array (N, 7).

    Returns:
        Euclidean distance in the xy-plane between first and last object position (meters).
    """
    object_pose = demo["locomanipulation_sdg_output_data"]["object_pose"]
    start_pose = object_pose[0]
    end_pose = object_pose[-1]
    distance = math.sqrt((start_pose[0] - end_pose[0]) ** 2 + (start_pose[1] - end_pose[1]) ** 2)
    return distance


def compute_relative_pose(target_pose: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
    """Compute the pose of target relative to base.

    Args:
        target_pose: 7D pose (position + quat) of the target in world frame.
        base_pose: 7D pose (position + quat) of the base frame in world frame.

    Returns:
        7D pose of target expressed in base frame.
    """
    base_pose = pose_to_transform(base_pose)
    target_transform = pose_to_transform(target_pose)
    relative_pose = pose_from_transform(base_pose.inv() * target_transform)
    return relative_pose


def create_directory_structure(output_path: str, video_key: str = "observation.images.ego_view") -> None:
    """Create the required directory structure for GNx LeRobot format.

    Creates meta/, data/chunk-000/, and videos/chunk-000/<video_key>/ under output_path.

    Args:
        output_path: Root path for the converted dataset.
        video_key: Key used for the video subfolder (e.g. "observation.images.ego_view").
    """
    base_path = Path(output_path)

    # Create main directories
    (base_path / "meta").mkdir(parents=True, exist_ok=True)

    # Create chunk directories
    # GNx: only support one chunk folder for now
    chunk_name = "chunk-000"
    (base_path / "data" / chunk_name).mkdir(parents=True, exist_ok=True)
    (base_path / "videos" / chunk_name / video_key).mkdir(parents=True, exist_ok=True)

    print(f"Created directory structure at {output_path}")


def extract_video_from_images(images: np.ndarray, output_path: str, fps: float = 20.0) -> None:
    """Convert an image sequence to an MP4 video file.

    Args:
        images: Image array of shape (S, H, W, C) in RGB, uint8 or float in [0, 1].
        output_path: Path for the output MP4 file.
        fps: Frames per second for the output video.
    """
    if len(images.shape) != 4:  # Expected: [S, H, W, C]
        raise ValueError(f"Expected 4D image array [S, H, W, C], got shape {images.shape}")

    S, H, W, C = images.shape

    # Ensure images are uint8
    if images.dtype != np.uint8:
        if images.dtype in [np.float32, np.float64]:
            # Assuming images are in [0, 1] range
            if images.max() <= 1.0:
                images = (images * 255).astype(np.uint8)
            else:
                images = np.clip(images, 0, 255).astype(np.uint8)
        else:
            images = np.clip(images, 0, 255).astype(np.uint8)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for s in tqdm.tqdm(range(S)):
        frame = images[s]
        # Convert RGB to BGR if needed (OpenCV uses BGR)
        if C == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)

    video_writer.release()
    print(f"Created video: {output_path}")


def create_modality_json(state: dict, action: dict) -> dict:
    """Build the modality.json structure for LeRobot format.

    Maps state and action key names to start/end indices in the concatenated state and
    action vectors, and adds fixed video and annotation keys.

    Args:
        state: Dict mapping state key names to arrays of shape (T, dim).
        action: Dict mapping action key names to arrays of shape (T, dim).

    Returns:
        Modality dict with "state", "action", "video", and "annotation" entries.
    """
    modality = {
        "state": {},
        "action": {},
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {
            "human.action.task_description": {"original_key": "annotation.human.action.task_description"},
            "human.validity": {"original_key": "annotation.human.validity"},
        },
    }

    index = 0
    for k, v in state.items():
        modality["state"][k] = {"start": index, "end": index + v.shape[1]}
        index = modality["state"][k]["end"]

    index = 0
    for k, v in action.items():
        modality["action"][k] = {"start": index, "end": index + v.shape[1]}
        index = modality["action"][k]["end"]

    return modality


def create_info_json(
    total_episodes: int,
    total_frames: int,
    state_dim: int,
    action_dim: int,
    fps: float = 20.0,
    image_shape: tuple[int, int, int] = (160, 256, 3),
    video_key: str = "observation.images.ego_view",
) -> dict:
    """Create the info.json file content for LeRobot dataset metadata.

    Args:
        total_episodes: Number of episodes in the dataset.
        total_frames: Total number of frames across all episodes.
        state_dim: Dimension of the concatenated state vector.
        action_dim: Dimension of the concatenated action vector.
        fps: Frames per second for video and timestamps.
        image_shape: (height, width, channels) for video feature.
        video_key: Key used for the video feature (e.g. "observation.images.ego_view").

    Returns:
        Dict suitable for writing to meta/info.json (codebase_version, features, paths, etc.).
    """

    # Build features dictionary
    features = {
        "observation.state": {
            "dtype": "float64",
            "shape": [state_dim],
            "names": [f"state_{i}" for i in range(state_dim)],
        },
        "action": {"dtype": "float64", "shape": [action_dim], "names": [f"action_{i}" for i in range(action_dim)]},
        "timestamp": {"dtype": "float64", "shape": [1]},
        "annotation.human.action.task_description": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "annotation.human.validity": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        video_key: {
            "dtype": "video",
            "shape": list(image_shape),
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": fps,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
    }

    info = {
        "codebase_version": "v2.0",
        "robot_type": "custom",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 2,
        "total_videos": total_episodes,
        "total_chunks": 0,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }

    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_path", type=str)

    args = parser.parse_args()

    output_path = args.output_path

    # Create directory
    create_directory_structure(output_path)
    dataset_paths = glob.glob(os.path.join(args.input_dir, "*.hdf5"))

    total_episodes = 0
    total_frames = 0
    episodes_data = []

    fps = 20.0
    task_description = "Pick up and drop off the object"

    # PROCESS DATASET
    for dataset_path in dataset_paths:
        dataset = h5py.File(dataset_path, "r")

        print(dataset["data"].keys())
        # PROCESS EPISODE
        for demo_name in dataset["data"].keys():
            demo = dataset["data"][demo_name]

            if get_total_object_displacement(demo) < 2.0:
                continue  # skip failed epsidoes

            base_pose = demo["locomanipulation_sdg_output_data"]["base_pose"]

            # Get state/observation
            state = {
                # state
                "left_hand_pose": compute_relative_pose(
                    np.concatenate([demo["obs"]["left_eef_pos"], demo["obs"]["left_eef_quat"]], axis=-1), base_pose
                ),
                "right_hand_pose": compute_relative_pose(
                    np.concatenate([demo["obs"]["right_eef_pos"], demo["obs"]["right_eef_quat"]], axis=-1), base_pose
                ),
                "left_hand_joint_positions": demo["obs"]["hand_joint_state"][:, 0:7],
                "right_hand_joint_positions": demo["obs"]["hand_joint_state"][:, 7:14],
                "object_pose": compute_relative_pose(
                    demo["locomanipulation_sdg_output_data"]["object_pose"], base_pose
                ),
                "goal_pose": compute_relative_pose(
                    demo["locomanipulation_sdg_output_data"]["base_goal_pose"], base_pose
                ),
                "end_fixture_pose": compute_relative_pose(
                    demo["locomanipulation_sdg_output_data"]["end_fixture_pose"], base_pose
                ),
            }

            obs = {"image": demo["obs"]["robot_pov_cam"][:]}

            action = {
                "left_hand_pose": compute_relative_pose(demo["actions"][:, 0:7], base_pose),
                "right_hand_pose": compute_relative_pose(demo["actions"][:, 7:14], base_pose),
                "left_hand_joint_positions": demo["actions"][:, 14:21],
                "right_hand_joint_positions": demo["actions"][:, 21:28],
                "base_velocity": demo["actions"][:, 28:31],
                "base_height": demo["actions"][:, 31:32],
            }

            episode_duration = len(base_pose)

            timestamps = np.arange(episode_duration, dtype=np.float64) / fps
            state_concat = np.concatenate([v for v in state.values()], axis=-1)
            action_concat = np.concatenate([v for v in action.values()], axis=-1)

            # Create parquet data with all required LeRobot fields
            parquet_data = {
                # Core data
                "observation.state": state_concat.tolist(),  # Concatenated state array per modality.json
                "action": action_concat.tolist(),  # Concatenated action array per modality.json
                "timestamp": timestamps.tolist(),  # Timestamp from episode start
                # Annotation system - indices to meta/tasks.jsonl
                "annotation.human.action.task_description": [0]
                * episode_duration,  # Points to task_index 0 (main task)
                "task_index": [0] * episode_duration,  # Main task index (same as above)
                "annotation.human.validity": [1] * episode_duration,  # Points to task_index 1 ("valid")
                # Episode tracking
                "episode_index": [total_episodes] * episode_duration,  # Episode number
                "index": list(range(total_frames, total_frames + episode_duration)),  # GLOBAL obs index
            }

            # Write parquet
            df = pd.DataFrame(parquet_data)
            parquet_path = Path(output_path) / "data" / "chunk-000" / f"episode_{total_episodes:06d}.parquet"
            df.to_parquet(parquet_path, index=False)

            # Write video
            video_path = (
                Path(output_path)
                / "videos"
                / "chunk-000"
                / "observation.images.ego_view"
                / f"episode_{total_episodes:06d}.mp4"
            )
            extract_video_from_images(obs["image"], str(video_path), fps)

            # Add to episodes data
            episodes_data.append(
                {
                    "episode_index": total_episodes,
                    "tasks": [task_description, "valid"],  # Task description and validity
                    "length": episode_duration,
                }
            )

            # Increment
            total_episodes += 1
            total_frames += episode_duration

            print(total_episodes)

    state_dim = sum(v.shape[1] for v in state.values())
    action_dim = sum(v.shape[1] for v in action.values())

    # Write modality
    modality_json = create_modality_json(state, action)

    with open(os.path.join(output_path, "meta", "modality.json"), "w") as f:
        json.dump(modality_json, f, indent=2)

    # Write info
    info_json = create_info_json(
        total_episodes=total_episodes, total_frames=total_frames, state_dim=state_dim, action_dim=action_dim
    )

    with open(os.path.join(output_path, "meta", "info.json"), "w") as f:
        json.dump(info_json, f, indent=2)

    # Write episodes
    with open(os.path.join(output_path, "meta", "episodes.jsonl"), "w") as f:
        for episode in episodes_data:
            f.write(json.dumps(episode) + "\n")

    # Write tasks
    tasks_data = [{"task_index": 0, "task": task_description}, {"task_index": 1, "task": "valid"}]
    with open(os.path.join(output_path, "meta", "tasks.jsonl"), "w") as f:
        for task in tasks_data:
            f.write(json.dumps(task) + "\n")
