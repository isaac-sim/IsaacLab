#!/usr/bin/env python3
"""
Convert IsaacLab HDF5 demo recordings → LeRobot v3.0 dataset format.

Supports both state-only and visuomotor (with cameras) recordings.

Run with the lerobot conda env:
  conda run -n lerobot python scripts/tools/convert_hdf5_to_lerobot.py \\
      --hdf5 logs/demos/OpenArm_Success/demo.hdf5 \\
      --output ~/datasets/openarm_stack \\
      --task "stack cubes with OpenArm" \\
      --fps 20

With cameras (visuomotor recording):
  conda run -n lerobot python scripts/tools/convert_hdf5_to_lerobot.py \\
      --hdf5 logs/demos/openarm_visuomotor.hdf5 \\
      --output ~/datasets/openarm_visuomotor \\
      --task "stack cubes with OpenArm" \\
      --fps 20 --cameras front_cam wrist_cam

Multiple HDF5 files are merged into one dataset:
  --hdf5 file1.hdf5 file2.hdf5 file3.hdf5

HDF5 layout expected (from record_demos.py):
  data/demo_N/processed_actions  (T, 8)    joint targets + gripper
  data/demo_N/states/articulation/robot/joint_position  (T, 22)
  data/demo_N/obs/front_cam      (T, H, W, 3)  uint8 RGB  [if cameras recorded]
  data/demo_N/obs/wrist_cam      (T, H, W, 3)  uint8 RGB  [if cameras recorded]
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ── helpers ────────────────────────────────────────────────────────────────

def write_parquet(path: Path, table: pa.Table):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


def compute_stats(arr: np.ndarray) -> dict:
    """Per-column statistics in LeRobot format."""
    return {
        "min":   arr.min(axis=0).tolist(),
        "max":   arr.max(axis=0).tolist(),
        "mean":  arr.mean(axis=0).tolist(),
        "std":   arr.std(axis=0).tolist(),
        "count": [int(len(arr))],
        "q01":   np.quantile(arr, 0.01, axis=0).tolist(),
        "q10":   np.quantile(arr, 0.10, axis=0).tolist(),
        "q50":   np.quantile(arr, 0.50, axis=0).tolist(),
        "q90":   np.quantile(arr, 0.90, axis=0).tolist(),
        "q99":   np.quantile(arr, 0.99, axis=0).tolist(),
    }


def compute_image_stats(frames_nhwc: np.ndarray) -> dict:
    """
    Stats for uint8 RGB frames stored in LeRobot's (C,1,1) shape per channel.
    frames_nhwc: (N, H, W, 3) uint8
    """
    f = frames_nhwc.astype(np.float32) / 255.0          # (N, H, W, 3)
    f = f.reshape(-1, 3)                                 # (N*H*W, 3)
    def wrap(v):                                         # scalar per channel → [[[v]]]
        return [[[float(x)]] for x in v]
    return {
        "min":   wrap(f.min(axis=0)),
        "max":   wrap(f.max(axis=0)),
        "mean":  wrap(f.mean(axis=0)),
        "std":   wrap(f.std(axis=0)),
        "count": [int(len(f))],
        "q01":   wrap(np.quantile(f, 0.01, axis=0)),
        "q10":   wrap(np.quantile(f, 0.10, axis=0)),
        "q50":   wrap(np.quantile(f, 0.50, axis=0)),
        "q90":   wrap(np.quantile(f, 0.90, axis=0)),
        "q99":   wrap(np.quantile(f, 0.99, axis=0)),
    }


def encode_video_ffmpeg(frames_nhwc: np.ndarray, output_path: Path, fps: int):
    """
    Encode (N, H, W, 3) uint8 RGB frames to H.264 MP4 via ffmpeg.
    ffmpeg is piped raw bytes on stdin.
    """
    _, H, W, _ = frames_nhwc.shape
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{W}x{H}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",           # high quality
        "-preset", "fast",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None
    proc.stdin.write(frames_nhwc.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {output_path}")


def make_action_names(dim: int) -> list:
    if dim == 8:
        return [f"left_joint_{i+1}" for i in range(7)] + ["gripper"]
    return [f"action_{i}" for i in range(dim)]


def make_state_names(dim: int) -> list:
    if dim == 22:
        return (
            [f"left_joint_{i+1}"        for i in range(7)] +
            [f"right_joint_{i+1}"       for i in range(7)] +
            [f"left_finger_{i+1}"       for i in range(4)] +
            [f"right_finger_{i+1}"      for i in range(4)]
        )
    return [f"joint_{i}" for i in range(dim)]


# ── episode loader ─────────────────────────────────────────────────────────

def load_episodes(hdf5_files: list, cameras: list) -> list:
    """Return list of episode dicts from all HDF5 files."""
    episodes = []
    for hdf5_path in hdf5_files:
        print(f"Reading {hdf5_path} ...")
        with h5py.File(hdf5_path, "r") as f:
            data_grp: h5py.Group = f["data"]  # type: ignore[assignment]
            for ep_name in sorted(data_grp.keys()):
                ep: h5py.Group = data_grp[ep_name]  # type: ignore[assignment]
                actions: np.ndarray = ep["processed_actions"][:]  # type: ignore[index]
                states: np.ndarray = ep["states/articulation/robot/joint_position"][:]  # type: ignore[index]
                T = min(len(actions), len(states))
                if T == 0:
                    print(f"  Skipping {ep_name}: empty")
                    continue

                ep_dict = {
                    "actions": actions[:T].astype(np.float32),
                    "states":  states[:T].astype(np.float32),
                    "success": bool(ep.attrs.get("success", False)),
                }

                # Load camera observations if present and requested
                for cam in cameras:
                    obs_key = f"obs/{cam}"
                    if obs_key in ep.keys():
                        raw: np.ndarray = ep[obs_key][:]  # type: ignore[index]
                        imgs = raw[:, 0] if raw.ndim == 5 else raw   # squeeze env dim
                        ep_dict[f"cam_{cam}"] = imgs[:T]
                    else:
                        print(f"  Warning: {ep_name} has no {obs_key}, skipping camera {cam}")

                episodes.append(ep_dict)
                cam_info = ", ".join(
                    f"{c}:{ep_dict[f'cam_{c}'].shape}"
                    for c in cameras if f"cam_{c}" in ep_dict
                )
                print(f"  {ep_name}: T={T} success={ep_dict['success']}"
                      + (f"  cameras=[{cam_info}]" if cam_info else ""))
    return episodes


# ── main conversion ────────────────────────────────────────────────────────

def hdf5_to_lerobot(
    hdf5_files: list,
    out: str,
    task_description: str,
    fps: int,
    cameras: list,
    chunks_size: int,
):
    out = Path(out)
    if out.exists():
        print(f"Removing existing {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    episodes = load_episodes(hdf5_files, cameras)
    if not episodes:
        raise ValueError("No valid episodes found.")

    # Cameras actually present in the loaded data
    active_cameras = [c for c in cameras if f"cam_{c}" in episodes[0]]
    if cameras and not active_cameras:
        print("Warning: no camera data found — producing state-only dataset")

    action_dim = episodes[0]["actions"].shape[1]
    state_dim  = episodes[0]["states"].shape[1]
    total_episodes = len(episodes)

    # Infer video size from first episode if cameras present
    cam_hw = {}
    for cam in active_cameras:
        h, w = episodes[0][f"cam_{cam}"].shape[1:3]
        cam_hw[cam] = (h, w)

    print(f"\n{total_episodes} episodes | action={action_dim}D | state={state_dim}D"
          + (f" | cameras={list(cam_hw.keys())}" if cam_hw else ""))

    # ── build per-frame rows and episode metadata ──────────────────────────
    all_actions    = []
    all_states     = []
    all_timestamps = []
    all_frame_idx  = []
    all_ep_idx     = []
    all_global_idx = []
    all_task_idx   = []

    ep_meta = []
    global_idx = 0

    # Track global video timestamps per camera per chunk for video encoding
    # cam_chunk_frames[cam][ci] = list of (T, H, W, 3) arrays
    cam_chunk_frames = {cam: {} for cam in active_cameras}
    # cam_ep_timestamps[cam][ep_idx] = (from_ts, to_ts) in the global chunk video
    cam_ep_timestamps = {cam: {} for cam in active_cameras}

    for ep_idx, ep in enumerate(episodes):
        T = len(ep["actions"])
        ci = ep_idx // chunks_size
        ep_from = global_idx

        for t in range(T):
            all_actions.append(ep["actions"][t])
            all_states.append(ep["states"][t])
            all_timestamps.append(float(t) / fps)
            all_frame_idx.append(t)
            all_ep_idx.append(ep_idx)
            all_global_idx.append(global_idx)
            all_task_idx.append(0)
            global_idx += 1

        ep_to = global_idx - 1

        # Track video frame ranges per camera per chunk
        for cam in active_cameras:
            if f"cam_{cam}" not in ep:
                continue
            if ci not in cam_chunk_frames[cam]:
                cam_chunk_frames[cam][ci] = []
            # Record where this episode starts in the chunk video
            existing = cam_chunk_frames[cam][ci]
            chunk_frame_offset = sum(len(f) for f in existing)
            from_ts = chunk_frame_offset / fps
            to_ts   = (chunk_frame_offset + T) / fps
            cam_ep_timestamps[cam][ep_idx] = (from_ts, to_ts)
            cam_chunk_frames[cam][ci].append(ep[f"cam_{cam}"])

        a_stats = compute_stats(ep["actions"])
        s_stats = compute_stats(ep["states"])
        meta = {
            "episode_index": ep_idx,
            "tasks":  [task_description],
            "length": T,
            "data/chunk_index": ci,
            "data/file_index":  0,
            "dataset_from_index": ep_from,
            "dataset_to_index":   ep_to,
            "meta/episodes/chunk_index": ci,
            "meta/episodes/file_index":  0,
            **{f"stats/action/{k}": v            for k, v in a_stats.items()},
            **{f"stats/observation.state/{k}": v  for k, v in s_stats.items()},
        }
        for cam in active_cameras:
            if ep_idx in cam_ep_timestamps.get(cam, {}):
                from_ts, to_ts = cam_ep_timestamps[cam][ep_idx]
                meta[f"videos/observation.images.{cam}/chunk_index"] = ci
                meta[f"videos/observation.images.{cam}/file_index"]  = 0
                meta[f"videos/observation.images.{cam}/from_timestamp"] = from_ts
                meta[f"videos/observation.images.{cam}/to_timestamp"]   = to_ts
        ep_meta.append(meta)

    total_frames = global_idx
    chunk_ids = sorted({m["data/chunk_index"] for m in ep_meta})

    # ── encode videos ──────────────────────────────────────────────────────
    if active_cameras:
        print("\nEncoding videos ...")
        for cam in active_cameras:
            for ci, frame_list in cam_chunk_frames[cam].items():
                all_frames = np.concatenate(frame_list, axis=0)  # (total_T, H, W, 3)
                vid_path = out / f"videos/observation.images.{cam}/chunk-{ci:03d}/file-000.mp4"
                print(f"  {vid_path}  ({len(all_frames)} frames)")
                encode_video_ffmpeg(all_frames, vid_path, fps)

    # ── write data parquet ─────────────────────────────────────────────────
    for ci in chunk_ids:
        ep_set = {m["episode_index"] for m in ep_meta if m["data/chunk_index"] == ci}
        mask = [i for i, e in enumerate(all_ep_idx) if e in ep_set]

        table = pa.table({
            "action": pa.array(
                [all_actions[i].tolist() for i in mask],
                type=pa.list_(pa.float32()),
            ),
            "observation.state": pa.array(
                [all_states[i].tolist() for i in mask],
                type=pa.list_(pa.float32()),
            ),
            "timestamp":     pa.array([all_timestamps[i] for i in mask], pa.float32()),
            "frame_index":   pa.array([all_frame_idx[i]  for i in mask], pa.int64()),
            "episode_index": pa.array([all_ep_idx[i]     for i in mask], pa.int64()),
            "index":         pa.array([all_global_idx[i] for i in mask], pa.int64()),
            "task_index":    pa.array([all_task_idx[i]   for i in mask], pa.int64()),
        })
        path = out / f"data/chunk-{ci:03d}/file-000.parquet"
        write_parquet(path, table)
        print(f"Written {path}  ({len(mask)} frames)")

    # ── write episodes parquet ─────────────────────────────────────────────
    for ci in chunk_ids:
        rows = [m for m in ep_meta if m["meta/episodes/chunk_index"] == ci]

        def col(key, dtype):
            return pa.array([r[key] for r in rows], type=dtype)

        def list_col(key, item_type=pa.float64()):
            return pa.array([r[key] for r in rows], type=pa.list_(item_type))

        cols = {
            "episode_index":             col("episode_index",             pa.int64()),
            "tasks":                     pa.array([r["tasks"] for r in rows], pa.list_(pa.string())),
            "length":                    col("length",                    pa.int64()),
            "data/chunk_index":          col("data/chunk_index",          pa.int64()),
            "data/file_index":           col("data/file_index",           pa.int64()),
            "dataset_from_index":        col("dataset_from_index",        pa.int64()),
            "dataset_to_index":          col("dataset_to_index",          pa.int64()),
            "meta/episodes/chunk_index": col("meta/episodes/chunk_index", pa.int64()),
            "meta/episodes/file_index":  col("meta/episodes/file_index",  pa.int64()),
            "stats/action/min":          list_col("stats/action/min"),
            "stats/action/max":          list_col("stats/action/max"),
            "stats/action/mean":         list_col("stats/action/mean"),
            "stats/action/std":          list_col("stats/action/std"),
            "stats/action/count":        list_col("stats/action/count", pa.int64()),
            "stats/action/q01":          list_col("stats/action/q01"),
            "stats/action/q10":          list_col("stats/action/q10"),
            "stats/action/q50":          list_col("stats/action/q50"),
            "stats/action/q90":          list_col("stats/action/q90"),
            "stats/action/q99":          list_col("stats/action/q99"),
            "stats/observation.state/min":   list_col("stats/observation.state/min"),
            "stats/observation.state/max":   list_col("stats/observation.state/max"),
            "stats/observation.state/mean":  list_col("stats/observation.state/mean"),
            "stats/observation.state/std":   list_col("stats/observation.state/std"),
            "stats/observation.state/count": list_col("stats/observation.state/count", pa.int64()),
            "stats/observation.state/q01":   list_col("stats/observation.state/q01"),
            "stats/observation.state/q10":   list_col("stats/observation.state/q10"),
            "stats/observation.state/q50":   list_col("stats/observation.state/q50"),
            "stats/observation.state/q90":   list_col("stats/observation.state/q90"),
            "stats/observation.state/q99":   list_col("stats/observation.state/q99"),
        }
        # Add per-episode video timing columns
        for cam in active_cameras:
            base = f"videos/observation.images.{cam}"
            cols[f"{base}/chunk_index"]    = col(f"{base}/chunk_index",    pa.int64())
            cols[f"{base}/file_index"]     = col(f"{base}/file_index",     pa.int64())
            cols[f"{base}/from_timestamp"] = col(f"{base}/from_timestamp", pa.float64())
            cols[f"{base}/to_timestamp"]   = col(f"{base}/to_timestamp",   pa.float64())

        path = out / f"meta/episodes/chunk-{ci:03d}/file-000.parquet"
        write_parquet(path, pa.table(cols))
        print(f"Written {path}  ({len(rows)} episodes)")

    # ── write tasks.parquet ────────────────────────────────────────────────
    # LeRobot reads tasks.parquet with pandas and uses `tasks.iloc[i].name`
    # to retrieve the task string. `.name` returns the DataFrame INDEX (not
    # a column), so the index must be the task description string.
    tasks_df = pd.DataFrame({"task_index": [0]}, index=[task_description])
    tasks_df.index.name = None  # match LeRobot's internal convention
    tasks_path = out / "meta/tasks.parquet"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_df.to_parquet(tasks_path)

    # ── build info.json ────────────────────────────────────────────────────
    action_names = make_action_names(action_dim)
    state_names  = make_state_names(state_dim)

    features = {
        "action": {"dtype": "float32", "shape": [action_dim], "names": action_names},
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": state_names},
        "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
        "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
        "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
        "index":         {"dtype": "int64",   "shape": [1], "names": None},
        "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
    }
    for cam in active_cameras:
        h, w = cam_hw[cam]
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": h,
                "video.width":  w,
                "video.codec":  "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps":    fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    info = {
        "codebase_version": "v3.0",
        "robot_type": "openarm",
        "total_episodes": total_episodes,
        "total_frames":   total_frames,
        "total_tasks":    1,
        "chunks_size":    chunks_size,
        "fps":            fps,
        "splits":         {"train": f"0:{total_episodes}"},
        "data_path":      "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path":     "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features":       features,
    }
    info_path = out / "meta/info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # ── write stats.json ───────────────────────────────────────────────────
    all_actions_np = np.concatenate([ep["actions"] for ep in episodes], axis=0)
    all_states_np  = np.concatenate([ep["states"]  for ep in episodes], axis=0)
    stats = {
        "action":            compute_stats(all_actions_np),
        "observation.state": compute_stats(all_states_np),
    }
    for cam in active_cameras:
        all_imgs = np.concatenate(
            [ep[f"cam_{cam}"] for ep in episodes if f"cam_{cam}" in ep], axis=0
        )
        stats[f"observation.images.{cam}"] = compute_image_stats(all_imgs)

    with open(out / "meta/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── summary ───────────────────────────────────────────────────────────
    print(f"\n✓ Dataset written to: {out}")
    print(f"  Episodes : {total_episodes}")
    print(f"  Frames   : {total_frames}")
    print(f"  FPS      : {fps}")
    print(f"  action   : {action_dim}D")
    print(f"  state    : {state_dim}D")
    if active_cameras:
        for cam in active_cameras:
            h, w = cam_hw[cam]
            print(f"  {cam}  : {h}×{w} → videos/observation.images.{cam}/")
    print()
    print("Load in lerobot:")
    print(f"  from lerobot.datasets.lerobot_dataset import LeRobotDataset")
    print(f"  ds = LeRobotDataset(repo_id='local/openarm', root='{out}', download_videos=False)")


# ── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert IsaacLab HDF5 demos → LeRobot v3.0 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--hdf5", nargs="+", required=True,
                        help="HDF5 file(s) to convert (merged into one dataset)")
    parser.add_argument("--output", required=True,
                        help="Output directory")
    parser.add_argument("--task", default="stack cubes with OpenArm",
                        help="Task description string")
    parser.add_argument("--fps", type=int, default=20,
                        help="Control frequency Hz (default 20 = 1/(dt=0.01 × decimation=5))")
    parser.add_argument("--cameras", nargs="*", default=[],
                        metavar="CAM",
                        help="Camera names to extract, e.g. --cameras front_cam wrist_cam. "
                             "Leave empty for state-only dataset.")
    parser.add_argument("--chunks-size", type=int, default=1000,
                        help="Max episodes per chunk (default: 1000)")
    args = parser.parse_args()

    hdf5_to_lerobot(
        hdf5_files=args.hdf5,
        out=args.output,
        task_description=args.task,
        fps=args.fps,
        cameras=args.cameras,
        chunks_size=args.chunks_size,
    )
