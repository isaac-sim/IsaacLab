#!/usr/bin/env python3
"""
Convert IsaacLab HDF5 demo recordings → LeRobot v3.0 dataset format.

Run with the lerobot conda env:
  conda run -n lerobot python -u scripts/tools/convert_hdf5_to_lerobot.py \\
      --hdf5 logs/demos/pickup.hdf5 \\
      --output ~/datasets/openarm \\
      --task "Pick up the red cube." \\
      --fps 30 --cameras front_cam wrist_cam body_cam
"""

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# ── output helpers (always flush, always stdout so conda run -u shows them) ──

def _print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)

def _tqdm(*args, **kwargs):
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(*args, **kwargs)


# ── NVENC detection (done once at startup) ────────────────────────────────────

def _detect_encoder() -> tuple[list[str], str]:
    """Return (ffmpeg encoder args, label). Prefers GPU NVENC, falls back to CPU."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        if "h264_nvenc" in out:
            return (
                ["-vcodec", "h264_nvenc", "-pix_fmt", "yuv420p",
                 "-preset", "p4", "-rc", "constqp", "-qp", "18"],
                "h264_nvenc (GPU)",
            )
    except Exception:
        pass
    return (
        ["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast"],
        "libx264 (CPU)",
    )

_ENCODER_ARGS, _ENCODER_LABEL = _detect_encoder()


# ── video encoding ─────────────────────────────────────────────────────────────

def encode_video_ffmpeg(frames_nhwc: np.ndarray, output_path: Path, fps: int):
    """Encode (N,H,W,3) uint8 RGB → H.264 MP4.  Uses NVENC when available."""
    _, H, W, _ = frames_nhwc.shape
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "pipe:0",
        *_ENCODER_ARGS,
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


# ── statistics ─────────────────────────────────────────────────────────────────

def compute_stats(arr: np.ndarray) -> dict:
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
    """Compute per-channel image stats.  Uses GPU (torch CUDA) when available."""
    def wrap(v):
        return [[[float(x)]] for x in v]

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA")

        # Stream through GPU in chunks — RTX 5090 has 32 GB VRAM so use large
        # chunks to keep the GPU pipeline full (less Python overhead per pixel).
        CHUNK = 10000
        N, H, W, C = frames_nhwc.shape
        device = "cuda"

        mins = torch.full((C,),  1e9, dtype=torch.float32, device=device)
        maxs = torch.full((C,), -1e9, dtype=torch.float32, device=device)
        sums  = torch.zeros(C, dtype=torch.float64, device=device)
        sums2 = torch.zeros(C, dtype=torch.float64, device=device)
        all_pix_cpu = []  # kept on CPU for quantile (avoids 32 GB alloc)

        for i in range(0, N, CHUNK):
            chunk = torch.from_numpy(frames_nhwc[i:i + CHUNK]).to(device, non_blocking=True)
            chunk = chunk.float().div_(255.0).reshape(-1, C)   # (n*H*W, C)
            mins  = torch.minimum(mins,  chunk.min(dim=0).values)
            maxs  = torch.maximum(maxs,  chunk.max(dim=0).values)
            sums  += chunk.sum(dim=0).double()
            sums2 += chunk.pow(2).sum(dim=0).double()
            all_pix_cpu.append(chunk.cpu())

        n    = N * H * W
        mean = (sums / n).float()
        std  = ((sums2 / n) - mean.double().pow(2)).clamp(0).sqrt().float()

        # Quantiles on CPU (concatenated tensor is still large but avoids
        # a full GPU copy of all data at once)
        all_pix = torch.cat(all_pix_cpu, dim=0)  # (N*H*W, C)
        qs = torch.quantile(all_pix, torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99]), dim=0)

        return {
            "min":   wrap(mins.cpu().tolist()),
            "max":   wrap(maxs.cpu().tolist()),
            "mean":  wrap(mean.cpu().tolist()),
            "std":   wrap(std.cpu().tolist()),
            "count": [int(n)],
            "q01":   wrap(qs[0].tolist()),
            "q10":   wrap(qs[1].tolist()),
            "q50":   wrap(qs[2].tolist()),
            "q90":   wrap(qs[3].tolist()),
            "q99":   wrap(qs[4].tolist()),
        }

    except Exception as e:
        _print(f"  [stats] GPU unavailable ({e}), using CPU")
        f = frames_nhwc.astype(np.float32) / 255.0
        f = f.reshape(-1, 3)
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


# ── parallel HDF5 loader ───────────────────────────────────────────────────────

def _load_one_episode(hdf5_path: str, ep_name: str, cameras: list) -> dict | None:
    """Load a single episode — each call opens its own h5py handle (thread-safe)."""
    with h5py.File(hdf5_path, "r") as f:
        ep = f["data"][ep_name]
        actions = ep["processed_actions"][:]
        states  = ep["states/articulation/robot/joint_position"][:]
        T = min(len(actions), len(states))
        if T == 0:
            return None
        ep_dict = {
            "name":    ep_name,
            "actions": actions[:T].astype(np.float32),
            "states":  states[:T].astype(np.float32),
            "success": bool(ep.attrs.get("success", False)),
        }
        for cam in cameras:
            key = f"obs/{cam}"
            if key in ep:
                raw = ep[key][:]
                imgs = raw[:, 0] if raw.ndim == 5 else raw
                ep_dict[f"cam_{cam}"] = imgs[:T]
    return ep_dict


def load_episodes(hdf5_files: list, cameras: list, n_workers: int = 8) -> list:
    """Load all episodes in parallel (one thread per episode, each with own file handle)."""
    jobs = []
    for hdf5_path in hdf5_files:
        _print(f"Scanning {hdf5_path} ...")
        with h5py.File(hdf5_path, "r") as f:
            for ep_name in sorted(f["data"].keys()):
                jobs.append((str(hdf5_path), ep_name, cameras))

    _print(f"Loading {len(jobs)} episode(s) with {n_workers} parallel readers ...")
    results_ordered = [None] * len(jobs)

    with _tqdm(total=len(jobs), desc="  Loading", unit="ep") as pbar:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_load_one_episode, *job): i for i, job in enumerate(jobs)}
            for fut in as_completed(futs):
                i = futs[fut]
                results_ordered[i] = fut.result()
                pbar.update(1)

    episodes = []
    for ep_dict in results_ordered:
        if ep_dict is None:
            continue
        cam_info = ", ".join(
            f"{c}:{ep_dict[f'cam_{c}'].shape}"
            for c in cameras if f"cam_{c}" in ep_dict
        )
        _print(f"  {ep_dict['name']}: T={len(ep_dict['actions'])} "
               f"success={ep_dict['success']}"
               + (f"  [{cam_info}]" if cam_info else ""))
        episodes.append(ep_dict)
    return episodes


# ── name helpers ───────────────────────────────────────────────────────────────

def make_action_names(dim: int) -> list:
    if dim == 8:
        return [f"left_joint_{i+1}" for i in range(7)] + ["gripper"]
    return [f"action_{i}" for i in range(dim)]

def make_state_names(dim: int) -> list:
    if dim == 22:
        return (
            [f"left_joint_{i+1}"  for i in range(7)] +
            [f"right_joint_{i+1}" for i in range(7)] +
            [f"left_finger_{i+1}" for i in range(4)] +
            [f"right_finger_{i+1}"for i in range(4)]
        )
    return [f"joint_{i}" for i in range(dim)]


# ── parquet helpers ────────────────────────────────────────────────────────────

def write_parquet(path: Path, table: pa.Table):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


# ── main conversion ────────────────────────────────────────────────────────────

def hdf5_to_lerobot(
    hdf5_files: list,
    out: str,
    task_description: str,
    fps: int,
    cameras: list,
    chunks_size: int,
):
    out_dir = Path(out)
    if out_dir.exists():
        _print(f"Removing existing {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    _print(f"Video encoder: {_ENCODER_LABEL}")

    episodes = load_episodes(hdf5_files, cameras)
    if not episodes:
        raise ValueError("No valid episodes found.")

    active_cameras = [c for c in cameras if f"cam_{c}" in episodes[0]]
    if cameras and not active_cameras:
        _print("Warning: no camera data found — state-only dataset")

    action_dim     = episodes[0]["actions"].shape[1]
    state_dim      = episodes[0]["states"].shape[1]
    total_episodes = len(episodes)

    cam_hw = {}
    for cam in active_cameras:
        h, w = episodes[0][f"cam_{cam}"].shape[1:3]
        cam_hw[cam] = (h, w)

    _print(f"\n{total_episodes} episodes | action={action_dim}D | state={state_dim}D"
           + (f" | cameras={list(cam_hw.keys())}" if cam_hw else ""))

    # ── build episode metadata and flat index arrays ──────────────────────────
    ep_meta    = []
    global_idx = 0
    cam_chunk_frames: dict = {cam: {} for cam in active_cameras}
    cam_ep_timestamps: dict = {cam: {} for cam in active_cameras}
    all_actions_parts   = []
    all_states_parts    = []

    _print("\nBuilding metadata ...")
    for ep_idx, ep in enumerate(_tqdm(episodes, desc="  Episodes", unit="ep")):
        T       = len(ep["actions"])
        ci      = ep_idx // chunks_size
        ep_from = global_idx
        ep_to   = global_idx + T - 1
        global_idx += T

        all_actions_parts.append(ep["actions"])
        all_states_parts.append(ep["states"])

        for cam in active_cameras:
            if f"cam_{cam}" not in ep:
                continue
            cam_chunk_frames[cam].setdefault(ci, [])
            offset    = sum(len(f) for f in cam_chunk_frames[cam][ci])
            from_ts   = offset / fps
            to_ts     = (offset + T) / fps
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
            **{f"stats/action/{k}": v           for k, v in a_stats.items()},
            **{f"stats/observation.state/{k}": v for k, v in s_stats.items()},
        }
        for cam in active_cameras:
            if ep_idx in cam_ep_timestamps.get(cam, {}):
                fts, tts = cam_ep_timestamps[cam][ep_idx]
                meta[f"videos/observation.images.{cam}/chunk_index"] = ci
                meta[f"videos/observation.images.{cam}/file_index"]  = 0
                meta[f"videos/observation.images.{cam}/from_timestamp"] = fts
                meta[f"videos/observation.images.{cam}/to_timestamp"]   = tts
        ep_meta.append(meta)

    total_frames = global_idx
    chunk_ids    = sorted({m["data/chunk_index"] for m in ep_meta})

    # Vectorised flat arrays — no Python loop over individual frames
    all_actions_np = np.concatenate(all_actions_parts, axis=0)
    all_states_np  = np.concatenate(all_states_parts,  axis=0)
    ep_lengths     = [len(ep["actions"]) for ep in episodes]
    all_ep_idx     = np.repeat(np.arange(total_episodes), ep_lengths)
    all_frame_idx  = np.concatenate([np.arange(T) for T in ep_lengths])
    all_global_idx = np.arange(total_frames)
    all_timestamps = all_frame_idx.astype(np.float32) / fps
    all_task_idx   = np.zeros(total_frames, dtype=np.int64)

    # ── encode videos in parallel (NVENC handles many concurrent streams) ─────
    if active_cameras:
        encode_jobs = []
        for cam in active_cameras:
            for ci, frame_list in cam_chunk_frames[cam].items():
                frames = np.concatenate(frame_list, axis=0)
                path   = out_dir / f"videos/observation.images.{cam}/chunk-{ci:03d}/file-000.mp4"
                encode_jobs.append((cam, ci, frames, path))

        # NVENC: each stream is lightweight on GPU — scale up to saturate the encoder.
        # CPU libx264: one worker per physical core pair to avoid cache thrashing.
        import os
        n_cpu_cores = os.cpu_count() or 4
        n_enc = min(len(encode_jobs), 16) if "nvenc" in _ENCODER_LABEL else min(len(encode_jobs), n_cpu_cores // 2)
        n_enc = max(n_enc, 1)
        _print(f"\nEncoding {len(encode_jobs)} video(s)  [{n_enc} parallel workers / {_ENCODER_LABEL}] ...")

        with _tqdm(total=len(encode_jobs), desc="Encoding", unit="video") as vbar:
            def _enc(job):
                cam, ci, frames, path = job
                encode_video_ffmpeg(frames, path, fps)
                vbar.set_postfix_str(f"{cam} chunk-{ci:03d} ({len(frames)} fr)")
                vbar.update(1)

            with ThreadPoolExecutor(max_workers=n_enc) as pool:
                list(pool.map(_enc, encode_jobs))

    # ── write data + episodes parquets in parallel ───────────────────────────
    _print("\nWriting parquet ...")

    def _write_data_chunk(ci):
        ep_set = {m["episode_index"] for m in ep_meta if m["data/chunk_index"] == ci}
        mask   = np.isin(all_ep_idx, list(ep_set))
        table  = pa.table({
            "action":             pa.array(all_actions_np[mask].tolist(), pa.list_(pa.float32())),
            "observation.state":  pa.array(all_states_np[mask].tolist(),  pa.list_(pa.float32())),
            "timestamp":          pa.array(all_timestamps[mask].tolist(), pa.float32()),
            "frame_index":        pa.array(all_frame_idx[mask].tolist(),  pa.int64()),
            "episode_index":      pa.array(all_ep_idx[mask].tolist(),     pa.int64()),
            "index":              pa.array(all_global_idx[mask].tolist(), pa.int64()),
            "task_index":         pa.array(all_task_idx[mask].tolist(),   pa.int64()),
        })
        path = out_dir / f"data/chunk-{ci:03d}/file-000.parquet"
        write_parquet(path, table)
        return f"  {path}  ({int(mask.sum())} frames)"

    def _write_episodes_chunk(ci):
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
        }
        for stat_key in ("action", "observation.state"):
            for q in ("min","max","mean","std","q01","q10","q50","q90","q99"):
                cols[f"stats/{stat_key}/{q}"] = list_col(f"stats/{stat_key}/{q}")
            cols[f"stats/{stat_key}/count"] = list_col(f"stats/{stat_key}/count", pa.int64())
        for cam in active_cameras:
            base = f"videos/observation.images.{cam}"
            cols[f"{base}/chunk_index"]    = col(f"{base}/chunk_index",    pa.int64())
            cols[f"{base}/file_index"]     = col(f"{base}/file_index",     pa.int64())
            cols[f"{base}/from_timestamp"] = col(f"{base}/from_timestamp", pa.float64())
            cols[f"{base}/to_timestamp"]   = col(f"{base}/to_timestamp",   pa.float64())

        path = out_dir / f"meta/episodes/chunk-{ci:03d}/file-000.parquet"
        write_parquet(path, pa.table(cols))
        return f"  {path}  ({len(rows)} episodes)"

    # Data chunks + episode chunks can all be built independently
    pq_jobs = [("data", ci) for ci in chunk_ids] + [("ep", ci) for ci in chunk_ids]
    n_pq = min(len(pq_jobs), 8)
    with _tqdm(total=len(pq_jobs), desc="  Parquet", unit="file") as pbar:
        def _pq_job(job):
            kind, ci = job
            msg = _write_data_chunk(ci) if kind == "data" else _write_episodes_chunk(ci)
            pbar.update(1)
            return msg

        with ThreadPoolExecutor(max_workers=n_pq) as pool:
            for msg in pool.map(_pq_job, pq_jobs):
                _print(msg)

    # ── tasks.parquet ─────────────────────────────────────────────────────────
    tasks_df = pd.DataFrame({"task_index": [0]}, index=[task_description])
    tasks_df.index.name = None
    tasks_path = out_dir / "meta/tasks.parquet"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_df.to_parquet(tasks_path)

    # ── info.json ─────────────────────────────────────────────────────────────
    features = {
        "action":             {"dtype": "float32", "shape": [action_dim], "names": make_action_names(action_dim)},
        "observation.state":  {"dtype": "float32", "shape": [state_dim],  "names": make_state_names(state_dim)},
        "timestamp":          {"dtype": "float32", "shape": [1], "names": None},
        "frame_index":        {"dtype": "int64",   "shape": [1], "names": None},
        "episode_index":      {"dtype": "int64",   "shape": [1], "names": None},
        "index":              {"dtype": "int64",   "shape": [1], "names": None},
        "task_index":         {"dtype": "int64",   "shape": [1], "names": None},
    }
    for cam in active_cameras:
        h, w = cam_hw[cam]
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": [h, w, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": h, "video.width": w,
                "video.codec": "h264", "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False, "video.fps": fps,
                "video.channels": 3, "has_audio": False,
            },
        }
    info = {
        "codebase_version": "v3.0", "robot_type": "openarm",
        "total_episodes": total_episodes, "total_frames": total_frames,
        "total_tasks": 1, "chunks_size": chunks_size, "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path":  "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "meta/info.json", "w") as fp:
        json.dump(info, fp, indent=2)

    # ── stats.json (GPU-accelerated, cameras in parallel) ────────────────────
    _print("\nComputing statistics ...")
    stats = {
        "action":            compute_stats(all_actions_np),
        "observation.state": compute_stats(all_states_np),
    }

    def _cam_stats(cam):
        all_imgs = np.concatenate(
            [ep[f"cam_{cam}"] for ep in episodes if f"cam_{cam}" in ep], axis=0
        )
        return cam, compute_image_stats(all_imgs)

    # Run cameras concurrently — GPU compute_image_stats releases the GIL via torch.
    # With 3 cameras and 32GB VRAM they'll queue on the GPU pipeline automatically.
    n_stat_workers = min(len(active_cameras), 3)
    _print(f"  Image stats: {len(active_cameras)} camera(s)  [{n_stat_workers} workers] ...")
    with _tqdm(total=len(active_cameras), desc="  Cam stats", unit="cam") as sbar:
        def _cam_stats_job(cam):
            result = _cam_stats(cam)
            sbar.update(1)
            return result

        with ThreadPoolExecutor(max_workers=n_stat_workers) as pool:
            for cam, cam_stat in pool.map(_cam_stats_job, active_cameras):
                stats[f"observation.images.{cam}"] = cam_stat

    with open(out_dir / "meta/stats.json", "w") as fp:
        json.dump(stats, fp, indent=2)

    # ── summary ───────────────────────────────────────────────────────────────
    _print(f"\n✓  {out_dir}")
    _print(f"   {total_episodes} episodes  |  {total_frames} frames  |  {fps} fps")
    _print(f"   action {action_dim}D  |  state {state_dim}D")
    for cam in active_cameras:
        h, w = cam_hw[cam]
        _print(f"   {cam}: {h}×{w}")
    _print(f"\nLoad with:")
    _print(f"  from lerobot.datasets.lerobot_dataset import LeRobotDataset")
    _print(f"  ds = LeRobotDataset(repo_id='local/openarm', root='{out_dir}', download_videos=False)")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HDF5 → LeRobot v3.0", epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdf5",        nargs="+", required=True)
    ap.add_argument("--output",      required=True)
    ap.add_argument("--task",        default="manipulation task")
    ap.add_argument("--fps",         type=int, default=20)
    ap.add_argument("--cameras",     nargs="*", default=[], metavar="CAM")
    ap.add_argument("--chunks-size", type=int, default=100,
                    help="Episodes per chunk (smaller = more parallel encode jobs, default 100)")
    ap.add_argument("--load-workers",type=int, default=8,
                    help="Parallel HDF5 reader threads (default 8)")
    args = ap.parse_args()

    hdf5_to_lerobot(
        hdf5_files=args.hdf5,
        out=args.output,
        task_description=args.task,
        fps=args.fps,
        cameras=args.cameras,
        chunks_size=args.chunks_size,
    )
