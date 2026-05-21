# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reproduction benchmark comparing a static base camera vs a wrist-mounted camera in dexsuite-reorient.

Piggy-backs on ``Isaac-Dexsuite-Kuka-Allegro-Reorient-v0``. Both runs use the existing
``single_camera`` scene + observation presets so the obs schema is identical between
configurations. Only the camera's ``prim_path`` and ``offset`` change between runs
(static world-frame vs ``/World/envs/env_.*/Robot/ee_link/palm_link/Camera``).

Outputs ``fps_<mode>.csv``, ``summary.csv``, ``summary.txt``, ``args.json`` and
per-env ``videos/<mode>_envN.mp4`` files in a single timestamped output directory.

Run via the workspace venv::

    source ../tools/activate.sh
    python scripts/benchmarks/benchmark_dexsuite_wrist_camera.py \\
        --num_envs 2 --total_epochs 10 --video_length 200 --num_video_envs 1 \\
        --resolution 128 --mode both --headless --enable_cameras \\
        --out_dir ./outputs/dexsuite_wrist_repro
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DEFAULT_TASK = "Isaac-Dexsuite-Kuka-Allegro-Reorient-v0"


def _rank_world() -> tuple[int, int]:
    """Return (rank, world_size) read from torchrun env vars, defaulting to (0, 1)."""
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def _fps_csv_path(out_dir: Path, mode: str, rank: int, world_size: int) -> Path:
    """Per-rank FPS CSV path. Single-GPU uses the plain `fps_<mode>.csv` for backward compat."""
    if world_size > 1:
        return out_dir / f"fps_{mode}_rank{rank}.csv"
    return out_dir / f"fps_{mode}.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark static vs wrist-mounted camera for dexsuite-reorient.",
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Gym task ID.")
    parser.add_argument("--num_envs", type=int, default=2, help="Number of parallel environments (1-3 typical).")
    parser.add_argument(
        "--resolution", type=int, default=128, help="Camera resolution (applied to both width and height — square)."
    )
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="env.step() calls per timed epoch.")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Epochs to discard from summary stats.")
    parser.add_argument("--total_epochs", type=int, default=10, help="Measured epochs (excluding warmup).")
    parser.add_argument(
        "--summary_last_n_epochs", type=int, default=5, help="How many trailing epochs to average for the summary."
    )
    parser.add_argument("--video_length", type=int, default=200, help="Frames per recorded video (after FPS phase).")
    parser.add_argument(
        "--num_video_envs", type=int, default=1, help="How many envs to record video from (capped at num_envs and 3)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed shared by both runs for matched actions.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./outputs/dexsuite_wrist_repro",
        help="Output directory; a timestamped subfolder is created inside (skipped when re-used by the orchestrator).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["base", "wrist", "both", "summarize"],
        default="both",
        help=(
            "Which configuration to run. 'both' spawns two subprocesses (single-GPU only); "
            "'summarize' aggregates existing per-rank fps_*.csv files into summary.csv/summary.txt "
            "without launching Isaac (used as the final step in multi-GPU runs)."
        ),
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Multi-GPU run via torchrun. Each rank reads LOCAL_RANK/RANK/WORLD_SIZE from env.",
    )
    # AppLauncher args (--headless, --enable_cameras, --device, --visualizer, ...).
    # add_launcher_args lazily imports AppLauncher inside its body, so this is light.
    from isaaclab_tasks.utils import add_launcher_args

    add_launcher_args(parser)
    return parser


def _resolve_out_dir(base_out_dir: str, mode: str) -> Path:
    """Return the timestamped run directory. Orchestrator creates it; subprocesses reuse it."""
    base = Path(base_out_dir)
    # In subprocess mode the orchestrator already passed a fully-resolved timestamped path.
    # We detect that by the presence of args.json inside.
    if (base / "args.json").exists():
        return base
    run_dir = base / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "videos").mkdir(exist_ok=True)
    return run_dir


def _record_args(out_dir: Path, args: argparse.Namespace) -> None:
    payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (out_dir / "args.json").write_text(json.dumps(payload, indent=2, default=str))


# ------------------------------------------------------------------------------------------------
# Orchestrator: --mode both
# ------------------------------------------------------------------------------------------------


def run_orchestrator(args: argparse.Namespace, hydra_args: list[str]) -> int:
    out_dir = _resolve_out_dir(args.out_dir, "both")
    _record_args(out_dir, args)

    common = [
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--resolution",
        str(args.resolution),
        "--steps_per_epoch",
        str(args.steps_per_epoch),
        "--warmup_epochs",
        str(args.warmup_epochs),
        "--total_epochs",
        str(args.total_epochs),
        "--summary_last_n_epochs",
        str(args.summary_last_n_epochs),
        "--video_length",
        str(args.video_length),
        "--num_video_envs",
        str(args.num_video_envs),
        "--seed",
        str(args.seed),
        "--out_dir",
        str(out_dir),
    ]
    # Forward AppLauncher flags relevant to headless / device / cameras.
    for flag in ("headless", "enable_cameras"):
        if getattr(args, flag, False):
            common.append(f"--{flag}")
    if getattr(args, "device", None):
        common += ["--device", args.device]

    for mode in ("base", "wrist"):
        cmd = [sys.executable, __file__, "--mode", mode, *common, *hydra_args]
        print(f"[orchestrator] launching: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"[orchestrator] subprocess for mode={mode} failed (rc={proc.returncode})", file=sys.stderr)
            return proc.returncode

    write_combined_summary(out_dir, args.summary_last_n_epochs)
    print(f"[orchestrator] done. artifacts in: {out_dir}", flush=True)
    return 0


def _read_fps_csv(csv_path: Path) -> list[tuple[int, float, bool]]:
    rows: list[tuple[int, float, bool]] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((int(r["epoch"]), float(r["fps"]), r["is_warmup"].lower() == "true"))
    return rows


def _collect_mode_csvs(out_dir: Path, mode: str) -> list[Path]:
    """Find FPS CSVs for a mode — either single `fps_<mode>.csv` or per-rank `fps_<mode>_rank*.csv`."""
    plain = out_dir / f"fps_{mode}.csv"
    if plain.exists():
        return [plain]
    return sorted(out_dir.glob(f"fps_{mode}_rank*.csv"))


def write_combined_summary(out_dir: Path, last_n: int) -> None:
    """Aggregate per-rank FPS CSVs into summary.csv + summary.txt.

    For multi-rank runs, per-epoch FPS values are summed across ranks to report total
    env-steps/sec throughput (the natural quantity for parallel renderers/simulators).
    Per-rank breakdown is also written so individual GPU behavior stays visible.
    """
    summary_rows = []
    headlines = []
    for mode in ("base", "wrist"):
        csv_paths = _collect_mode_csvs(out_dir, mode)
        if not csv_paths:
            print(f"[summary] no fps CSV for mode={mode}, skipping", file=sys.stderr)
            continue
        # Each CSV is one rank. Index per-epoch FPS across ranks, then sum.
        per_rank_rows = [_read_fps_csv(p) for p in csv_paths]
        n_epochs = min(len(r) for r in per_rank_rows)
        # Sum FPS per epoch across ranks (skipping warmup-flagged epochs).
        total_fps_per_epoch: list[float] = []
        for e in range(n_epochs):
            is_warmup_any = any(r[e][2] for r in per_rank_rows)
            if is_warmup_any:
                continue
            total_fps_per_epoch.append(sum(r[e][1] for r in per_rank_rows))
        last = total_fps_per_epoch[-last_n:] if len(total_fps_per_epoch) >= last_n else total_fps_per_epoch
        if not last:
            continue
        mean = statistics.fmean(last)
        std = statistics.stdev(last) if len(last) > 1 else 0.0
        # Per-rank breakdown for visibility into rank imbalance.
        per_rank_means = []
        for r in per_rank_rows:
            measured = [fps for _, fps, is_w in r if not is_w]
            tail = measured[-last_n:] if len(measured) >= last_n else measured
            per_rank_means.append(statistics.fmean(tail) if tail else 0.0)
        summary_rows.append(
            {
                "mode": mode,
                "world_size": len(csv_paths),
                "total_fps_mean": f"{mean:.3f}",
                "total_fps_std": f"{std:.3f}",
                "total_fps_min": f"{min(last):.3f}",
                "total_fps_max": f"{max(last):.3f}",
                "per_rank_means": ";".join(f"{m:.3f}" for m in per_rank_means),
                "last_n": len(last),
            }
        )
        headlines.append((mode, mean, std, len(csv_paths)))

    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "world_size",
                "total_fps_mean",
                "total_fps_std",
                "total_fps_min",
                "total_fps_max",
                "per_rank_means",
                "last_n",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = []
    for mode, mean, std, ws in headlines:
        suffix = f"  (world_size={ws})" if ws > 1 else ""
        lines.append(f"{mode:>5}: {mean:8.2f} +/- {std:6.2f} FPS{suffix}")
    if len(headlines) == 2:
        base_mean = headlines[0][1] if headlines[0][0] == "base" else headlines[1][1]
        wrist_mean = headlines[1][1] if headlines[1][0] == "wrist" else headlines[0][1]
        ratio = wrist_mean / base_mean if base_mean > 0 else float("nan")
        lines.append(f"wrist/base ratio: {ratio:.3f}  (delta: {(ratio - 1.0) * 100:+.1f}%)")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")


def run_summarize(args: argparse.Namespace) -> int:
    """Aggregate existing per-rank fps CSVs. Used as the final step in a multi-GPU job."""
    out_dir = _resolve_out_dir(args.out_dir, "summarize")
    write_combined_summary(out_dir, args.summary_last_n_epochs)
    summary_path = out_dir / "summary.txt"
    if summary_path.exists():
        print(f"[summarize] wrote {summary_path}:", flush=True)
        print(summary_path.read_text(), flush=True)
    else:
        print("[summarize] WARNING: no summary written (no per-mode CSVs found?)", file=sys.stderr)
    return 0


# ------------------------------------------------------------------------------------------------
# Per-config runner: --mode base | --mode wrist
# ------------------------------------------------------------------------------------------------


def setup_env_cfg(args: argparse.Namespace, mode: str):
    """Build the env config: pin the obs preset to single_camera, then swap base_camera prim_path/offset."""
    from isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.camera_cfg import (
        BASE_CAMERA_CFG,
        WRIST_CAMERA_CFG,
    )
    from isaaclab_tasks.utils import resolve_task_config

    # Pre-pend Hydra preset overrides so the resolved env_cfg.scene has a `base_camera` slot
    # and observations.base_image is wired to SceneEntityCfg("base_camera"). Users can override
    # by passing their own env.scene=... env.observations=... on the CLI; we only inject defaults.
    hydra_argv_extras = []
    user_argv = " ".join(sys.argv)
    if "env.scene=" not in user_argv:
        hydra_argv_extras.append("env.scene=single_camera")
    if "env.observations=" not in user_argv:
        hydra_argv_extras.append("env.observations=single_camera")
    if hydra_argv_extras:
        sys.argv += hydra_argv_extras

    env_cfg, _ = resolve_task_config(args.task, "")
    env_cfg.scene.num_envs = args.num_envs

    src = WRIST_CAMERA_CFG if mode == "wrist" else BASE_CAMERA_CFG
    cam = env_cfg.scene.base_camera  # already-resolved CameraCfg from preset (renderer_cfg resolved)
    cam.prim_path = src.prim_path
    cam.offset = src.offset
    cam.data_types = ["rgb"]
    cam.width = args.resolution
    cam.height = args.resolution
    return env_cfg


def run_one_config(args: argparse.Namespace, mode: str) -> int:
    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401  (registers gym envs)
    from isaaclab_tasks.utils import launch_simulation

    rank, world_size = _rank_world()
    is_rank0 = rank == 0
    tag_prefix = f"[{mode}|r{rank}]" if world_size > 1 else f"[{mode}]"

    out_dir = _resolve_out_dir(args.out_dir, mode)

    # Offset the seed by rank so each rank samples a different random action stream,
    # exercising the renderer with varied scene state. Matched across modes (base/wrist
    # use the same seed) so the comparison stays apples-to-apples.
    torch.manual_seed(args.seed + rank)
    env_cfg = setup_env_cfg(args, mode)

    fps_csv_path = _fps_csv_path(out_dir, mode, rank, world_size)
    fps_csv = fps_csv_path.open("w", newline="")
    fps_writer = csv.writer(fps_csv)
    fps_writer.writerow(["epoch", "fps", "is_warmup"])

    # Buffer frames per env in RAM during the video phase; encode via imageio inside
    # the with-block (Kit is still alive). cv2.VideoWriter is unusable here because
    # libavcodec loaded by Kit silently drops frames after env.step has rendered;
    # imageio_ffmpeg shells out to a standalone ffmpeg binary and is unaffected.
    frame_buffers: dict[int, list] = {}
    cuda_available = torch.cuda.is_available()

    def sync():
        if cuda_available:
            torch.cuda.synchronize()

    def sample_actions(action_space_shape, device):
        return torch.rand(action_space_shape, device=device) * 2.0 - 1.0

    try:
        with launch_simulation(env_cfg, args):
            env = gym.make(args.task, cfg=env_cfg)
            env.reset()
            device = env.unwrapped.device
            action_shape = env.action_space.shape

            # FPS phase --------------------------------------------------------------
            with torch.inference_mode():
                total_epochs = args.warmup_epochs + args.total_epochs
                for epoch in range(total_epochs):
                    is_warmup = epoch < args.warmup_epochs
                    sync()
                    t0 = time.perf_counter()
                    for _ in range(args.steps_per_epoch):
                        env.step(sample_actions(action_shape, device))
                    sync()
                    dt = time.perf_counter() - t0
                    fps = (args.steps_per_epoch * args.num_envs) / dt if dt > 0 else 0.0
                    fps_writer.writerow([epoch, f"{fps:.4f}", str(is_warmup)])
                    fps_csv.flush()
                    tag = "warmup" if is_warmup else "measure"
                    print(f"{tag_prefix} epoch {epoch:>3} ({tag}): {fps:.2f} FPS  (dt={dt:.3f}s)", flush=True)

            # Video phase ------------------------------------------------------------
            # All ranks step env in lockstep (so any internal collectives stay synced),
            # but only rank 0 buffers + encodes frames.
            n_video_envs = max(0, min(args.num_video_envs, args.num_envs, 3))
            if n_video_envs > 0 and args.video_length > 0:
                import numpy as np

                if is_rank0:
                    for i in range(n_video_envs):
                        frame_buffers[i] = []
                with torch.inference_mode():
                    for _ in range(args.video_length):
                        env.step(sample_actions(action_shape, device))
                        if not is_rank0:
                            continue
                        rgb = env.unwrapped.scene["base_camera"].data.output["rgb"]
                        frames = rgb[:n_video_envs].detach().cpu().numpy()
                        if frames.dtype != np.uint8:
                            f_max = float(frames.max()) if frames.size else 0.0
                            if f_max <= 1.0 + 1e-6:
                                frames = (frames.clip(0.0, 1.0) * 255.0).astype(np.uint8)
                            else:
                                frames = frames.clip(0, 255).astype(np.uint8)
                        for i in range(n_video_envs):
                            frame_buffers[i].append(frames[i].copy())
                if is_rank0:
                    print(f"{tag_prefix} video phase: captured {args.video_length} frames per env", flush=True)

            # Encode videos BEFORE the `with launch_simulation` block exits: AppLauncher.app.close()
            # tears down the process and code after the `with` is unreachable. Use imageio (with
            # imageio_ffmpeg's bundled ffmpeg binary) — cv2.VideoWriter mp4v silently drops frames
            # when libavcodec is loaded into the Kit-active runtime.
            if frame_buffers and is_rank0:
                import imageio.v2 as imageio

                for i, frames in frame_buffers.items():
                    if not frames:
                        continue
                    h, w = frames[0].shape[:2]
                    path = out_dir / "videos" / f"{mode}_env{i}.mp4"
                    with imageio.get_writer(str(path), fps=30, codec="libx264", macro_block_size=1) as writer:
                        for f in frames:
                            writer.append_data(f)
                    print(
                        f"{tag_prefix} wrote video {path} ({len(frames)} frames, {w}x{h}, {path.stat().st_size} bytes)",
                        flush=True,
                    )

            print(f"{tag_prefix} wrote {fps_csv_path}", flush=True)
            env.close()
    finally:
        fps_csv.close()

    return 0


# ------------------------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------------------------


def main() -> int:
    parser = build_parser()
    args, hydra_args = parser.parse_known_args()
    # Hydra reads from sys.argv directly (see zero_agent.py pattern).
    sys.argv = [sys.argv[0]] + hydra_args

    if args.mode == "both" and args.distributed:
        print(
            "ERROR: --mode both is not supported with --distributed. Under torchrun the script "
            "must be invoked twice (once with --mode base, once with --mode wrist), followed by a "
            "rank-0-only `--mode summarize` step to aggregate per-rank CSVs.",
            file=sys.stderr,
        )
        return 2

    if args.mode == "both":
        return run_orchestrator(args, hydra_args)
    if args.mode == "summarize":
        return run_summarize(args)
    return run_one_config(args, args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
