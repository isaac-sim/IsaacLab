# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the visual DR wiring over a rollout and check it against expectations.

This is a wiring check rather than a benchmark. It runs a vectorized rollout with
short episodes so resets land mid-run, and asserts the things that are easy to get
silently wrong: that generation happens only on frames a policy consumes, that an
episode keeps one style, that resetting one environment leaves the others alone,
and that residency can be handed to a learner and taken back.

    .venv/bin/python scripts/visual_dr/run_rollout.py --num_envs 4 --steps 24
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("run_rollout")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=24)
parser.add_argument(
    "--decision_period",
    type=int,
    default=None,
    help="Environment actions per policy decision; defaults to the task config",
)
parser.add_argument("--probability", type=float, default=None, help="Override the task config's restyle probability")
parser.add_argument("--episode_length_s", type=float, default=0.5, help="Short, so resets happen mid-rollout")
parser.add_argument(
    "--workers",
    default="",
    help="Comma-separated GPU indices to run Cosmos workers on, e.g. 3,6. Empty keeps generation in this process.",
)
parser.add_argument(
    "--no_composite",
    action="store_true",
    help="Let the model render the foreground too, instead of pasting the render back",
)
parser.add_argument("--num_steps", type=int, default=None, help="Override the backend's sampler steps")
parser.add_argument("--checkpoint", default=None, help="Registered name, s3:// URI, or local checkpoint directory")
parser.add_argument("--guidance", type=float, default=None, help="Classifier-free guidance on the prompt")
parser.add_argument(
    "--mask_guidance", action="store_true", help="Send the preserved mask to Cosmos for guided denoising"
)
parser.add_argument("--backend", choices=("cosmos", "passthrough"), default="passthrough")
parser.add_argument("--offload_at", default="", help="Comma-separated steps at which to offload and re-activate")
parser.add_argument(
    "--policy",
    choices=("scripted", "zero"),
    default="scripted",
    help="scripted drives a pick-and-stack so the scene actually moves",
)
parser.add_argument(
    "--video_dir",
    default="",
    help="Directory for per-camera mp4s; empty disables recording. Distinct from\n"
    "AppLauncher's --video, which records Kit's viewport instead.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import contextlib
import sys
import time
from pathlib import Path

import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_contrib.visual_dr import PassthroughBackend, RemoteCosmosBackendCfg, VisualDRRuntime

from isaaclab_tasks.contrib.stack.config.franka.stack_visual_dr_env_cfg import FrankaStackRuntimeDRCfg
from isaaclab_tasks.utils import PresetCfg

sys.path.insert(0, str(Path(__file__).parent))
from scripted_policy import ScriptedStackPolicy  # noqa: E402


class CountingBackend:
    """Wraps a backend to record what the runtime actually asked it to do."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.frames = 0
        self.seconds = 0.0
        self.residency: list[str] = []

    def activate(self) -> None:
        self.residency.append("activate")
        self.inner.activate()

    def generate(self, frame, request):
        self.calls += 1
        self.frames += frame.num_envs
        start = time.perf_counter()
        result = self.inner.generate(frame, request)
        self.seconds += time.perf_counter() - start
        return result

    def offload(self) -> None:
        self.residency.append("offload")
        self.inner.offload()

    def close(self) -> None:
        self.residency.append("close")
        self.inner.close()


def _write_videos(recording: dict[str, list], out: Path, env) -> None:
    """Write one mp4 per stream at the environment's own step rate."""
    import imageio.v2 as imageio

    out.mkdir(parents=True, exist_ok=True)
    fps = max(1, round(1.0 / (env.cfg.sim.dt * env.cfg.decimation)))
    for name, frames in recording.items():
        path = out / f"{name}.mp4"
        # macro_block_size=1 keeps the sensor's own resolution instead of padding
        # it to a multiple of 16, so the video matches what the policy saw.
        with imageio.get_writer(path, fps=fps, macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(frame[..., :3])
        print(f"[rollout] wrote {path} ({len(frames)} frames @ {fps} fps)", flush=True)


def main() -> None:
    cfg = FrankaStackRuntimeDRCfg()
    cfg.scene.num_envs = args.num_envs
    physics = cfg.sim.physics
    if isinstance(physics, PresetCfg):
        cfg.sim.physics = physics.default
    cfg.episode_length_s = args.episode_length_s
    if args.probability is not None:
        cfg.visual_dr.probability = args.probability
    if args.decision_period is not None:
        cfg.visual_dr.decision_period = args.decision_period
    if args.checkpoint:
        cfg.visual_dr.backend.checkpoint = args.checkpoint
    if args.guidance is not None:
        cfg.visual_dr.backend.guidance = args.guidance
    if args.mask_guidance:
        cfg.visual_dr.backend.mask_guidance = True
    if args.num_steps is not None:
        cfg.visual_dr.backend.num_steps = args.num_steps

    if args.no_composite:
        for camera_cfg in cfg.visual_dr.cameras.values():
            camera_cfg.composite_foreground = False

    worker_devices = tuple(int(d) for d in args.workers.split(",") if d.strip())
    if worker_devices:
        # Same settings, generated elsewhere: copy the configured backend across so
        # the only difference from an in-process run is where the model lives.
        from isaaclab_contrib.visual_dr.remote import RemoteCosmosBackend

        source = cfg.visual_dr.backend
        cfg.visual_dr.backend = RemoteCosmosBackendCfg(
            class_type=RemoteCosmosBackend,
            devices=worker_devices,
            max_batch=len(worker_devices),
            checkpoint=source.checkpoint,
            control_kind=source.control_kind,
            control_guidance=source.control_guidance,
            control_weight=source.control_weight,
            depth_range_m=source.depth_range_m,
            num_steps=source.num_steps,
            resolution=source.resolution,
            aspect_ratio=source.aspect_ratio,
            guidance=source.guidance,
            compile=source.compile,
            fp8=source.fp8,
            prompts=source.prompts,
        )
    if args.backend == "passthrough":
        cfg.visual_dr.backend.class_type = PassthroughBackend

    env = ManagerBasedRLEnv(cfg=cfg)
    runtime = VisualDRRuntime(cfg.visual_dr, env.num_envs, env.device)
    backend = CountingBackend(runtime.backend)
    runtime.backend = backend
    env.visual_dr_runtime = runtime
    runtime.activate()

    offload_steps = {int(s) for s in args.offload_at.split(",") if s.strip()}
    recording: dict[str, list] = {}
    cameras = tuple(cfg.visual_dr.cameras)
    expected_frames = 0
    episode_styles: dict[tuple[int, int], int] = {}
    style_violations = 0
    reset_steps = 0

    try:
        policy = ScriptedStackPolicy(env) if args.policy == "scripted" else None
        env.reset()
        # reset() computes observations too, so its frames count towards the total.
        expected_frames += int(runtime._restyle.sum()) * len(cameras)
        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        print(
            f"[rollout] {env.num_envs} envs, decision_period={cfg.visual_dr.decision_period}, "
            f"probability={cfg.visual_dr.probability}, cameras={cameras}",
            flush=True,
        )
        print(
            f"[rollout] reset      | episodes={runtime._episode_ids.tolist()} "
            f"consumed={runtime._consumed.int().tolist()} "
            f"restyled={runtime._restyle.int().tolist()}",
            flush=True,
        )

        for step in range(args.steps):
            if step in offload_steps:
                # The point of offload is that a learner can use the memory, so
                # report what it actually frees rather than just that it ran.
                def memory() -> str:
                    gib = 2**30
                    return (
                        f"alloc={torch.cuda.memory_allocated() / gib:.1f} "
                        f"reserved={torch.cuda.memory_reserved() / gib:.1f} "
                        f"free={torch.cuda.mem_get_info()[0] / gib:.1f}"
                    )

                print(f"[rollout] step {step}: before offload  {memory()}", flush=True)
                runtime.offload()
                print(f"[rollout] step {step}: after offload   {memory()}", flush=True)
                runtime.activate()
                print(f"[rollout] step {step}: after reactivate {memory()}", flush=True)
            if policy is not None:
                actions = policy.act()
            observations = env.step(actions)[0]

            if args.video_dir:
                for camera in cameras:
                    sensor = env.scene.sensors[camera]
                    recording.setdefault(f"{camera}_raw", []).append(
                        sensor.data.output["rgb"].torch[0].detach().cpu().numpy().copy()
                    )
                    recording.setdefault(f"{camera}_policy_input", []).append(
                        observations["policy"][camera][0].detach().cpu().numpy().copy()
                    )

            consumed = runtime._consumed
            restyled = runtime._restyle
            episodes = runtime._episode_ids
            if bool((env.episode_length_buf == 0).any()):
                reset_steps += 1
            # Each camera is generated separately, for the environments selected.
            expected_frames += int(restyled.sum()) * len(cameras)

            # A style must not change while an episode continues.
            for env_id in range(env.num_envs):
                key = (env_id, int(episodes[env_id]))
                seed = int(runtime._style_seeds[env_id])
                if episode_styles.setdefault(key, seed) != seed:
                    style_violations += 1

            print(
                f"[rollout] step {step:3d} | episodes={episodes.tolist()} "
                f"consumed={consumed.int().tolist()} restyled={restyled.int().tolist()}",
                flush=True,
            )

        print("\n[rollout] ---- summary ----", flush=True)
        print(f"[rollout] steps                : {args.steps}", flush=True)
        print(f"[rollout] steps with a reset   : {reset_steps}", flush=True)
        print(f"[rollout] episodes seen        : {sorted({e for _, e in episode_styles})}", flush=True)
        print(
            f"[rollout] backend calls        : {backend.calls} "
            f"(frames per call bounded by max_batch={cfg.visual_dr.backend.max_batch})",
            flush=True,
        )
        print(f"[rollout] frames generated     : {backend.frames} (expected {expected_frames})", flush=True)
        if backend.frames:
            print(f"[rollout] mean seconds / frame : {backend.seconds / backend.frames:.3f}", flush=True)
        print(f"[rollout] residency events     : {backend.residency}", flush=True)
        print(f"[rollout] style violations     : {style_violations}", flush=True)
        print(f"[rollout] passthrough errors   : {len(runtime.errors)}", flush=True)
        if policy is not None:
            print(f"[rollout] policy phase / env   : {policy.phase.tolist()}", flush=True)
        if recording:
            _write_videos(recording, Path(args.video_dir), env)

        ok = backend.frames == expected_frames and style_violations == 0
        print(f"\n[rollout] WIRING {'OK' if ok else 'MISMATCH'}", flush=True)
    finally:
        with contextlib.suppress(Exception):
            runtime.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
