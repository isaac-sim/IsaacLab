# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure sampler steps against latency and appearance, on a real simulated frame.

Runs inside a live Isaac Sim so the timings reflect the attention backend the
rollout actually uses, and loads the model once so the sweep is cheap. Writes one
contact sheet comparing every step count against the raw frame, because the only
way to choose a step count is to look at what it produces.

    .venv/bin/python scripts/visual_dr/bench_steps.py --steps 1,2,4,8 --repeats 3
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("bench_steps")
parser.add_argument("--steps", default="1,2,3,4,6,8", help="Comma-separated sampler step counts")
parser.add_argument("--repeats", type=int, default=3, help="Generations per step count, after a warmup")
parser.add_argument("--camera", default="table_cam")
parser.add_argument("--out", default="/tmp/dr_steps")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import time
from pathlib import Path

import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_contrib.visual_dr.backends import DRFrame, DRRequest
from isaaclab_contrib.visual_dr.cosmos import CosmosBackend
from isaaclab_contrib.visual_dr.observations import preserve_mask

from isaaclab_tasks.contrib.stack.config.franka.stack_visual_dr_env_cfg import FrankaStackRuntimeDRCfg
from isaaclab_tasks.utils import PresetCfg


def main() -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    step_counts = [int(s) for s in args.steps.split(",") if s.strip()]

    cfg = FrankaStackRuntimeDRCfg()
    cfg.scene.num_envs = 1
    physics = cfg.sim.physics
    if isinstance(physics, PresetCfg):
        cfg.sim.physics = physics.default

    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()

    sensor = env.scene.sensors[args.camera]
    rgb = sensor.data.output["rgb"].torch.clone()
    depth = sensor.data.output["distance_to_image_plane"].torch.clone()
    info = sensor.data.info.get("semantic_segmentation") or {}
    segmentation = sensor.data.output["semantic_segmentation"].torch.clone()
    mask = preserve_mask(segmentation, info.get("idToLabels", {}), cfg.visual_dr.cameras[args.camera]).clone()
    frame = DRFrame(rgb, depth, mask, segmentation)

    backend_cfg = cfg.visual_dr.backend
    backend_cfg.max_batch = 1
    backend = CosmosBackend(backend_cfg)
    backend.activate()

    print(
        f"[bench] camera {rgb.shape[2]}x{rgb.shape[1]} | hint={backend_cfg.control_kind} "
        f"| resolution={backend_cfg.resolution} aspect={backend_cfg.aspect_ratio} "
        f"| guidance={backend_cfg.guidance} control_guidance={backend_cfg.control_guidance} "
        f"| max_batch={backend_cfg.max_batch}",
        flush=True,
    )
    prompt = backend_cfg.prompts.variants[0]
    request = DRRequest(torch.zeros(1, dtype=torch.long, device=rgb.device), (prompt,), args.camera)

    panels = [("raw", rgb[0])]
    results = []
    for count in step_counts:
        backend_cfg.num_steps = count
        # One untimed generation first: the first call at a new shape pays for
        # kernel selection, which would otherwise land entirely on this row.
        generated = backend.generate(frame, request)
        timings = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            generated = backend.generate(frame, request)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
        mean = sum(timings) / len(timings)
        results.append((count, mean, min(timings), max(timings)))
        composited = torch.where(frame.preserve, frame.rgb, generated)
        panels.append((f"{count} steps", composited[0]))
        print(
            f"[bench] {count:2d} steps: mean {mean:.3f}s  min {min(timings):.3f}s  max {max(timings):.3f}s", flush=True
        )

    print("\n[bench] steps | mean s | min s | max s | frames/min", flush=True)
    for count, mean, low, high in results:
        print(f"[bench] {count:5d} | {mean:6.3f} | {low:5.3f} | {high:5.3f} | {60.0 / mean:9.1f}", flush=True)
    # Cost is close to affine in steps; report the fit so other counts can be predicted.
    if len(results) >= 2:
        (c0, m0, *_), (c1, m1, *_) = results[0], results[-1]
        per_step = (m1 - m0) / max(c1 - c0, 1)
        print(f"[bench] fit: {per_step:.3f} s per step + {m0 - per_step * c0:.3f} s fixed", flush=True)

    from PIL import Image, ImageDraw

    images = []
    for label, tensor in panels:
        image = Image.fromarray(tensor.detach().cpu().numpy())
        ImageDraw.Draw(image).text((4, 4), label, fill=(255, 0, 0))
        images.append(image)
    sheet = Image.new("RGB", (sum(i.width for i in images), max(i.height for i in images)))
    offset = 0
    for image in images:
        sheet.paste(image, (offset, 0))
        offset += image.width
    sheet.save(out / "steps.png")
    print(f"\n[bench] wrote {out / 'steps.png'}", flush=True)

    backend.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
