# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drive the Franka stacking demo with runtime visual DR and dump what the policy reads.

This is the first honest look at output quality: unlike the standalone smoke
test, the control map here is the renderer's real ``distance_to_image_plane`` and
the preserved region is a real semantic mask.

    .venv/bin/python scripts/visual_dr/run_demo.py --num-envs 2 --steps 8 --out /tmp/dr_demo
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("run_demo")
parser.add_argument("--num-envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--out", default="/tmp/dr_demo")
parser.add_argument("--camera", default="table_cam")
parser.add_argument("--backend", choices=("cosmos", "passthrough"), default="cosmos")
parser.add_argument("--probability", type=float, default=1.0, help="1.0 so every dumped frame is generated")
parser.add_argument("--max-batch", type=int, default=None, help="Override the backend's frames per call")
parser.add_argument("--num-steps", type=int, default=None, help="Override sampler steps")
parser.add_argument("--guidance", type=float, default=None, help="Classifier-free guidance on the prompt")
parser.add_argument(
    "--control-guidance",
    type=float,
    default=None,
    help="Lower lets the prompt invent background geometry instead of following depth",
)
parser.add_argument("--prompt", default=None, help="Use a single prompt instead of the task bank")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# Camera sensors need the rendering extensions in a headless, viewport-free launch.
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import contextlib
from pathlib import Path

import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_contrib.visual_dr import PassthroughBackend, PromptBankCfg, VisualDRRuntime
from isaaclab_contrib.visual_dr.demo import FrankaStackRuntimeDRCfg
from isaaclab_contrib.visual_dr.observations import preserve_mask

from isaaclab_tasks.utils import PresetCfg


def save_contact_sheet(path: Path, panels: list[tuple[str, torch.Tensor]]) -> None:
    """Write raw | depth | mask | output side by side, the way a bad frame is diagnosed."""
    from PIL import Image, ImageDraw

    images = []
    for label, tensor in panels:
        array = tensor.detach().cpu()
        if array.dtype != torch.uint8:
            finite = torch.isfinite(array)
            clean = torch.where(finite, array, torch.zeros_like(array))
            lo, hi = clean.min(), clean.max()
            array = ((clean - lo) / (hi - lo).clamp_min(1e-6) * 255).to(torch.uint8)
        if array.shape[-1] == 1:
            array = array.repeat(1, 1, 3)
        image = Image.fromarray(array.numpy())
        ImageDraw.Draw(image).text((4, 4), label, fill=(255, 0, 0))
        images.append(image)

    width = sum(i.width for i in images)
    sheet = Image.new("RGB", (width, max(i.height for i in images)))
    offset = 0
    for image in images:
        sheet.paste(image, (offset, 0))
        offset += image.width
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    out = Path(args.out)
    cfg = FrankaStackRuntimeDRCfg()
    cfg.scene.num_envs = args.num_envs
    # Constructing the config directly skips Hydra, which is what normally resolves
    # the physics preset ``StackEnvCfg`` leaves on ``sim.physics``.
    physics = cfg.sim.physics
    if isinstance(physics, PresetCfg):
        cfg.sim.physics = physics.default
    cfg.visual_dr.probability = args.probability
    if args.max_batch is not None:
        cfg.visual_dr.backend.max_batch = args.max_batch
    if args.num_steps is not None:
        cfg.visual_dr.backend.num_steps = args.num_steps
    if args.guidance is not None:
        cfg.visual_dr.backend.guidance = args.guidance
    if args.control_guidance is not None:
        cfg.visual_dr.backend.control_guidance = args.control_guidance
    if args.prompt:
        cfg.visual_dr.backend.prompts = PromptBankCfg(variants=(args.prompt,))
    if args.backend == "passthrough":
        cfg.visual_dr.backend.class_type = PassthroughBackend

    env = ManagerBasedRLEnv(cfg=cfg)
    # Attached only now: constructing the environment probes observation shapes,
    # and those probes must never load a diffusion model.
    runtime = VisualDRRuntime(cfg.visual_dr, env.num_envs, env.device)
    env.visual_dr_runtime = runtime
    runtime.activate()

    try:
        env.reset()
        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        for step in range(args.steps):
            observations = env.step(actions)[0]
            sensor = env.scene.sensors[args.camera]
            raw = sensor.data.output["rgb"].torch
            depth = sensor.data.output["distance_to_image_plane"].torch
            info = sensor.data.info.get("semantic_segmentation") or {}
            mask = preserve_mask(
                sensor.data.output["semantic_segmentation"].torch,
                info.get("idToLabels", {}),
                cfg.visual_dr.cameras[args.camera],
            )
            processed = observations["policy"][args.camera]
            save_contact_sheet(
                out / f"step_{step:03d}.png",
                [
                    ("raw", raw[0]),
                    ("depth", depth[0]),
                    ("preserve", mask[0].to(torch.uint8) * 255),
                    ("policy input", processed[0]),
                ],
            )
            print(f"[demo] step {step}: wrote {out / f'step_{step:03d}.png'}", flush=True)
        if runtime.errors:
            print(f"[demo] {len(runtime.errors)} frames passed through unrandomized: {runtime.errors[:3]}")
    finally:
        with contextlib.suppress(Exception):
            runtime.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
