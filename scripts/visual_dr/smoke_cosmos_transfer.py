# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the runtime Cosmos backend on a recorded DRFrame without launching Kit.

The torch file contains NHWC rgb, depth, preserve, and optional segmentation
buffers. Use only trusted torch files. Compare --mask_guidance and
--no-mask_guidance with the same recipe and seed; outputs contain no pixel paste.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from isaaclab_contrib.visual_dr.backends import DRFrame, DRRequest
from isaaclab_contrib.visual_dr.cfg import CameraDRCfg, CosmosBackendCfg, PromptBankCfg, VisualDRCfg
from isaaclab_contrib.visual_dr.cosmos import CosmosBackend
from isaaclab_contrib.visual_dr.recipes import apply_visual_dr_recipe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", required=True, help="Trusted torch file containing recorded DRFrame tensors")
    parser.add_argument("--recipe", default=str(Path(__file__).parent / "recipes/cosmos_nano.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mask_guidance", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prompt", default="A photorealistic laboratory at sunset, warm light through the windows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="/tmp/dr_smoke")
    args = parser.parse_args()
    cfg = VisualDRCfg(
        backend=CosmosBackendCfg(
            class_type=CosmosBackend, device=args.device, prompts=PromptBankCfg(variants=(args.prompt,))
        ),
        cameras={"recorded": CameraDRCfg(composite_foreground=False)},
    )
    cfg = apply_visual_dr_recipe(cfg, args.recipe)
    if args.checkpoint is not None:
        cfg.backend.checkpoint = args.checkpoint
    if args.mask_guidance is not None:
        cfg.backend.mask_guidance = args.mask_guidance
    torch.cuda.set_device(args.device)
    recorded = torch.load(args.frame, map_location=args.device, weights_only=True)
    frame = DRFrame(
        **{name: recorded[name] for name in ("rgb", "depth", "preserve", "segmentation") if name in recorded}
    )
    frame.validate()
    # Saved masks are already prepared; recipe erosion is applied by observation
    # collection, not repeated on an existing DRFrame.
    request = DRRequest(
        torch.full((frame.num_envs,), args.seed, device=args.device), (args.prompt,) * frame.num_envs, "recorded"
    )
    cfg.backend.max_batch = frame.num_envs
    backend = CosmosBackend(cfg.backend)
    try:
        backend.activate()
        generated = backend.generate(frame, request)
        assert generated.shape == frame.rgb.shape and generated.dtype == torch.uint8
        output = Path(args.out)
        output.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(generated.cpu()):
            Image.fromarray(image.numpy()).save(output / f"generated_{index}.png")
        print(f"Generated {frame.num_envs} frames in {output}; mask_guidance={cfg.backend.mask_guidance}")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
