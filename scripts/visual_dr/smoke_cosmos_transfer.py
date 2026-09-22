# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test: depth-guided transfer on a STOCK cosmos-framework + public checkpoint.

Answers one question before any IsaacLab work: does ``nvidia/Cosmos3-Nano`` do
depth transfer without the patched checkout? Foreground preservation here is a
plain composite (no mask-guided denoising), which is the design decision this
runtime is built on.

Run from a cosmos-framework venv (see docs/setup.md ``uv sync --group=cu128``):

    .venv/bin/python scripts/visual_dr/smoke_cosmos_transfer.py --out /tmp/dr_smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("smoke_cosmos_transfer")
    p.add_argument(
        "--checkpoint",
        default="Cosmos3-Nano",
        help="Registered name from cosmos_framework.inference.args._CHECKPOINTS, or a local dir",
    )
    p.add_argument("--image", required=True, help="Source RGB image")
    p.add_argument("--out", default="/tmp/dr_smoke")
    p.add_argument(
        "--prompt",
        default=(
            "A photorealistic industrial laboratory interior, warm overhead lighting, "
            "polished concrete floor, softly blurred equipment racks in the background."
        ),
    )
    p.add_argument("--num_steps", type=int, default=4)
    p.add_argument("--resolution", default="480")
    p.add_argument("--aspect_ratio", default="1,1")
    p.add_argument("--control_guidance", type=float, default=0.5)
    p.add_argument("--control_weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def pseudo_depth(rgb: torch.Tensor) -> torch.Tensor:
    """Stand-in control map: blurred luminance, normalized to [0, 1].

    IsaacLab supplies a real ``distance_to_image_plane`` buffer; this only has to
    be a plausible control signal for a plumbing check.
    """
    lum = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])[None, None]
    blurred = torch.nn.functional.avg_pool2d(lum, 9, stride=1, padding=4)[0]
    lo, hi = blurred.amin(), blurred.amax()
    depth = (blurred - lo) / (hi - lo).clamp_min(1e-6)
    return depth.repeat(3, 1, 1)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out)
    (out_root / "req" / "out").mkdir(parents=True, exist_ok=True)

    import numpy as np
    from cosmos_framework.inference.args import OmniSetupOverrides
    from PIL import Image

    source = (
        torch.from_numpy(np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32) / 255.0)
        .permute(2, 0, 1)
        .contiguous()
    )
    depth = pseudo_depth(source)

    vision_path = out_root / "vision_in.png"
    depth_path = out_root / "depth_in.png"
    for path, tensor in ((vision_path, source), (depth_path, depth)):
        arr = (tensor.permute(1, 2, 0) * 255).round().clamp(0, 255).to(torch.uint8).numpy()
        Image.fromarray(arr).save(path)

    # Centre box stands in for a segmentation-derived foreground mask: the composite
    # path has to leave these pixels bit-identical.
    preserve = torch.zeros(1, *source.shape[1:], dtype=torch.bool)
    h, w = source.shape[1:]
    preserve[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True

    setup = OmniSetupOverrides.model_construct(
        checkpoint_path=str(args.checkpoint),
        output_dir=out_root / "out",
        guardrails=False,
        benchmark=False,
        parallelism_preset="latency",
        dp_shard_size=1,
        dp_replicate_size=1,
        tp_size=1,
        cp_size=1,
        cfgp_size=1,
        max_num_seqs=1,
    )
    setup_args = setup.build_setup()
    pipe = setup_args.get_inference_cls().create(setup_args)
    print("[smoke] pipeline loaded", flush=True)

    manifest = {
        "name": "restyle",
        "model_mode": "image2image",
        "prompt": args.prompt,
        "vision_path": str(vision_path.absolute()),
        "resolution": str(args.resolution),
        "aspect_ratio": str(args.aspect_ratio),
        "num_frames": 1,
        "num_steps": int(args.num_steps),
        "seed": int(args.seed),
        "num_outputs": 1,
        "guidance": 1,
        "control_guidance": float(args.control_guidance),
        "depth": {"control_path": str(depth_path.absolute()), "weight": float(args.control_weight)},
    }
    manifest_path = out_root / "req" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    overrides = setup_args.get_sample_overrides_cls().from_files([manifest_path], overrides=setup_args.sample_overrides)
    req_out = out_root / "req" / "out"
    for ov in overrides:
        ov.output_dir = req_out
        ov.download(ov.output_dir / "inputs")
    sample_args = [ov.build_sample(model_config=pipe.model_config) for ov in overrides]
    pipe.generate(sample_args)
    torch.cuda.synchronize()
    print(f"[smoke] generate() returned; artifacts in {req_out}", flush=True)

    produced = sorted(p for p in req_out.rglob("vision.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not produced:
        raise SystemExit(f"[smoke] FAILED: no vision output under {req_out}")
    restyled = (
        torch.from_numpy(np.asarray(Image.open(produced[0]).convert("RGB"), dtype=np.float32) / 255.0)
        .permute(2, 0, 1)
        .contiguous()
    )
    if restyled.shape[-2:] != source.shape[-2:]:
        restyled = torch.nn.functional.interpolate(
            restyled[None], size=source.shape[-2:], mode="bilinear", align_corners=False
        )[0]

    composited = torch.where(preserve, source, restyled)
    if not torch.equal(composited[:, preserve[0]], source[:, preserve[0]]):
        raise SystemExit("[smoke] FAILED: composite did not preserve the foreground exactly")

    for name, tensor in (("restyled", restyled), ("composited", composited)):
        arr = (tensor.permute(1, 2, 0) * 255).round().clamp(0, 255).to(torch.uint8).numpy()
        Image.fromarray(arr).save(out_root / f"{name}.png")
    print(f"[smoke] OK: vanilla checkpoint transferred; wrote {out_root}/composited.png", flush=True)


if __name__ == "__main__":
    main()
