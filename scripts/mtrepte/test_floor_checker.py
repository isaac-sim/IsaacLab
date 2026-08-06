#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Minimal Newton GL floor-only test for quickly iterating on checker tile size.

Usage:
    uv run python scripts/mtrepte/test_floor_checker.py --checker-scale 10 --out /tmp/floor.png
    uv run python scripts/mtrepte/test_floor_checker.py --checker-scale 1000 --out /tmp/floor.png
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--checker-scale", type=float, default=None, help="Override checker_scale in GLSL shader")
parser.add_argument("--out", type=str, default="/tmp/floor_checker.png", help="Output PNG path")
parser.add_argument("--frames", type=int, default=5, help="Frames to render before saving")
parser.add_argument("--streaming", action="store_true", default=False, help="Enable streaming view")
args = parser.parse_args()

# -- Patch shader BEFORE importing any Newton GL code --
if args.checker_scale is not None:
    import newton._src.viewer.gl.shaders as _shaders

    _OLD = "    float checker_scale = "
    src = _shaders.shape_fragment_shader
    # Find the line and replace it
    lines = src.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("float checker_scale = "):
            lines[i] = f"    float checker_scale = {args.checker_scale:.1f};"
            break
    _shaders.shape_fragment_shader = "\n".join(lines)
    print(f"[floor_test] Patched checker_scale = {args.checker_scale}")

import newton
import newton.viewer
import warp as wp

# Build minimal model: just one floor plane
builder = newton.ModelBuilder()
# Add ground plane (width=0 → infinite, displayed as 1000m in viewer)
builder.add_shape_plane(body=-1, width=0.0, length=0.0)

model = builder.finalize(device="cuda")
state = model.state()

# Create GL viewer headlessly
viewer = newton.viewer.ViewerGL(width=1280, height=720, headless=True)
viewer.set_model(model)
viewer.set_camera(wp.vec3(4.0, 0.0, 2.5), pitch=-20.0, yaw=0.0)

print(f"[floor_test] Rendering {args.frames} frames...")

for i in range(args.frames):
    viewer.begin_frame(float(i) / 30.0)
    viewer.log_state(state)
    viewer.end_frame()

# Capture frame
frame = viewer.get_frame()
pixels = frame.numpy()

# Save to PNG
try:
    from PIL import Image

    img = Image.fromarray(pixels)
    img.save(args.out)
    print(f"[floor_test] Saved {pixels.shape} to {args.out}")
except Exception as e:
    print(f"[floor_test] PIL save failed: {e}, trying cv2...")
    import cv2

    cv2.imwrite(args.out, cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    print(f"[floor_test] Saved via cv2 to {args.out}")

viewer.close()
print("[floor_test] Done.")
