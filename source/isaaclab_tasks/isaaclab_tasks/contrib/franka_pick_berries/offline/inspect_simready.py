# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render repeatable turntable views of a berry asset with the Newton RTX viewer."""

import argparse
import math
from pathlib import Path

import newton
import numpy as np
import warp as wp
from newton.viewer import ViewerRTX
from PIL import Image, ImageDraw

from pxr import UsdGeom, Vt

from isaaclab_tasks.contrib.franka_pick_berries.assets.usd_asset import load_berry
from isaaclab_tasks.contrib.franka_pick_berries.rendering.settings import require_live_gaussian_renderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["full", "skin", "scan"], default="full")
    parser.add_argument("--frames", type=int, default=24)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    require_live_gaussian_renderer()
    wp.init()
    wp.set_device("cuda:0")
    source, asset, _, _ = load_berry(args.asset)
    model = newton.ModelBuilder().finalize()
    viewer = ViewerRTX(width=768, height=768, headless=True, environment="studio", async_rendering=False)
    viewer.set_model(model)
    UsdGeom.Xform.Define(viewer.stage, "/World/Berry").GetPrim().GetReferences().AddReference(str(args.asset.resolve()))
    if args.variant != "full":
        ids = asset["source_ids"]
        mask = ids != -1 if args.variant == "skin" else ids >= 0
        prim = viewer.stage.GetPrimAtPath("/World/Berry/Gaussians")
        for name, key, ctor in (
            ("positions", "xyz", Vt.Vec3fArray),
            ("scales", "scales", Vt.Vec3fArray),
            ("orientations", "rotations", Vt.QuatfArray),
            ("opacities", "alpha", Vt.FloatArray),
            ("radiance:sphericalHarmonicsCoefficients", "sh", Vt.Vec3fArray),
        ):
            values = asset[key][mask]
            if key == "sh":
                values = values.reshape(-1, 3)
            attr = prim.GetAttribute(name)
            value = ctor.FromNumpy(np.ascontiguousarray(values))
            attr.Set(value)
            for time in attr.GetTimeSamples():
                attr.Set(value, time)
        original = source.GetPrimAtPath("/Berry/Gaussians")
        for name, ctor in (
            ("primvars:squishyShQuaternion", Vt.Vec4fArray),
            ("primvars:interior", Vt.IntArray),
            ("primvars:repairedSkin", Vt.IntArray),
        ):
            if original.HasAttribute(name):
                value = ctor.FromNumpy(np.ascontiguousarray(np.asarray(original.GetAttribute(name).Get())[mask]))
                attr = prim.GetAttribute(name)
                attr.Set(value)
                for time in attr.GetTimeSamples():
                    attr.Set(value, time)
    center = (np.quantile(asset["xyz"], 0.01, axis=0) + np.quantile(asset["xyz"], 0.99, axis=0)) / 2
    radius = np.max(np.ptp(asset["xyz"], axis=0)) * 1.9
    views = [(a, 20) for a in (0, 90, 180, 270)] + [(0, 80), (0, -80)]
    sheet = Image.new("RGB", (768 * 3, 800 * 2), "#202020")
    try:
        for index, (azimuth, elevation) in enumerate(views):
            a, e = np.radians([azimuth, elevation])
            eye = center + radius * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
            direction = center - eye
            viewer.set_camera(
                wp.vec3(*eye),
                math.degrees(math.asin(direction[2] / radius)),
                math.degrees(math.atan2(direction[1], direction[0])),
            )
            viewer.camera.near = 0.0001
            viewer.camera.pivot = type(viewer.camera.pos)(*center)
            for frame in range(args.frames):
                viewer.begin_frame((index * args.frames + frame) / 30)
                viewer.log_state(model.state())
                viewer.end_frame()
            from ovrtx import Device

            for product in viewer._render_products.values():
                for var in product.frames[-1].render_vars.values():
                    if var.source_name == "LdrColor":
                        with var.map(device=Device.CPU) as mapping:
                            image = Image.fromarray(np.from_dlpack(mapping).copy()).convert("RGB")
                        mapping = None
                        image.save(args.output / f"view-{index}.png")
                        sheet.paste(image, (index % 3 * 768, index // 3 * 800))
            ImageDraw.Draw(sheet).text(
                (index % 3 * 768 + 12, index // 3 * 800 + 775),
                f"{args.variant} | azimuth {azimuth}, elevation {elevation}",
                fill="white",
            )
            print(f"Rendered {args.variant} view {index}", flush=True)
        sheet.save(args.output / "turntable.jpg")
    finally:
        renderer = viewer._rtx
        viewer.close()
        if renderer is not None:
            renderer.destroy()


if __name__ == "__main__":
    main()
