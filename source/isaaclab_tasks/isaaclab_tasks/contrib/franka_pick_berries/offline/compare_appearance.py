# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a frozen before/after SH comparison without changing camera or sampling.

The baseline is an exported single-product USD scene. Only the SH coefficient
attribute is replaced when --asset is given. Use separate processes per image.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

from pxr import Usd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--asset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=40)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    stage = Usd.Stage.Open(Usd.Stage.Open(str(args.scene)).Flatten())
    if args.asset:
        asset = Usd.Stage.Open(str(args.asset))
        target = stage.GetPrimAtPath("/World/Berry/Gaussians")
        source = asset.GetPrimAtPath("/Berry/Gaussians")
        name = "radiance:sphericalHarmonicsCoefficients"
        if len(source.GetAttribute(name).Get()) != len(target.GetAttribute(name).Get()):
            raise ValueError("Before/after comparison requires identical Gaussian counts")
        target.GetAttribute(name).Set(source.GetAttribute(name).Get())
    path = args.output / "scene.usdc"
    stage.GetRootLayer().Export(str(path))
    products = {str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName() == "RenderProduct"}
    if len(products) != 1:
        raise ValueError("Expected one render product")
    os.environ.setdefault("OVRTX_SKIP_USD_CHECK", "1")
    import ovrtx

    renderer = ovrtx.Renderer(config=ovrtx.RendererConfig(sync_mode=True, log_level="error"))
    try:
        renderer.open_usd(str(path))
        for _ in range(args.frames):
            result = renderer.step(render_products=products, delta_time=1 / 30)
        frame = next(iter(result.values())).frames[-1]
        for var in frame.render_vars.values():
            if var.source_name == "LdrColor":
                with var.map(device=ovrtx.Device.CPU) as mapping:
                    Image.fromarray(np.from_dlpack(mapping).copy()).save(args.output / "appearance.png")
                mapping = None
        print(f"Rendered {args.output}; progression={frame.progression}, converged={frame.converged}", flush=True)
    finally:
        result = frame = None
        renderer.destroy()


if __name__ == "__main__":
    main()
