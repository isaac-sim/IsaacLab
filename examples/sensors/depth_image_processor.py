# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run depth-to-image transfer as camera tensor processors on CPU with an illustrative model."""

import argparse
from pathlib import Path

import numpy as np
import torch
import warp as wp
from isaaclab_experimental.image_transfer import FunctionProcessorCfg, ImageTransferProcessorCfg, depth_to_control

from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass
from isaaclab.utils.warp import ProxyArray


class TintStream:
    """Apply a fixed tint; this example does not run a learned model."""

    def step(self, controls, reset_rows, seeds):
        tint = torch.tensor([1.0, 0.6, 0.2])
        return [pixels.float().mul(tint).round().to(torch.uint8) for pixels in controls]

    def close(self):
        pass


class TintModel:
    def __init__(self, cfg):
        pass

    def open_stream(self, num_views, seeds):
        return TintStream()

    def close(self):
        pass


@configclass
class TintModelCfg(BackendCfg):
    class_type: type = TintModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file.")
    # The same list goes into CameraCfg.tensor_processors; the camera renders "distance_to_image_plane".
    cfgs = [
        FunctionProcessorCfg(
            function=depth_to_control,
            params={"near": 1.0, "far": 9.0},
            inputs=["distance_to_image_plane"],
            outputs=["control"],
        ),
        ImageTransferProcessorCfg(backend=TintModelCfg()),
    ]
    depth = torch.linspace(1, 9, 96).view(1, 1, 96, 1).expand(1, 64, 96, 1).contiguous()
    buffers = {"distance_to_image_plane": ProxyArray(wp.from_torch(depth))}
    processors = []
    try:
        # Construct and run in order, as a camera does: each processor reads the outputs of the earlier ones.
        for cfg in cfgs:
            processors.append(cfg.func(cfg, buffers))
            buffers.update(processors[-1](buffers))
        rgb = buffers["rgb"].torch
        assert rgb[0, 0, 0].tolist() == [255, 153, 51]
        assert rgb[0, 0, -1].tolist() == [0, 0, 0]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, depth=depth.numpy(), rgb=rgb.numpy())
        print(f"Saved depth {tuple(depth.shape)} and RGB {tuple(rgb.shape)} to {args.output}")
    finally:
        for processor in processors:
            processor.close()


if __name__ == "__main__":
    main()
