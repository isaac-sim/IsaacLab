# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run depth-to-image transfer as a modifier chain on CPU with an illustrative model."""

import argparse
from pathlib import Path

import numpy as np
import torch

from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierCfg, ModifierChain

from isaaclab_contrib.image_transfer import ImageTransferModifierCfg, depth_to_control


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
    # The same list goes into CameraCfg.modifiers under "distance_to_image_plane".
    chain = ModifierChain(
        [
            ModifierCfg(func=depth_to_control, params={"near": 1.0, "far": 9.0}),
            ImageTransferModifierCfg(backend=TintModelCfg()),
        ],
        "cpu",
    )
    try:
        depth = torch.linspace(1, 9, 96).view(1, 1, 96, 1).expand(1, 64, 96, 1)
        rgb = chain(depth)
        assert rgb[0, 0, 0].tolist() == [255, 153, 51]
        assert rgb[0, 0, -1].tolist() == [0, 0, 0]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, depth=depth.numpy(), rgb=rgb.numpy())
        print(f"Saved depth {tuple(depth.shape)} and RGB {tuple(rgb.shape)} to {args.output}")
    finally:
        chain.close()


if __name__ == "__main__":
    main()
