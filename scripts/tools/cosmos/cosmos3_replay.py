# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay recorded normalized controls through a Cosmos3 world-model service to generate a video.

The recorded controls are already model controls, so this drives ``Cosmos3Model`` directly, without an environment.
Start the service first, in the Cosmos Framework environment, with
``isaaclab-cosmos-world-model-server --checkpoint .pretrained_checkpoints/medical_pad_unfold/world_model``.

Usage:

.. code-block:: bash

    uv sync --extra video  # imageio, through moviepy
    uv run python scripts/tools/cosmos/cosmos3_replay.py --output outputs/cosmos3/replay
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from isaaclab_neural.models.cosmos3 import Cosmos3Cfg, Cosmos3Model

ASSETS = Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/contrib/medical_pad_unfold/assets"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--endpoint", default=Cosmos3Cfg().endpoint, help="Endpoint of the world-model service.")
parser.add_argument("--initial-image", type=Path, default=ASSETS / "validation_initial.png", help="Start image.")
parser.add_argument("--prompt", default="unfold the medical nursing pad", help="Task description.")
parser.add_argument(
    "--actions", type=Path, default=ASSETS / "replay_actions.json", help="normalized_unified_v1 JSON sidecar."
)
parser.add_argument("--output", type=Path, required=True, help="Directory for the generated video.")
parser.add_argument("--seed", type=int, default=1234, help="Sampling seed.")
args = parser.parse_args()


def main():
    actions = np.asarray(json.loads(args.actions.read_text())["action"], dtype=np.float32)
    model = Cosmos3Model(Cosmos3Cfg(endpoint=args.endpoint), num_envs=1, device="cpu")
    spec = model.spec()
    args.output.mkdir(parents=True)
    with imageio.get_writer(args.output / "generated.mp4", fps=spec.fps, macro_block_size=1) as writer:
        start = np.asarray(Image.open(args.initial_image).convert("RGB"))
        model.reset([0], torch.from_numpy(start)[None], [args.prompt], [args.seed])
        writer.append_data(start)
        for chunk in actions.reshape(-1, 1, spec.chunk_length, actions.shape[-1]):
            for frame in model.step_controls(chunk)[0].numpy():
                writer.append_data(frame)
    model.close()


if __name__ == "__main__":
    main()
