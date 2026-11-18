# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay recorded normalized controls through Cosmos3 to generate a video.

The recorded controls are already model controls, so this drives ``Cosmos3Model`` directly, without an environment.

Usage:

.. code-block:: bash

    uv run --extra cosmos3 python scripts/tools/cosmos/cosmos3_replay.py \
        --checkpoint .pretrained_checkpoints/medical_pad_unfold/world_model --output outputs/cosmos3/replay
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from isaaclab_neural.models.cosmos3 import Cosmos3Cfg, Cosmos3Model

ASSETS = Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/contrib/medical_pad_unfold/assets"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", type=Path, required=True, help="Local DCP root or complete HF export.")
parser.add_argument("--config-file", type=Path, help="Saved model YAML; required for a DCP checkpoint.")
parser.add_argument("--initial-image", type=Path, default=ASSETS / "validation_initial.png", help="Start image.")
parser.add_argument("--prompt", default="unfold the medical nursing pad", help="Task description.")
parser.add_argument(
    "--actions", type=Path, default=ASSETS / "replay_actions.json", help="normalized_unified_v1 JSON sidecar."
)
parser.add_argument("--output", type=Path, required=True, help="Directory for the generated video.")
parser.add_argument("--mode", choices=("window", "session"), default="window", help="Cosmos3 inference mode.")
parser.add_argument("--seed", type=int, default=1234, help="Sampling seed.")
args = parser.parse_args()


def main():
    actions = np.asarray(json.loads(args.actions.read_text())["action"], dtype=np.float32)
    cfg = Cosmos3Cfg(
        checkpoint_path=str(args.checkpoint),
        config_file=None if args.config_file is None else str(args.config_file),
        output_dir=str(args.output),
        mode=args.mode,
    )
    args.output.mkdir(parents=True)
    model = Cosmos3Model(cfg, num_envs=1)
    with imageio.get_writer(args.output / "generated.mp4", fps=cfg.fps, macro_block_size=1) as writer:
        writer.append_data(model.reset(np.array([0]), [args.initial_image], [args.prompt], np.array([args.seed]))[0])
        for chunk in actions.reshape(-1, 1, cfg.chunk_size, 59):
            for frame in model.step_controls(chunk)[0]:
                writer.append_data(frame)
    model.close()


if __name__ == "__main__":
    main()
