# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load Cosmos once, then accept camera sessions independently of Isaac Lab."""

from __future__ import annotations

import argparse
import logging
import os

from isaaclab_experimental.cosmos._protocol import DEFAULT_ENDPOINT, DEFAULT_MAX_EPISODE_FRAMES
from isaaclab_experimental.cosmos.server._framework import CosmosInferenceModel
from isaaclab_experimental.cosmos.server.service import serve


def main(args: list[str] | None = None) -> None:
    """Run the streaming service in the Cosmos Framework environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"unix:///path (same machine, owner-only) or tcp://host:port (default {DEFAULT_ENDPOINT}).",
    )
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument(
        "--max-episode-frames",
        type=int,
        default=DEFAULT_MAX_EPISODE_FRAMES,
        help="Longest episode in frames, 1 + 4*k with k >= 1 (default: the trained horizon, 201); 0 removes the cap.",
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=1,
        help="Most camera views (environments) one session may batch; each needs GPU memory for its history.",
    )
    parser.add_argument(
        "--kv-window",
        type=int,
        default=30,
        help="Generation history in latent frames (Sim-Transfer recipe: 30); shorter is faster but remembers less.",
    )
    parser.add_argument(
        "--attention-sink", type=int, default=3, help="Earliest latent frames always kept in the history window."
    )
    options = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ["COSMOS_TRAINING"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
    model = CosmosInferenceModel(
        options.checkpoint,
        options.device,
        use_compile=not options.no_compile,
        max_episode_frames=options.max_episode_frames or None,
        max_views=options.max_views,
        kv_window=options.kv_window,
        attention_sink=options.attention_sink,
    )
    try:
        serve(model, options.endpoint, warmup=options.warmup)
    finally:
        model.close()


if __name__ == "__main__":
    main()
