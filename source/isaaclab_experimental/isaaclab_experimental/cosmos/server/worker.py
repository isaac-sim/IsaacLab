# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load Cosmos once, then accept camera sessions independently of Isaac Lab."""

from __future__ import annotations

import argparse
import logging
import os

from isaaclab_experimental.cosmos.server._framework import CosmosInferenceModel
from isaaclab_experimental.cosmos.server.service import serve


def main(args: list[str] | None = None) -> None:
    """Run the streaming service in the Cosmos Framework environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument(
        "--max-episode-frames",
        type=int,
        default=0,
        help="Optional episode cap in frames, 1 + 4*k with k >= 1; default 0 leaves the budget to each session.",
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
    serve(
        lambda: CosmosInferenceModel(
            options.checkpoint,
            options.device,
            use_compile=not options.no_compile,
            max_episode_frames=options.max_episode_frames or None,
            kv_window=options.kv_window,
            attention_sink=options.attention_sink,
        ),
        host=options.host,
        port=options.port,
        warmup=options.warmup,
    )


if __name__ == "__main__":
    main()
