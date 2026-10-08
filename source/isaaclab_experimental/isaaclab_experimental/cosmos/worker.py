# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load Cosmos once, then accept camera sessions independently of Isaac Lab."""

from __future__ import annotations

import argparse
import logging

from ._framework import CosmosInferenceModel
from .service import serve


def main(args: list[str] | None = None) -> None:
    """Run the worker with the optional Cosmos Framework interpreter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    options = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model = CosmosInferenceModel(options.checkpoint, options.device, use_compile=not options.no_compile)
    try:
        serve(model, host=options.host, port=options.port, warmup=options.warmup)
    finally:
        model.close()


if __name__ == "__main__":
    main()
