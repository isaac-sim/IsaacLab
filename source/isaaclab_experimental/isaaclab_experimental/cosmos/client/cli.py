# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect an independently running Cosmos service."""

from __future__ import annotations

import argparse
import json
import sys

from isaaclab_experimental.cosmos._protocol import DEFAULT_ENDPOINT, request


def main(args: list[str] | None = None) -> int:
    """Run the ``isaaclab cosmos status`` command."""
    parser = argparse.ArgumentParser(description=__doc__, prog="isaaclab cosmos")
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="Inspect the service.")
    status.add_argument(
        "--endpoint", default=DEFAULT_ENDPOINT, help=f"unix:///path or tcp://host:port (default {DEFAULT_ENDPOINT})."
    )
    status.add_argument("--timeout", type=float, default=5.0, help="Connection and response timeout [s].")
    options = parser.parse_args(args)
    try:
        reply, _ = request(options.endpoint, {"op": "status"}, timeout=options.timeout)
        print(json.dumps(reply, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Cosmos: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
