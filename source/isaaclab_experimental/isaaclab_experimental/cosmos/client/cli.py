# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect an independently running Cosmos service, or move its weights between host and GPU memory.

python -m isaaclab_experimental.cosmos.client.cli status
python -m isaaclab_experimental.cosmos.client.cli offload
python -m isaaclab_experimental.cosmos.client.cli activate
"""

from __future__ import annotations

import argparse
import json
import sys

from isaaclab_experimental.cosmos._protocol import DEFAULT_ENDPOINT, DEFAULT_TIMEOUT_S, request


def main(args: list[str] | None = None) -> int:
    """Run the ``status``, ``offload``, or ``activate`` command."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        prog="python -m isaaclab_experimental.cosmos.client.cli",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description, timeout in (
        ("status", "Inspect the service.", 5.0),
        ("offload", "Move an idle service's weights to host memory.", DEFAULT_TIMEOUT_S),
        ("activate", "Move the service's weights back to its GPU.", DEFAULT_TIMEOUT_S),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument(
            "--endpoint",
            default=DEFAULT_ENDPOINT,
            help=f"unix:///path or tcp://host:port (default {DEFAULT_ENDPOINT}).",
        )
        # Moving the weights takes longer than a status reply.
        command.add_argument("--timeout", type=float, default=timeout, help="Connection and response timeout [s].")
    options = parser.parse_args(args)
    try:
        reply, _ = request(options.endpoint, {"op": options.command}, timeout=options.timeout)
        print(json.dumps(reply, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Cosmos: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
