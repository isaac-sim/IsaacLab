# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Start and manage a resident Cosmos service in its own Python environment."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from ._protocol import parse_endpoint, request

DEFAULT_ENDPOINT = "tcp://127.0.0.1:5555"


def main(args: list[str] | None = None) -> int:
    """Run the ``isaaclab cosmos start|status|stop`` command."""
    parser = argparse.ArgumentParser(description=__doc__, prog="isaaclab cosmos")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="Load Cosmos and serve cameras in the foreground.")
    start.add_argument(
        "--framework-root", type=Path, required=True, help="Cosmos Framework checkout with its own .venv."
    )
    start.add_argument("--checkpoint", required=True, help="Sim-Transfer checkpoint directory or Framework name.")
    start.add_argument("--python", type=Path, help="Override the Framework's .venv/bin/python interpreter.")
    start.add_argument("--device", default="cuda:0", help="CUDA device in the Cosmos process.")
    start.add_argument("--no-compile", action="store_true", help="Use eager inference instead of compiled CUDA graphs.")
    start.add_argument(
        "--warmup", action="store_true", help="Generate a disposable session before reporting readiness."
    )
    start.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    for command in ("status", "stop"):
        control = commands.add_parser(
            command, help="Inspect the service." if command == "status" else "Stop the service."
        )
        control.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
        control.add_argument("--timeout", type=float, default=5.0, help="Connection and response timeout [s].")
    options = parser.parse_args(args)
    try:
        if options.command == "start":
            return _start(options)
        reply, _ = request(
            options.endpoint,
            {"op": "status" if options.command == "status" else "shutdown"},
            timeout=options.timeout,
        )
        print(json.dumps(reply, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Cosmos: {error}", file=sys.stderr)
        return 1


def _start(options: argparse.Namespace) -> int:
    host, port = parse_endpoint(options.endpoint)
    framework_root = options.framework_root.expanduser().absolute()
    if not (framework_root / "cosmos_framework").is_dir():
        raise ValueError(f"Cosmos Framework package not found in {framework_root}.")
    # Resolving this symlink would lose the selected virtual environment.
    python = (options.python or framework_root / ".venv/bin/python").expanduser().absolute()
    if not python.is_file():
        raise ValueError(f"Cosmos interpreter not found: {python}. Set up the Framework environment or pass --python.")
    checkpoint_path = Path(options.checkpoint).expanduser()
    checkpoint = str(checkpoint_path.absolute()) if checkpoint_path.exists() else options.checkpoint

    import isaaclab

    environment = os.environ.copy()
    source_paths = [str(Path(__file__).parents[2]), str(Path(isaaclab.__file__).parents[1]), str(framework_root)]
    environment["PYTHONPATH"] = os.pathsep.join([*source_paths, environment.get("PYTHONPATH", "")])
    environment.pop("PYTHONHOME", None)
    environment.pop("VIRTUAL_ENV", None)
    environment["COSMOS_TRAINING"] = "0"
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
    command = [
        str(python),
        "-m",
        "isaaclab_experimental.cosmos.worker",
        "--checkpoint",
        checkpoint,
        "--device",
        options.device,
        "--host",
        host,
        "--port",
        str(port),
    ]
    if options.no_compile:
        command.append("--no-compile")
    if options.warmup:
        command.append("--warmup")
    process = subprocess.Popen(command, cwd=framework_root, env=environment, start_new_session=True)
    try:
        return process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
