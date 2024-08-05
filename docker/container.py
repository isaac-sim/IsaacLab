#!/usr/bin/env python3

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import shutil
from pathlib import Path

from utils import x11_utils
from utils.isaaclab_container_interface import IsaacLabContainerInterface


def main():
    parser = argparse.ArgumentParser(description="Utility for using Docker with Isaac Lab.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # We have to create separate parent parsers for common options to our subparsers
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("profile", nargs="?", default="base", help="Optional container profile specification.")
    parent_parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help=(
            "Allows additional .yaml files to be passed to the docker compose command. Files will be merged with"
            " docker-compose.yaml in the order in which they are provided."
        ),
    )
    parent_parser.add_argument(
        "--env-files",
        nargs="*",
        default=None,
        help=(
            "Allows additional .env files to be passed to the docker compose command. Files will be merged with"
            " .env.base in the order in which they are provided."
        ),
    )

    # Actual command definition begins here
    subparsers.add_parser(
        "start",
        help="Build the docker image and create the container in detached mode.",
        parents=[parent_parser],
    )
    subparsers.add_parser(
        "enter", help="Begin a new bash process within an existing Isaac Lab container.", parents=[parent_parser]
    )
    config = subparsers.add_parser(
        "config",
        help=(
            "Generate a docker-compose.yaml from the passed yamls, .envs, and either print to the terminal or create a"
            " yaml at output_yaml"
        ),
        parents=[parent_parser],
    )
    config.add_argument(
        "--output-yaml", nargs="?", default=None, help="Yaml file to write config output to. Defaults to None."
    )
    subparsers.add_parser(
        "copy", help="Copy build and logs artifacts from the container to the host machine.", parents=[parent_parser]
    )
    subparsers.add_parser("stop", help="Stop the docker container and remove it.", parents=[parent_parser])

    args = parser.parse_args()

    if not shutil.which("docker"):
        raise RuntimeError("Docker is not installed! Please check the 'Docker Guide' for instruction.")

    # Creating container interface
    ci = IsaacLabContainerInterface(
        context_dir=Path(__file__).resolve().parent, profile=args.profile, yamls=args.files, envs=args.env_files
    )

    print(f"[INFO] Using container profile: {ci.profile}")
    if args.command == "start":
        x11_outputs = x11_utils.x11_check(ci.statefile)
        if x11_outputs is not None:
            (x11_yaml, x11_envar) = x11_outputs
            ci.add_yamls += x11_yaml
            ci.environ.update(x11_envar)
        ci.start()
    elif args.command == "enter":
        x11_utils.x11_refresh(ci.statefile)
        ci.enter()
    elif args.command == "config":
        ci.config(args.output_yaml)
    elif args.command == "copy":
        ci.copy()
    elif args.command == "stop":
        ci.stop()
        x11_utils.x11_cleanup(ci.statefile)
    else:
        raise RuntimeError(f"Invalid command provided: {args.command}")


if __name__ == "__main__":
    main()
