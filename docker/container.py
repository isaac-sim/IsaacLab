#!/usr/bin/env python3

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import shutil
import sys
from collections.abc import Container
from pathlib import Path

# __package__ is empty when this file runs as a script (./docker/container.py), where
# docker/ is itself on sys.path, and "docker" when the tests import it as docker.container.
# The relative form keeps docker.utils.* a single module object, so patches applied by the
# tests affect the same modules this CLI uses.
if __package__:
    from .utils import ContainerInterface, x11_utils
else:
    from utils import ContainerInterface, x11_utils


def reorder_profile_first(argv: list[str], commands: Container[str]) -> list[str]:
    """Accept ``<profile> <command>`` in addition to ``<command> <profile>``.

    Stepping a container through its lifecycle only changes the command, so keeping it last
    lets the previous line be reused by editing its tail: ``kitless build`` then
    ``kitless start``. Profile names are never command names, so the leading token
    identifies the order unambiguously.

    Args:
        argv: Command line arguments without the program name.
        commands: The subcommand names this CLI accepts.

    Returns:
        The arguments with the command first, unchanged if it already is.
    """
    if len(argv) >= 2 and not argv[0].startswith("-") and argv[0] not in commands and argv[1] in commands:
        return [argv[1], argv[0], *argv[2:]]
    return argv


def parse_cli_args() -> argparse.Namespace:
    """Parse command line arguments.

    This function creates a parser object and adds subparsers for each command. The function then parses the
    command line arguments and returns the parsed arguments.

    Returns:
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Utility for using Docker with Isaac Lab.")

    # We have to create separate parent parsers for common options to our subparsers
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "profile",
        nargs="?",
        default="base",
        help="Optional container profile specification. Examples: 'base', 'ros2', or 'kitless'.",
    )
    parent_parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help=(
            "Allows additional '.yaml' files to be passed to the docker compose command. These files will be merged"
            " with 'docker-compose.yaml' in their provided order."
        ),
    )
    parent_parser.add_argument(
        "--env-files",
        nargs="*",
        default=None,
        help=(
            "Allows additional '.env' files to be passed to the docker compose command. These files will be merged with"
            " the profile's default environment files in their provided order."
        ),
    )
    parent_parser.add_argument(
        "--suffix",
        nargs="?",
        default=None,
        help=(
            "Optional docker image and container name suffix.  Defaults to None, in which case, the docker name"
            " suffix is set to the empty string. A hyphen is inserted in between the profile and the suffix if"
            ' the suffix is a nonempty string.  For example, if "base" is passed to profile, and "custom" is'
            " passed to suffix, then the produced docker image and container will be named ``isaac-lab-base-custom``."
        ),
    )
    parent_parser.add_argument(
        "--info",
        action="store_true",
        help="Print the container interface information. This is useful for debugging purposes.",
    )

    # Actual command definition begins here
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "build",
        help="Build the docker image without creating the container.",
        parents=[parent_parser],
    )
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

    # parse the arguments to determine the command
    args = parser.parse_args(reorder_profile_first(sys.argv[1:], subparsers.choices))

    return args


def main(args: argparse.Namespace):
    """Main function for the Docker utility."""
    # check if docker is installed
    if not shutil.which("docker"):
        raise RuntimeError(
            "Docker is not installed! Please check the 'Docker Guide' for instruction: "
            "https://isaac-sim.github.io/IsaacLab/source/deployment/docker.html"
        )

    # creating container interface
    ci = ContainerInterface(
        context_dir=Path(__file__).resolve().parent,
        profile=args.profile,
        yamls=args.files,
        envs=args.env_files,
        suffix=args.suffix,
    )
    if args.info:
        print("[INFO] Printing container interface information...\n")
        ci.print_info()
        return

    print(f"[INFO] Using container profile: {ci.profile}")
    if args.command == "build":
        # check if x11 forwarding is enabled
        x11_outputs = x11_utils.x11_check(ci.statefile)
        # if x11 forwarding is enabled, add the x11 yaml and environment variables
        if x11_outputs is not None:
            (x11_yaml, x11_envar) = x11_outputs
            ci.add_yamls += x11_yaml
            ci.environ.update(x11_envar)
        # build the image
        ci.build()
    elif args.command == "start":
        # check if x11 forwarding is enabled
        x11_outputs = x11_utils.x11_check(ci.statefile)
        # if x11 forwarding is enabled, add the x11 yaml and environment variables
        if x11_outputs is not None:
            (x11_yaml, x11_envar) = x11_outputs
            ci.add_yamls += x11_yaml
            ci.environ.update(x11_envar)
        # start the container
        ci.start()
    elif args.command == "enter":
        # Entering a container that is not up used to be an error telling the user to run
        # 'start' themselves. Do it for them: 'stop' also deletes the .xauth file, so the
        # refresh below would fail on its own anyway.
        if not ci.is_container_running():
            print(f"[INFO] Container '{ci.container_name}' is not running. Starting it first...\n")
            x11_outputs = x11_utils.x11_check(ci.statefile)
            if x11_outputs is not None:
                (x11_yaml, x11_envar) = x11_outputs
                ci.add_yamls += x11_yaml
                ci.environ.update(x11_envar)
            ci.start()
        # refresh the x11 forwarding
        x11_utils.x11_refresh(ci.statefile)
        # enter the container
        ci.enter()
    elif args.command == "config":
        ci.config(args.output_yaml)
    elif args.command == "copy":
        ci.copy()
    elif args.command == "stop":
        # stop the container
        ci.stop()
        # cleanup the x11 forwarding
        x11_utils.x11_cleanup(ci.statefile)
    else:
        raise RuntimeError(f"Invalid command provided: {args.command}. Please check the help message.")


if __name__ == "__main__":
    args_cli = parse_cli_args()
    main(args_cli)
