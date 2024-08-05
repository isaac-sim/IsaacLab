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

    # We have to create a separate parent parser for common options to our subparsers
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("profile", nargs="?", default="base", help="Optional container profile specification.")

    subparsers.add_parser(
        "start", help="Build the docker image and create the container in detached mode.", parents=[parent_parser]
    )
    subparsers.add_parser(
        "enter", help="Begin a new bash process within an existing Isaac Lab container.", parents=[parent_parser]
    )
    subparsers.add_parser(
        "copy", help="Copy build and logs artifacts from the container to the host machine.", parents=[parent_parser]
    )
    subparsers.add_parser("stop", help="Stop the docker container and remove it.", parents=[parent_parser])

    args = parser.parse_args()

    if not shutil.which("docker"):
        raise RuntimeError("Docker is not installed! Please check the 'Docker Guide' for instruction.")

    # Creating container interface
    ci = IsaacLabContainerInterface(context_dir=Path(__file__).resolve().parent, profile=args.profile)

    print(f"[INFO] Using container profile: {ci.profile}")
    if args.command == "start":
        print(f"[INFO] Building the docker image and starting the container {ci.container_name} in the background...")
        x11_outputs = x11_utils.x11_check(ci.statefile)
        if x11_outputs is not None:
            (x11_yaml, x11_envar) = x11_outputs
            ci.add_yamls += x11_yaml
            ci.environ.update(x11_envar)
        ci.start()
    elif args.command == "enter":
        print(f"[INFO] Entering the existing {ci.container_name} container in a bash session...")
        x11_utils.x11_refresh(ci.statefile)
        ci.enter()
    elif args.command == "copy":
        print(f"[INFO] Copying artifacts from the 'isaac-lab-{ci.container_name}' container...")
        ci.copy()
        print("\n[INFO] Finished copying the artifacts from the container.")
    elif args.command == "stop":
        print(f"[INFO] Stopping the launched docker container {ci.container_name}...")
        ci.stop()
        x11_utils.x11_cleanup(ci.statefile)
    else:
        raise RuntimeError(f"Invalid command provided: {args.command}")


if __name__ == "__main__":
    main()
