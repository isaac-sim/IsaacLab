#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(
    description=(
        "Utility to launch IsaacSim with the eval_sim extension loaded. Arguments not listed                           "
        "      here will be passed directly as arguments to isaac_sim.sh"
    )
)
parser.add_argument(
    "-l",
    "--livestream",
    action="store_true",
    help="Stream the display with the Omniverse Livestreaming Client, rather than display locally.",
)
parser.add_argument("-de", "--debug", action="store_true", help="Load the Omni VSCode debug extension.")


def get_script_dir():
    """Returns the directory where the script is located."""
    return Path(__file__).resolve().parent


def resolve_args(args):
    resolved_args = []
    # Add nucleus asset roots to prevent long timeout
    resolved_args += [
        "--/persistent/isaac/asset_root/default='http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1",
        "--/persistent/isaac/asset_root/nvidia='http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1",
    ]
    if args.debug:
        resolved_args += ["--enable", "omni.kit.debug.vscode"]
    return resolved_args


def get_environment_path(env_var, default_path):
    """Check if the environment variable is set and return the path, otherwise try to deduce it."""
    env_path = os.environ.get(env_var)
    if not env_path:
        print(f"[INFO] ${env_var} is not defined.\n       Looking in parent directory.")
        if default_path.exists():
            env_path = str(default_path.resolve())
            print(f"[INFO] Found {env_var} at: {env_path}")
        else:
            print(f"[ERROR] Couldn't find {env_var}.")
            sys.exit(1)
    return env_path


def choose_simulator_exe(livestream_flag, isaacsim_path):
    """Determine the correct executable based on the livestreaming flag."""
    if livestream_flag:
        isaacsim_exe = f"{isaacsim_path}/isaac-sim.headless.native.sh"
        print("[WARNING] IsaacSim will attempt to display via livestreaming.")
        time.sleep(0.5)
    else:
        isaacsim_exe = f"{isaacsim_path}/isaac-sim.sh"
        print(
            "[WARNING] IsaacSim will attempt to display locally. If your environment is non-graphical, IsaacSim will"
            " crash."
        )
        print("[WARNING] If your system doesn't have a local display or XForwarding, pass -l to enable livestreaming.")
        time.sleep(0.5)
    return isaacsim_exe


def construct_command(isaacsim_exe, isaaclab_path, evalsim_exts_dir, resolved_args, unknown_args):
    """Construct the command to run IsaacSim with the appropriate parameters."""
    print(f"[INFO] Passing unknown args {unknown_args} to isaac_sim.sh")
    command = (
        [
            isaacsim_exe,
            "--ext-folder",
            f"{isaaclab_path}/source",
            "--enable",
            "isaaclab",
            "--ext-path",
            str(evalsim_exts_dir),
            "--enable",
            "rai.eval_sim",
        ]
        + resolved_args
        + unknown_args
    )
    return command


def main():
    script_dir = get_script_dir()
    evalsim_exts_dir = script_dir.parent
    known_args, unknown_args = parser.parse_known_args()

    isaaclab_path = get_environment_path("ISAACLAB_PATH", script_dir.parents[3])
    isaacsim_path = get_environment_path("ISAACSIM_PATH", Path(isaaclab_path) / "_isaac_sim")

    envvar_livestream = os.getenv("HEADLESS") == "1" or os.getenv("LIVESTREAM") == "1"
    livestream_flag = known_args.livestream or envvar_livestream or "DISPLAY" not in os.environ

    isaacsim_exe = choose_simulator_exe(livestream_flag, isaacsim_path)
    resolved_args = resolve_args(known_args)
    command = construct_command(isaacsim_exe, isaaclab_path, evalsim_exts_dir, resolved_args, unknown_args)
    try:
        # Execute the evalsim launch command
        subprocess.run(command)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise RuntimeError(f"Unexpected error occurred during EvalSim execution with error: {e}")
    finally:
        print("[INFO] Closed EvalSim")


if __name__ == "__main__":
    main()
