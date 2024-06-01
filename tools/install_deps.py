# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
A script with various methods of installing dependencies
defined in an extension.toml
"""

import argparse
import os
import shutil
import sys
import toml
from subprocess import SubprocessError, run

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to install dependencies based on an extension.toml")
parser.add_argument("type", type=str, choices=["all", "apt", "rosdep"], help="The type of packages to install")
parser.add_argument("path", type=str, help="The path to the extension which will have its deps installed")


def install_apt_packages(path):
    """
    A function which attempts to install apt packages for Isaac Lab extensions.
    It looks in {extension_root}/config/extension.toml for [isaaclab_settings][apt_deps]
    and then attempts to install them. Exits on failure to stop the build process
    from continuing despite missing dependencies.

    Args:
        path: A path to the extension root
    """
    try:
        if shutil.which("apt"):
            with open(f"{path}/config/extension.toml") as fd:
                ext_toml = toml.load(fd)
                if "isaaclab_settings" in ext_toml and "apt_deps" in ext_toml["isaaclab_settings"]:
                    deps = ext_toml["isaaclab_settings"]["apt_deps"]
                    print(f"[INFO] Installing the following apt packages: {deps}")
                    run_and_print(["apt-get", "update"])
                    run_and_print(["apt-get", "install", "-y"] + deps)
                else:
                    print("[INFO] No apt packages to install")
        else:
            raise RuntimeError("Exiting because 'apt' is not a known command")
    except SubprocessError as e:
        print(f"[ERROR]: {str(e.stderr, encoding='utf-8')}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR]: {e}")
        sys.exit(1)


def install_rosdep_packages(path):
    """
    A function which attempts to install rosdep packages for Isaac Lab extensions.
    It looks in {extension_root}/config/extension.toml for [isaaclab_settings][ros_ws]
    and then attempts to install all rosdeps under that workspace.
    Exits on failure to stop the build process from continuing despite missing dependencies.

    Args:
        path: A path to the extension root
    """
    try:
        if shutil.which("rosdep"):
            with open(f"{path}/config/extension.toml") as fd:
                ext_toml = toml.load(fd)
                if "isaaclab_settings" in ext_toml and "ros_ws" in ext_toml["isaaclab_settings"]:
                    ws_path = ext_toml["isaaclab_settings"]["ros_ws"]
                    if not os.path.exists("/etc/ros/rosdep/sources.list.d/20-default.list"):
                        run_and_print(["rosdep", "init"])
                        run_and_print(["rosdep", "update", "--rosdistro=humble"])
                    run_and_print([
                        "rosdep",
                        "install",
                        "--from-paths",
                        f"{path}/{ws_path}/src",
                        "--ignore-src",
                        "-y",
                        "--rosdistro=humble",
                    ])
                else:
                    print("[INFO] No rosdep packages to install")
        else:
            raise RuntimeError("Exiting because 'rosdep' is not a known command")
    except SubprocessError as e:
        print(f"[ERROR]: {str(e.stderr, encoding='utf-8')}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR]: {e}")
        sys.exit(1)


def run_and_print(args):
    """
    Runs a subprocess.run(args=args, capture_output=True, check=True),
    and prints the output
    """
    completed_process = run(args=args, capture_output=True, check=True)
    print(f"{str(completed_process.stdout, encoding='utf-8')}")


def main():
    args = parser.parse_args()
    if args.type == "all":
        install_apt_packages(args.path)
        install_rosdep_packages(args.path)
    elif args.type == "apt":
        install_apt_packages(args.path)
    elif args.type == "rosdep":
        install_rosdep_packages(args.path)
    else:
        print(f"[ERROR] '{args.type}' type dependencies not installable")
        sys.exit(1)


if __name__ == "__main__":
    main()
