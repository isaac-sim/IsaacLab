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
import toml
from subprocess import run

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to install dependencies based on an extension.toml")
parser.add_argument("type", type=str, choices=["all", "apt", "rosdep"], help="The type of packages to install")
parser.add_argument("extensions_dir", type=str, help="The path to the directory beneath which we search for extensions")


def install_apt_packages(paths):
    """
    A function which attempts to install apt packages for Isaac Lab extensions.
    For each path in arg paths, it looks in {extension_root}/config/extension.toml for [isaac_lab_settings][apt_deps]
    and then attempts to install them. Exits on failure to stop the build process
    from continuing despite missing dependencies.

    Args:
        paths: A list of paths to the extension root

    Raises:
        RuntimeError: If 'apt' is not a known command
    """
    for path in paths:
        if shutil.which("apt"):
            if not os.path.exists(f"{path}/config/extension.toml"):
                raise RuntimeError(
                    "During the installation of an IsaacSim extension's dependencies, an extension.toml was unable to"
                    " be found. All IsaacSim extensions must have a configuring .toml at"
                    " (extension_root)/config/extension.toml"
                )
            with open(f"{path}/config/extension.toml") as fd:
                ext_toml = toml.load(fd)
                if "isaac_lab_settings" in ext_toml and "apt_deps" in ext_toml["isaac_lab_settings"]:
                    deps = ext_toml["isaac_lab_settings"]["apt_deps"]
                    print(f"[INFO] Installing the following apt packages: {deps}")
                    run_and_print(["apt-get", "update"])
                    run_and_print(["apt-get", "install", "-y"] + deps)
                else:
                    print("[INFO] No apt packages to install")
        else:
            raise RuntimeError("Exiting because 'apt' is not a known command")


def install_rosdep_packages(paths):
    """A function which attempts to install rosdep packages for Isaac Lab extensions.
    For each path in arg paths, it looks in {extension_root}/config/extension.toml for [isaac_lab_settings][ros_ws]
    and then attempts to install all rosdeps under that workspace.
    Exits on failure to stop the build process from continuing despite missing dependencies.

    Args:
        path: A list of paths to the extension roots

    Raises:
        RuntimeError: If 'rosdep' is not a known command
    """
    for path in paths:
        if shutil.which("rosdep"):
            if not os.path.exists(f"{path}/config/extension.toml"):
                raise RuntimeError(
                    "During the installation of an IsaacSim extension's dependencies, an extension.toml was unable to"
                    " be found. All IsaacSim extensions must have a configuring .toml at"
                    " (extension_root)/config/extension.toml"
                )
            with open(f"{path}/config/extension.toml") as fd:
                ext_toml = toml.load(fd)
                if "isaac_lab_settings" in ext_toml and "ros_ws" in ext_toml["isaac_lab_settings"]:
                    ws_path = ext_toml["isaac_lab_settings"]["ros_ws"]
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


def run_and_print(args):
    """
    Runs a subprocess.run(args=args, capture_output=True, check=True),
    and prints the output

    Args:
        args: a list of arguments to be passed to subprocess.run()
    """
    completed_process = run(args=args, capture_output=True, check=True)
    print(f"{str(completed_process.stdout, encoding='utf-8')}")


def main():
    args = parser.parse_args()
    # Get immediate children of args.extensions_dir
    extension_paths = [os.path.join(args.extensions_dir, x) for x in next(os.walk(args.extensions_dir))[1]]
    if args.type == "all":
        install_apt_packages(extension_paths)
        install_rosdep_packages(extension_paths)
    elif args.type == "apt":
        install_apt_packages(extension_paths)
    elif args.type == "rosdep":
        install_rosdep_packages(extension_paths)
    else:
        raise RuntimeError(f"'{args.type}' type dependencies not installable")


if __name__ == "__main__":
    main()
