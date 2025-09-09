# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script is a utility to install dependencies mentioned in an extension.toml file of an extension.

The script takes in two arguments:

1. type: The type of dependencies to install. It can be one of the following: ['all', 'apt', 'rosdep'].
2. extensions_dir: The path to the directory beneath which we search for extensions.

The script will search for all extensions in the extensions_dir and then look for an extension.toml file in each
extension's config directory. If the extension.toml file exists, the script will look for the following keys in the
[isaac_lab_settings] section:

* **apt_deps**: A list of apt packages to install.
* **ros_ws**: The path to the ROS workspace in the extension. If the path is not absolute, the script assumes that
  the path is relative to the extension root and resolves it accordingly.

If the type is 'all', the script will install both apt and rosdep packages. If the type is 'apt', the script will only
install apt packages. If the type is 'rosdep', the script will only install rosdep packages.

For more information, please check the `documentation`_.

.. _documentation: https://isaac-sim.github.io/IsaacLab/source/setup/developer.html#extension-dependency-management
"""

import argparse
import os
import shutil
import toml
from subprocess import PIPE, STDOUT, Popen

# add argparse arguments
parser = argparse.ArgumentParser(description="A utility to install dependencies based on extension.toml files.")
parser.add_argument("type", type=str, choices=["all", "apt", "rosdep"], help="The type of packages to install.")
parser.add_argument("extensions_dir", type=str, help="The path to the directory containing extensions.")
parser.add_argument("--ros_distro", type=str, default="humble", help="The ROS distribution to use for rosdep.")


def install_apt_packages(paths: list[str]):
    """Installs apt packages listed in the extension.toml file for Isaac Lab extensions.

    For each path in the input list of paths, the function looks in ``{path}/config/extension.toml`` for
    the ``[isaac_lab_settings][apt_deps]`` key. It then attempts to install the packages listed in the
    value of the key. The function exits on failure to stop the build process from continuing despite missing
    dependencies.

    Args:
        paths: A list of paths to the extension's root.

    Raises:
        SystemError: If 'apt' is not a known command. This is a system error.
    """
    for path in paths:
        if shutil.which("apt"):
            # Check if the extension.toml file exists
            if not os.path.exists(f"{path}/config/extension.toml"):
                print(
                    "[WARN] During the installation of 'apt' dependencies, unable to find a"
                    f" valid file at: {path}/config/extension.toml."
                )
                continue
            # Load the extension.toml file and check for apt_deps
            with open(f"{path}/config/extension.toml") as fd:
                ext_toml = toml.load(fd)
                if "isaac_lab_settings" in ext_toml and "apt_deps" in ext_toml["isaac_lab_settings"]:
                    deps = ext_toml["isaac_lab_settings"]["apt_deps"]
                    print(f"[INFO] Installing the following apt packages: {deps}")
                    run_and_print(["apt-get", "update"])
                    run_and_print(["apt-get", "install", "-y"] + deps)
                else:
                    print(f"[INFO] No apt packages specified for the extension at: {path}")
        else:
            raise SystemError("Unable to find 'apt' command. Please ensure that 'apt' is installed on your system.")


def install_rosdep_packages(paths: list[str], ros_distro: str = "humble"):
    """Installs ROS dependencies listed in the extension.toml file for Isaac Lab extensions.

    For each path in the input list of paths, the function looks in ``{path}/config/extension.toml`` for
    the ``[isaac_lab_settings][ros_ws]`` key. It then attempts to install the ROS dependencies under the workspace
    listed in the value of the key. The function exits on failure to stop the build process from continuing despite
    missing dependencies.

    If the path to the ROS workspace is not absolute, the function assumes that the path is relative to the extension
    root and resolves it accordingly. The function also checks if the ROS workspace exists before proceeding with
    the installation of ROS dependencies. If the ROS workspace does not exist, the function raises an error.

    Args:
        path: A list of paths to the extension roots.
        ros_distro: The ROS distribution to use for rosdep. Default is 'humble'.

    Raises:
        FileNotFoundError: If a valid ROS workspace is not found while installing ROS dependencies.
        SystemError: If 'rosdep' is not a known command. This is raised if 'rosdep' is not installed on the system.
    """
    for path in paths:
        if shutil.which("rosdep"):
            # Check if the extension.toml file exists
            if not os.path.exists(f"{path}/config/extension.toml"):
                print(
                    "[WARN] During the installation of 'rosdep' dependencies, unable to find a"
                    f" valid file at: {path}/config/extension.toml."
                )
                continue
            # Load the extension.toml file and check for ros_ws
            with open(f"{path}/config/extension.toml") as fd:
                ext_toml = toml.load(fd)
                if "isaac_lab_settings" in ext_toml and "ros_ws" in ext_toml["isaac_lab_settings"]:
                    # resolve the path to the ROS workspace
                    ws_path = ext_toml["isaac_lab_settings"]["ros_ws"]
                    if not os.path.isabs(ws_path):
                        ws_path = os.path.join(path, ws_path)
                    # check if the workspace exists
                    if not os.path.exists(f"{ws_path}/src"):
                        raise FileNotFoundError(
                            "During the installation of 'rosdep' dependencies, unable to find a"
                            f" valid ROS workspace at: {ws_path}."
                        )
                    # install rosdep if not already installed
                    if not os.path.exists("/etc/ros/rosdep/sources.list.d/20-default.list"):
                        run_and_print(["rosdep", "init"])
                        run_and_print(["rosdep", "update", f"--rosdistro={ros_distro}"])
                    # install rosdep packages
                    run_and_print([
                        "rosdep",
                        "install",
                        "--from-paths",
                        f"{ws_path}/src",
                        "--ignore-src",
                        "-y",
                        f"--rosdistro={ros_distro}",
                    ])
                else:
                    print(f"[INFO] No rosdep packages specified for the extension at: {path}")
        else:
            raise SystemError(
                "Unable to find 'rosdep' command. Please ensure that 'rosdep' is installed on your system."
                "You can install it by running:\n\t sudo apt-get install python3-rosdep"
            )


def run_and_print(args: list[str]):
    """Runs a subprocess and prints the output to stdout.

    This function wraps Popen and prints the output to stdout in real-time.

    Args:
        args: A list of arguments to pass to Popen.
    """
    print(f'Running "{args}"')
    with Popen(args, stdout=PIPE, stderr=STDOUT, env=os.environ) as p:
        while p.poll() is None:
            text = p.stdout.read1().decode("utf-8")
            print(text, end="", flush=True)
        return_code = p.poll()
        if return_code != 0:
            raise RuntimeError(f'Subprocess with args: "{args}" failed. The returned error code was: {return_code}')


def main():
    # Parse the command line arguments
    args = parser.parse_args()
    # Get immediate children of args.extensions_dir
    extension_paths = [os.path.join(args.extensions_dir, x) for x in next(os.walk(args.extensions_dir))[1]]

    # Install dependencies based on the type
    if args.type == "all":
        install_apt_packages(extension_paths)
        install_rosdep_packages(extension_paths, args.ros_distro)
    elif args.type == "apt":
        install_apt_packages(extension_paths)
    elif args.type == "rosdep":
        install_rosdep_packages(extension_paths, args.ros_distro)
    else:
        raise ValueError(f"'Invalid dependency type: '{args.type}'. Available options: ['all', 'apt', 'rosdep'].")


if __name__ == "__main__":
    main()
