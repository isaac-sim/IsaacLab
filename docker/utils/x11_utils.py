# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for managing X11 forwarding in the docker container."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .state_file import StateFile


# This method of x11 enabling forwarding was inspired by osrf/rocker
# https://github.com/osrf/rocker
def configure_x11(statefile: StateFile) -> dict[str, str]:
    """Configure X11 forwarding by creating and managing a temporary .xauth file.

    If xauth is not installed, the function prints an error message and exits. The message
    instructs the user to install xauth with 'apt install xauth'.

    If the .xauth file does not exist, the function creates it and configures it with the necessary
    xauth cookie.

    Args:
        statefile: An instance of the configuration file class.

    Returns:
        A dictionary with two key-value pairs:

        - "__ISAACLAB_TMP_XAUTH": The path to the temporary .xauth file.
        - "__ISAACLAB_TMP_DIR": The path to the directory where the temporary .xauth file is stored.

    """
    # check if xauth is installed
    if not shutil.which("xauth"):
        print("[INFO] xauth is not installed.")
        print("[INFO] Please install it with 'apt install xauth'")
        exit(1)

    # set the namespace to X11 for the statefile
    statefile.namespace = "X11"
    # load the value of the temporary xauth file
    tmp_xauth_value = statefile.get_variable("__ISAACLAB_TMP_XAUTH")

    if tmp_xauth_value is None or not Path(tmp_xauth_value).exists():
        # create a temporary directory to store the .xauth file
        tmp_dir = subprocess.run(["mktemp", "-d"], capture_output=True, text=True, check=True).stdout.strip()
        # create the .xauth file
        tmp_xauth_value = create_x11_tmpfile(tmpdir=Path(tmp_dir))
        # set the statefile variable
        statefile.set_variable("__ISAACLAB_TMP_XAUTH", str(tmp_xauth_value))
    else:
        tmp_dir = Path(tmp_xauth_value).parent

    return {"__ISAACLAB_TMP_XAUTH": str(tmp_xauth_value), "__ISAACLAB_TMP_DIR": str(tmp_dir)}


def x11_check(statefile: StateFile) -> tuple[list[str], dict[str, str]] | None:
    """Check and configure X11 forwarding based on user input and existing state.

    This function checks if X11 forwarding is enabled in the configuration file. If it is not configured,
    the function prompts the user to enable or disable X11 forwarding. If X11 forwarding is enabled, the function
    configures X11 forwarding by creating a temporary .xauth file.

    Args:
        statefile: An instance of the configuration file class.

    Returns:
        If X11 forwarding is enabled, the function returns a tuple containing the following:

        - A list containing the x11.yaml file configuration option for docker-compose.
        - A dictionary containing the environment variables for the container.

        If X11 forwarding is disabled, the function returns None.
    """
    # set the namespace to X11 for the statefile
    statefile.namespace = "X11"
    # check if X11 forwarding is enabled
    is_x11_forwarding_enabled = statefile.get_variable("X11_FORWARDING_ENABLED")

    if is_x11_forwarding_enabled is None:
        print("[INFO] X11 forwarding from the Isaac Lab container is disabled by default.")
        print(
            "[INFO] It will fail if there is no display, or this script is being run via ssh without proper"
            " configuration."
        )
        x11_answer = input("Would you like to enable it? (y/N) ")

        # parse the user's input
        if x11_answer.lower() == "y":
            is_x11_forwarding_enabled = "1"
            print("[INFO] X11 forwarding is enabled from the container.")
        else:
            is_x11_forwarding_enabled = "0"
            print("[INFO] X11 forwarding is disabled from the container.")

        # remember the user's choice and set the statefile variable
        statefile.set_variable("X11_FORWARDING_ENABLED", is_x11_forwarding_enabled)
    else:
        # print the current configuration
        print(f"[INFO] X11 Forwarding is configured as '{is_x11_forwarding_enabled}' in '.container.cfg'.")

        # print help message to enable/disable X11 forwarding
        if is_x11_forwarding_enabled == "1":
            print("\tTo disable X11 forwarding, set 'X11_FORWARDING_ENABLED=0' in '.container.cfg'.")
        else:
            print("\tTo enable X11 forwarding, set 'X11_FORWARDING_ENABLED=1' in '.container.cfg'.")

    if is_x11_forwarding_enabled == "1":
        x11_envars = configure_x11(statefile)
        # If X11 forwarding is enabled, return the proper args to
        # compose the x11.yaml file. Else, return an empty string.
        return ["--file", "x11.yaml"], x11_envars

    return None


def x11_cleanup(statefile: StateFile):
    """Clean up the temporary .xauth file used for X11 forwarding.

    If the .xauth file exists, this function deletes it and remove the corresponding state variable.

    Args:
        statefile: An instance of the configuration file class.
    """
    # set the namespace to X11 for the statefile
    statefile.namespace = "X11"

    # load the value of the temporary xauth file
    tmp_xauth_value = statefile.get_variable("__ISAACLAB_TMP_XAUTH")

    # if the file exists, delete it and remove the state variable
    if tmp_xauth_value is not None and Path(tmp_xauth_value).exists():
        print(f"[INFO] Removing temporary Isaac Lab '.xauth' file: {tmp_xauth_value}.")
        Path(tmp_xauth_value).unlink()
        statefile.delete_variable("__ISAACLAB_TMP_XAUTH")


def create_x11_tmpfile(tmpfile: Path | None = None, tmpdir: Path | None = None) -> Path:
    """Creates an .xauth file with an MIT-MAGIC-COOKIE derived from the current ``DISPLAY`` environment variable.

    Args:
        tmpfile: A Path to a file which will be filled with the correct .xauth info.
        tmpdir: A Path to the directory where a random tmp file will be made.
            This is used as an ``--tmpdir arg`` to ``mktemp`` bash command.

    Returns:
        The Path to the .xauth file.
    """
    if tmpfile is None:
        if tmpdir is None:
            add_tmpdir = ""
        else:
            add_tmpdir = f"--tmpdir={tmpdir}"
        # Create .tmp file with .xauth suffix
        tmp_xauth = Path(
            subprocess.run(
                ["mktemp", "--suffix=.xauth", f"{add_tmpdir}"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    else:
        tmpfile.touch()
        tmp_xauth = tmpfile

    # Derive current MIT-MAGIC-COOKIE and make it universally addressable
    xauth_cookie = subprocess.run(
        ["xauth", "nlist", os.environ["DISPLAY"]], capture_output=True, text=True, check=True
    ).stdout.replace("ffff", "")

    # Merge the new cookie into the create .tmp file
    subprocess.run(["xauth", "-f", tmp_xauth, "nmerge", "-"], input=xauth_cookie, text=True, check=True)

    return tmp_xauth


def x11_refresh(statefile: StateFile):
    """Refresh the temporary .xauth file used for X11 forwarding.

    If x11 is enabled, this function generates a new .xauth file with the current MIT-MAGIC-COOKIE-1.
    The new file uses the same filename so that the bind-mount and ``XAUTHORITY`` var from build-time
    still work.

    As the envar ``DISPLAY` informs the contents of the MIT-MAGIC-COOKIE-1, that value within the container
    will also need to be updated to the current value on the host. Currently, this done automatically in
    :meth:`ContainerInterface.enter` method.

    The function exits if X11 forwarding is enabled but the temporary .xauth file does not exist. In this case,
    the user must rebuild the container.

    Args:
        statefile: An instance of the configuration file class.
    """
    # set the namespace to X11 for the statefile
    statefile.namespace = "X11"

    # check if X11 forwarding is enabled
    is_x11_forwarding_enabled = statefile.get_variable("X11_FORWARDING_ENABLED")
    # load the value of the temporary xauth file
    tmp_xauth_value = statefile.get_variable("__ISAACLAB_TMP_XAUTH")

    # print the current configuration
    if is_x11_forwarding_enabled is not None:
        status = "enabled" if is_x11_forwarding_enabled == "1" else "disabled"
        print(f"[INFO] X11 Forwarding is {status} from the settings in '.container.cfg'")

    # if the file exists, delete it and create a new one
    if tmp_xauth_value is not None and Path(tmp_xauth_value).exists():
        # remove the file and create a new one
        Path(tmp_xauth_value).unlink()
        create_x11_tmpfile(tmpfile=Path(tmp_xauth_value))
        # update the statefile with the new path
        statefile.set_variable("__ISAACLAB_TMP_XAUTH", str(tmp_xauth_value))
    elif tmp_xauth_value is None:
        if is_x11_forwarding_enabled is not None and is_x11_forwarding_enabled == "1":
            print(
                "[ERROR] X11 forwarding is enabled but the temporary .xauth file does not exist."
                " Please rebuild the container by running: './docker/container.py start'"
            )
            sys.exit(1)
        else:
            print("[INFO] X11 forwarding is disabled. No action taken.")
