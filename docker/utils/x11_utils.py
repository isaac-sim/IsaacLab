# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from utils.statefile import Statefile


# This method of x11 enabling forwarding was inspired by osrf/rocker
# https://github.com/osrf/rocker
def configure_x11(statefile: Statefile) -> dict[str, str]:
    """
    Configure X11 forwarding by creating and managing a temporary .xauth file.

    If xauth is not installed, prompt the user to install it. If the .xauth file
    does not exist, create it and configure it with the necessary xauth cookie.

    Args:
        statefile: An instance of the Statefile class to manage state variables.

    Returns:
        dict: A dictionary where the key is __ISAACLAB_TMP_XAUTH (referenced in x11.yaml)
              and the value is the corresponding tmp file which has been created.
    """
    if not shutil.which("xauth"):
        print("[INFO] xauth is not installed.")
        print("[INFO] Please install it with 'apt install xauth'")
        exit(1)
    statefile.namespace = "X11"
    __ISAACLAB_TMP_XAUTH = statefile.load_variable("__ISAACLAB_TMP_XAUTH")
    if __ISAACLAB_TMP_XAUTH is None or not Path(__ISAACLAB_TMP_XAUTH).exists():
        __ISAACLAB_TMP_DIR = subprocess.run(["mktemp", "-d"], capture_output=True, text=True, check=True).stdout.strip()
        __ISAACLAB_TMP_XAUTH = create_x11_tmpfile(tmpdir=Path(__ISAACLAB_TMP_DIR))
        statefile.set_variable("__ISAACLAB_TMP_XAUTH", str(__ISAACLAB_TMP_XAUTH))
    else:
        __ISAACLAB_TMP_DIR = Path(__ISAACLAB_TMP_XAUTH).parent
    return {"__ISAACLAB_TMP_XAUTH": str(__ISAACLAB_TMP_XAUTH), "__ISAACLAB_TMP_DIR": str(__ISAACLAB_TMP_DIR)}


def x11_check(statefile: Statefile) -> tuple[list[str], dict[str, str]] | None:
    """
    Check and configure X11 forwarding based on user input and existing state.

    Prompt the user to enable or disable X11 forwarding if not already configured.
    Configure X11 forwarding if enabled.

    Args:
        statefile: An instance of the Statefile class to manage state variables.

    Returns:
        list or str: A list containing the x11.yaml file configuration option if X11 forwarding is enabled,
                     otherwise None
    """
    statefile.namespace = "X11"
    __ISAACLAB_X11_FORWARDING_ENABLED = statefile.load_variable("__ISAACLAB_X11_FORWARDING_ENABLED")
    if __ISAACLAB_X11_FORWARDING_ENABLED is None:
        print("[INFO] X11 forwarding from the Isaac Lab container is off by default.")
        print(
            "[INFO] It will fail if there is no display, or this script is being run via ssh without proper"
            " configuration."
        )
        x11_answer = input("Would you like to enable it? (y/N) ")
        if x11_answer.lower() == "y":
            __ISAACLAB_X11_FORWARDING_ENABLED = "1"
            statefile.set_variable("__ISAACLAB_X11_FORWARDING_ENABLED", "1")
            print("[INFO] X11 forwarding is enabled from the container.")
        else:
            __ISAACLAB_X11_FORWARDING_ENABLED = "0"
            statefile.set_variable("__ISAACLAB_X11_FORWARDING_ENABLED", "0")
            print("[INFO] X11 forwarding is disabled from the container.")
    else:
        print(f"[INFO] X11 Forwarding is configured as {__ISAACLAB_X11_FORWARDING_ENABLED} in .container.cfg")
        if __ISAACLAB_X11_FORWARDING_ENABLED == "1":
            print("[INFO] To disable X11 forwarding, set __ISAACLAB_X11_FORWARDING_ENABLED=0 in .container.cfg")
        else:
            print("[INFO] To enable X11 forwarding, set __ISAACLAB_X11_FORWARDING_ENABLED=1 in .container.cfg")

    if __ISAACLAB_X11_FORWARDING_ENABLED == "1":
        x11_envar = configure_x11(statefile)
        # If X11 forwarding is enabled, return the proper args to
        # compose the x11.yaml file. Else, return an empty string.
        return (["--file", "x11.yaml"], x11_envar)

    return None


def x11_cleanup(statefile: Statefile):
    """
    Clean up the temporary .xauth file used for X11 forwarding.

    If the .xauth file exists, delete it and remove the corresponding state variable.

    Args:
        statefile: An instance of the Statefile class to manage state variables.
    """
    statefile.namespace = "X11"
    __ISAACLAB_TMP_XAUTH = statefile.load_variable("__ISAACLAB_TMP_XAUTH")
    if __ISAACLAB_TMP_XAUTH is not None and Path(__ISAACLAB_TMP_XAUTH).exists():
        print(f"[INFO] Removing temporary Isaac Lab .xauth file {__ISAACLAB_TMP_XAUTH}.")
        Path(__ISAACLAB_TMP_XAUTH).unlink()
        statefile.delete_variable("__ISAACLAB_TMP_XAUTH")


def create_x11_tmpfile(tmpfile: Path | None = None, tmpdir: Path | None = None) -> Path:
    """
    Creates an .xauth file with an MIT-MAGIC-COOKIE derived from the current DISPLAY,
    returns its location as a Path.

    Args:
        tmpfile: A Path to a file which will be filled with the correct .xauth info
        tmpdir: A Path to the directory where a random tmp file will be made,
                used as an --tmpdir arg to mktemp

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


def x11_refresh(statefile: Statefile):
    """
    If x11 is enabled, generates a new .xauth file with the current MIT-MAGIC-COOKIE-1,
    using the same filename so that the bind-mount and
    XAUTHORITY var from build-time still work. DISPLAY will also
    need to be updated in the container environment command.

    Args:
        statefile: An instance of the Statefile class to manage state variables.
    """
    statefile.namespace = "X11"
    __ISAACLAB_TMP_XAUTH = Path(statefile.load_variable("__ISAACLAB_TMP_XAUTH"))
    if __ISAACLAB_TMP_XAUTH is not None and __ISAACLAB_TMP_XAUTH.exists():
        __ISAACLAB_TMP_XAUTH.unlink()
        create_x11_tmpfile(tmpfile=__ISAACLAB_TMP_XAUTH)
        statefile.set_variable("__ISAACLAB_TMP_XAUTH", str(__ISAACLAB_TMP_XAUTH))
