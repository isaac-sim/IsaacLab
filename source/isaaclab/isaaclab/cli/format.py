# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import shutil
import subprocess

from .utils import ISAACLAB_ROOT, extract_python_exe, print_info, run_command


def format_code():
    """Run code formatting using pre-commit."""
    python_exe = extract_python_exe()

    # Reset the python path to avoid conflicts with pre-commit.
    # This is needed because the pre-commit hooks are installed in a
    # separate virtual environment and it uses the system python to run the hooks.
    env = os.environ.copy()
    if env.get("CONDA_DEFAULT_ENV") or env.get("VIRTUAL_ENV"):
        env["PYTHONPATH"] = ""

    # Check if pre-commit is installed and install it if not.
    # We check both the executable and the python module.
    pre_commit_installed = False
    if shutil.which("pre-commit"):
        pre_commit_installed = True
    else:
        try:
            subprocess.run(
                [python_exe, "-m", "pip", "show", "pre-commit"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            pre_commit_installed = True
        except subprocess.CalledProcessError:
            pass

    if not pre_commit_installed:
        print_info("Installing pre-commit module...")
        run_command([python_exe, "-m", "pip", "install", "pre-commit"])

    print_info("Formatting the repository...")

    try:
        # Run pre-commit as a module since we may have just installed it.
        run_command([python_exe, "-m", "pre_commit", "run", "--all-files"], env=env)

    except SystemExit:
        # Pre-commit exits with code=1 when files changed,
        # that is expected.
        pass

        # To verify if the error is due to pre-commit just changing files,
        # run pre-commit again to see if it exits with code=0.
        run_command(
            [python_exe, "-m", "pre_commit", "run", "--all-files"],
            cwd=ISAACLAB_ROOT,
            env=env,
        )

    finally:
        pass
