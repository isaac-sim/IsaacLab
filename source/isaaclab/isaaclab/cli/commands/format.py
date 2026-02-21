# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import subprocess

from ..utils import ISAACLAB_ROOT, extract_python_exe, print_info, run_command


def command_format() -> None:
    """Run code formatting using pre-commit."""
    python_exe = extract_python_exe()

    def _run_pre_commit() -> None:
        run_command([python_exe, "-m", "pre_commit", "run", "--all-files"], cwd=ISAACLAB_ROOT)

    # Check if pre-commit is installed.

    pre_commit_module = False

    result = run_command(
        [python_exe, "-m", "pip", "show", "pre-commit"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if result.returncode == 0:
        pre_commit_module = True

    # If pre-commit is not installed, install it.
    if not pre_commit_module:
        print_info('Pre-commit not found. Installing "pre-commit" module...')
        run_command([python_exe, "-m", "pip", "install", "pre-commit"])

    print_info("Formatting the repository...")

    try:
        # Run pre-commit as a module since we may have just installed it.
        _run_pre_commit()

    except SystemExit:
        # Pre-commit exits with code=1 when files changed, that is expected.
        # To verify if the error is due to pre-commit just changing files,
        # run pre-commit again to see if it exits with code=0.
        print_info("Pre-commit changed some files, running it again to validate...")
        _run_pre_commit()

    finally:
        pass
