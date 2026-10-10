# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import subprocess

from ..utils import ISAACLAB_ROOT, extract_python_exe, print_info, run_command


def command_format(files: list[str] | None = None) -> None:
    """Run pre-commit once on selected files, or all tracked files when none are supplied."""
    python_exe = extract_python_exe()

    # Install the formatting tool into the interpreter running the CLI when needed.
    result = run_command(
        [python_exe, "-c", "import pre_commit"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if result.returncode != 0:
        print_info('Pre-commit not found. Installing "pre-commit" module...')
        run_command(["uv", "pip", "install", "--python", python_exe, "pre-commit"])

    print_info("Formatting the repository...")

    scope = ["--files", *files] if files else ["--all-files"]
    run_command([python_exe, "-m", "pre_commit", "run", *scope], cwd=ISAACLAB_ROOT)
