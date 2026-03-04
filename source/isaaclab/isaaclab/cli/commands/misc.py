# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Misc commands"""

from ..utils import (
    ISAACLAB_ROOT,
    extract_isaacsim_exe,
    extract_python_exe,
    is_windows,
    print_info,
    print_warning,
    run_command,
    run_python_command,
)


def command_run_isaacsim(sim_args: list[str]) -> None:
    """Run Isaac Sim (-s).

    Args:
        sim_args: Additional arguments passed to the Isaac Sim executable.
    """

    isaacsim_exe = extract_isaacsim_exe()
    print_info(f"Running Isaac Sim from: {isaacsim_exe}")

    isaacsim_exe.append("--ext-folder")
    isaacsim_exe.append(str(ISAACLAB_ROOT / "source"))
    isaacsim_exe.extend(sim_args)

    run_command(isaacsim_exe, check=False)


def command_new(new_args: list[str]) -> None:
    """Create a new external project or internal task from template (-n).

    Args:
        new_args: Arguments forwarded to the template generator CLI.
    """

    print_info("Installing template dependencies...")
    reqs = ISAACLAB_ROOT / "tools" / "template" / "requirements.txt"
    run_python_command("pip", ["install", "-q", "-r", str(reqs)], is_module=True)

    print_info("Running template generator...")
    cli_script = ISAACLAB_ROOT / "tools" / "template" / "cli.py"
    run_python_command(cli_script, new_args)


def command_test(test_args: list[str]) -> None:
    """Run pytest for Isaac Lab tests (-t).

    Args:
        test_args: Additional pytest arguments.
    """
    run_python_command("-m", ["pytest", str(ISAACLAB_ROOT / "tools")] + test_args)


def command_vscode_settings() -> None:
    """Update the vscode settings from template and Isaac Sim settings"""

    print_info("Setting up vscode settings...")

    # Path to setup_vscode.py.
    setup_vscode_script = ISAACLAB_ROOT / ".vscode" / "tools" / "setup_vscode.py"

    # Check if the file exists before attempting to run it.
    if setup_vscode_script.exists():
        run_python_command(setup_vscode_script, [])
        print_info("VS Code settings generated successfully.")
    else:
        print_warning("Unable to find the script 'setup_vscode.py'. Aborting vscode settings setup.")


def command_build_docs() -> None:
    """Build the documentation."""
    print_info("Building documentation...")
    python_exe = extract_python_exe()
    docs_dir = ISAACLAB_ROOT / "docs"

    # Install reqs.
    run_command(
        [python_exe, "-m", "pip", "install", "-r", "requirements.txt"],
        cwd=docs_dir,
    )

    # Build
    # sphinx-build -b html -d _build/doctrees . _build/current
    # using python -m sphinx.
    out_dir = docs_dir / "_build" / "current"
    cmd = [
        python_exe,
        "-m",
        "sphinx",
        "-b",
        "html",
        "-d",
        "_build/doctrees",
        ".",
        str(out_dir),
    ]
    run_command(cmd, cwd=docs_dir)

    index_path = out_dir / "index.html"
    print_info(f"Documentation built at {index_path}")
    if not is_windows():
        print_info(f"Open with: xdg-open {index_path}")


def command_run_docker(args: list[str]) -> None:
    """Run the docker container helper script (docker/container.py).

    Args:
        args: Arguments forwarded to ``docker/container.py``.
    """
    script_path = ISAACLAB_ROOT / "docker" / "container.py"
    print_info(f"Running docker utility script from: {script_path}")
    run_python_command(script_path, args)
