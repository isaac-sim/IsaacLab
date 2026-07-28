# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Misc commands"""

import os
import shutil
from pathlib import Path

from ..utils import (
    ISAACLAB_ROOT,
    extract_isaacsim_exe,
    extract_python_exe,
    get_pip_command,
    is_windows,
    print_error,
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
    pip_cmd = get_pip_command(python_exe)
    run_command(
        pip_cmd + ["install", "-r", "requirements.txt"],
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


def command_build_isaacsim(source_path: str) -> None:
    """Build Isaac Sim from source and make it usable through ``uv`` (--isaacsim_source).

    Builds the Isaac Sim checkout when it has no build output yet, packages that build into
    Python wheels, links the wheels into the Isaac Lab repository as ``_isaac_sim_wheels``,
    and re-resolves Isaac Sim from those wheels. Afterwards, run Isaac Lab against the build
    with ``uv run --extra isaacsim-local``.

    Args:
        source_path: Path to an Isaac Sim source checkout.
    """
    isaacsim_root = Path(source_path).expanduser().resolve()
    build_script = isaacsim_root / ("build.bat" if is_windows() else "build.sh")
    repo_script = isaacsim_root / ("repo.bat" if is_windows() else "repo.sh")

    if not build_script.is_file():
        print_error(f"'{isaacsim_root}' is not an Isaac Sim source checkout ({build_script.name} not found).")
        print_info("Clone it first with: git clone https://github.com/isaac-sim/IsaacSim.git")
        raise SystemExit(1)

    build_dir = isaacsim_root / "_build"
    if build_dir.is_dir():
        print_info(f"Using the existing Isaac Sim build in {build_dir}.")
        print_info(f"To rebuild, run {build_script} yourself before this command.")
    else:
        print_info("Building Isaac Sim from source. This takes a while...")
        run_command([str(build_script)], cwd=isaacsim_root)

    print_info("Packaging the Isaac Sim build as Python wheels...")
    for repo_args in (["python_package", "--create"], ["comment_archive_deps"], ["python_package", "--wheel"]):
        run_command([str(repo_script)] + repo_args, cwd=isaacsim_root)

    wheel_dir = build_dir / "packages" / "dist"
    wheels = sorted(wheel_dir.glob("*.whl")) if wheel_dir.is_dir() else []
    if not wheels:
        print_error(f"No wheels were produced in {wheel_dir}.")
        raise SystemExit(1)
    print_info(f"Built {len(wheels)} wheel(s) in {wheel_dir}.")

    # A stable, git-ignored path inside the repository keeps ``UV_FIND_LINKS`` short and
    # valid across shells, the same way ``_isaac_sim`` does for the Kit build.
    link_path = ISAACLAB_ROOT / "_isaac_sim_wheels"
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink():
            link_path.unlink()
        else:
            print_error(f"{link_path} exists and is not a symbolic link. Remove it and re-run.")
            raise SystemExit(1)
    try:
        link_path.symlink_to(wheel_dir, target_is_directory=True)
        find_links = link_path.name
        print_info(f"Linked {link_path} -> {wheel_dir}")
    except OSError as error:
        # Windows requires elevation (or developer mode) to create symbolic links.
        find_links = str(wheel_dir)
        print_warning(f"Could not create {link_path} ({error}). Using the absolute wheel path instead.")

    # Re-resolve Isaac Sim so the lock file picks the local wheels over the published release.
    if shutil.which("uv") is not None:
        print_info("Re-resolving Isaac Sim from the local wheels...")
        run_command(["uv", "lock", "--upgrade-package", "isaacsim"], env={**os.environ, "UV_FIND_LINKS": find_links})
    else:
        print_warning("uv was not found on PATH. Run 'uv lock --upgrade-package isaacsim' once uv is available.")

    export_cmd = f"set UV_FIND_LINKS={find_links}" if is_windows() else f"export UV_FIND_LINKS={find_links}"
    print_info("Isaac Sim is ready. Run Isaac Lab against it with:")
    print_info(f"  {export_cmd}")
    print_info(
        "  uv run --extra isaacsim-local isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct presets=physx"
    )


def command_run_docker(args: list[str]) -> None:
    """Run the docker container helper script (docker/container.py).

    Args:
        args: Arguments forwarded to ``docker/container.py``.
    """
    script_path = ISAACLAB_ROOT / "docker" / "container.py"
    print_info(f"Running docker utility script from: {script_path}")
    run_python_command(script_path, args)
