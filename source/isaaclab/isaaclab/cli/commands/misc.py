# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Misc commands"""

import argparse
import platform
import shutil
import sys
from pathlib import Path

from ..utils import (
    ISAAC_SIM_SOURCE_BUILD_MARKER,
    ISAACLAB_ROOT,
    extract_isaacsim_exe,
    is_windows,
    print_error,
    print_info,
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

    print_info("Running template generator...")
    cli_script = ISAACLAB_ROOT / "tools" / "template" / "cli.py"
    run_python_command(cli_script, new_args)


def command_test(test_args: list[str]) -> None:
    """Run pytest for Isaac Lab tests (-t).

    Args:
        test_args: Additional pytest arguments.
    """
    run_python_command("-m", ["pytest", str(ISAACLAB_ROOT / "tools")] + test_args)


def command_editor(editor_args: list[str], project_dir: Path | None = None) -> None:
    """Generate editor settings and import paths for a workspace.

    Args:
        editor_args: Editor setup command arguments.
        project_dir: Workspace root. Defaults to the current directory.
    """
    parser = argparse.ArgumentParser(prog="isaaclab --editor", description="Set up editor paths and settings.")
    parser.add_argument("--isaac_path", help="Absolute path to the Isaac Sim installation.")
    parser.add_argument("--verbose", action="store_true", help="Print discovered extension paths.")
    args = parser.parse_args(editor_args)

    # The installation CLI must start before Isaac Lab's runtime dependencies are installed.
    from ...utils.editor import setup_editor

    print_info("Setting up editor paths and settings...")
    try:
        setup_editor(project_dir or Path.cwd(), isaac_path=args.isaac_path, verbose=args.verbose)
    except ValueError as error:
        parser.error(str(error))


def command_build_docs() -> None:
    """Build the documentation."""
    print_info("Building documentation...")
    docs_dir = ISAACLAB_ROOT / "docs"

    uv_exe = shutil.which("uv")
    if uv_exe is None:
        print_error("uv could not be found. Please install uv and try again.")
        print_error("https://docs.astral.sh/uv/getting-started/installation/")
        raise SystemExit(1)

    out_dir = docs_dir / "_build" / "current"
    cmd = [
        uv_exe,
        "run",
        "--isolated",
        "--extra",
        "test",
        "--",
        "python",
        "-m",
        "sphinx",
        "-W",
        "--keep-going",
        "-j",
        "auto",
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

    Runs Isaac Sim's incremental build and links its release tree into Isaac Lab as ``_isaac_sim``.
    Python commands launched through the Isaac Lab CLI use the active environment's interpreter
    through Isaac Sim's ``python.sh`` or ``python.bat`` wrapper, so they load the live build without
    packaging or installing it as wheels.

    Args:
        source_path: Path to an Isaac Sim source checkout.
    """
    isaacsim_root = Path(source_path).expanduser().resolve()
    build_script = isaacsim_root / ("build.bat" if is_windows() else "build.sh")

    if not build_script.is_file():
        print_error(f"'{isaacsim_root}' is not an Isaac Sim source checkout ({build_script.name} not found).")
        print_info("Clone it first with: git clone https://github.com/isaac-sim/IsaacSim.git")
        raise SystemExit(1)

    print_info("Incrementally building Isaac Sim from source. This may take a while...")
    run_command([str(build_script)], cwd=isaacsim_root)

    release_dir = _resolve_isaacsim_release_dir(isaacsim_root)
    python_launcher = release_dir / ("python.bat" if is_windows() else "python.sh")
    if not python_launcher.is_file():
        print_error(f"The Isaac Sim build did not produce {python_launcher}.")
        raise SystemExit(1)

    link_path = ISAACLAB_ROOT / "_isaac_sim"
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink():
            link_path.unlink()
        else:
            print_error(f"{link_path} exists and is not a symbolic link. Remove it and re-run.")
            raise SystemExit(1)
    try:
        link_path.symlink_to(release_dir, target_is_directory=True)
    except OSError as error:
        print_error(f"Could not link {link_path} to {release_dir}: {error}")
        if is_windows():
            print_info("Enable Windows Developer Mode or run from an elevated terminal, then retry.")
        raise SystemExit(1) from error
    (release_dir / ISAAC_SIM_SOURCE_BUILD_MARKER).touch()
    print_info(f"Linked {link_path} -> {release_dir}")
    _repoint_source_build_prebundles()

    print_info("Isaac Sim is ready. Python commands now use the live source build through '_isaac_sim'.")
    print_info("Run Isaac Lab against it with:")
    print_info("  uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=isaacsim_physx")


def _resolve_isaacsim_release_dir(isaacsim_root: Path) -> Path:
    """Resolve the platform-specific Isaac Sim release directory."""
    machine = platform.machine().lower()
    targets = {
        ("linux", "amd64"): "linux-x86_64",
        ("linux", "x86_64"): "linux-x86_64",
        ("linux", "aarch64"): "linux-aarch64",
        ("linux", "arm64"): "linux-aarch64",
        ("win32", "amd64"): "windows-x86_64",
        ("win32", "x86_64"): "windows-x86_64",
    }
    target = targets.get((sys.platform, machine))
    if target is None:
        print_error(f"Isaac Sim source builds are not supported on platform '{sys.platform}' with machine '{machine}'.")
        raise SystemExit(1)
    return isaacsim_root / "_build" / target / "release"


def _repoint_source_build_prebundles() -> None:
    """Keep Isaac Sim's prebundled packages from shadowing the active environment."""
    # ``install`` imports ``command_editor`` from this module, so defer this import until
    # both command modules are initialized. Reuse the same protection as the legacy installer.
    from .install import _repoint_prebundle_packages

    _repoint_prebundle_packages()


def command_run_docker(args: list[str]) -> None:
    """Run the docker container helper script (docker/container.py).

    Args:
        args: Arguments forwarded to ``docker/container.py``.
    """
    script_path = ISAACLAB_ROOT / "docker" / "container.py"
    print_info(f"Running docker utility script from: {script_path}")
    run_python_command(script_path, args)
