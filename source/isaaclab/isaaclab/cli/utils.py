# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Path to Isaac Lab installation.
ISAACLAB_ROOT = Path(__file__).parents[4].resolve()

# Default path to look for Isaac Sim is _isaac_sim symlink.
DEFAULT_ISAAC_SIM_PATH = ISAACLAB_ROOT / "_isaac_sim"

# ANSI colors.
_ANSI_COLOR_RESET = "\033[0m"
_ANSI_COLOR_INFO = "\033[36m"  # cyan
_ANSI_COLOR_WARNING = "\033[33m"  # yellow
_ANSI_COLOR_ERROR = "\033[31m"  # red
_ANSI_COLOR_DEBUG = "\033[1;32m"  # bold green


def is_windows():
    """Check if the platform is Windows."""
    return platform.system().lower() == "windows"


def is_arm():
    """Check if the architecture is ARM (likely Mac)."""
    machine = platform.system().lower()
    return "aarch64" in machine or "arm64" in machine


def _colorize(label, color, stream):
    """Colorize a label, if the stream supports colors."""

    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    color_supported = hasattr(stream, "isatty") and stream.isatty()

    if color_supported:
        return f"{color}{label}{_ANSI_COLOR_RESET}"

    return label


def print_info(message, stream=sys.stdout):
    label = _colorize("[INFO]", _ANSI_COLOR_INFO, stream)
    print(f"{label} {message}", file=stream)


def print_warning(message, stream=sys.stdout):
    label = _colorize("[WARNING]", _ANSI_COLOR_WARNING, stream)
    print(f"{label} {message}", file=stream)


def print_error(message, stream=sys.stderr):
    label = _colorize("[ERROR]", _ANSI_COLOR_ERROR, stream)
    print(f"{label} {message}", file=stream)


def print_debug(message, stream=sys.stdout):
    if os.environ.get("DEBUG") != "1":
        return
    label = _colorize("[DEBUG]", _ANSI_COLOR_DEBUG, stream)
    print(f"{label} {message}", file=stream)


def run_command(cmd, cwd=None, env=None, shell=False, check=True, stdout=None, stderr=None):
    """Run a command in a subprocess."""

    if cwd is None:
        cwd = ISAACLAB_ROOT

    command_str = " ".join(str(part) for part in cmd) if isinstance(cmd, (list, tuple)) else str(cmd)

    # Print some debug info.
    print_debug(f'run_command(): CWD: "{cwd}"')
    print_debug(f'run_command(): CMD: "{command_str}"')

    if env is None:
        print_debug("run_command(): ENV: <os.environ>")
    else:
        current_env = os.environ
        env_added = {key: value for key, value in env.items() if key not in current_env}
        env_changed = {
            key: {"from": current_env[key], "to": value}
            for key, value in env.items()
            if key in current_env and current_env[key] != value
        }
        env_removed = [key for key in current_env if key not in env]

        if not env_added and not env_changed and not env_removed:
            print_debug("run_command(): ENV: <os.environ>")
        else:
            if env_added:
                print_debug(f"run_command(): ENV added: {env_added}")
            if env_changed:
                print_debug(f"run_command(): ENV changed: {env_changed}")
            if env_removed:
                print_debug(f"run_command(): ENV removed: {env_removed}")

    try:
        return subprocess.run(cmd, cwd=cwd, env=env, shell=shell, check=check, stdout=stdout, stderr=stderr)
    except subprocess.CalledProcessError as e:
        print_error(f'Command failed with code {e.returncode}: "{command_str}"')
        sys.exit(e.returncode)


def extract_python_exe():
    """
    Find the Python executable to use.
    """
    python_exe = None

    # Try conda python.
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        print_debug(f"extract_python_exe(): Found CONDA_PREFIX: {conda_prefix}")
        # Use conda python.
        if is_windows():
            python_exe = Path(conda_prefix) / "python.exe"
        else:
            python_exe = Path(conda_prefix) / "bin" / "python"
            if not python_exe.exists():
                python_exe = Path(conda_prefix) / "bin" / "python3"
    else:
        print_debug("extract_python_exe(): No CONDA_PREFIX found.")

    # Try uv virtual environment python.
    if not python_exe or not Path(python_exe).exists():
        if python_exe:
            print_debug(
                f'extract_python_exe(): Conda python "{python_exe}" not found, '
                "trying to find virtual environment python."
            )

        venv_prefix = os.environ.get("VIRTUAL_ENV")
        if venv_prefix:
            print_debug(f"extract_python_exe(): Found VIRTUAL_ENV: {venv_prefix}")
            if is_windows():
                python_exe = Path(venv_prefix) / "Scripts" / "python.exe"
            else:
                python_exe = Path(venv_prefix) / "bin" / "python"
                if not python_exe.exists():
                    python_exe = Path(venv_prefix) / "bin" / "python3"
        else:
            print_debug("extract_python_exe(): No VIRTUAL_ENV found.")

    # Try kit python.
    if not python_exe or not Path(python_exe).exists():
        if python_exe:
            print_debug(
                f'extract_python_exe(): Virtual env python "{python_exe}" does not exist, trying to find Kit python...'
            )

        if is_windows():
            python_exe = DEFAULT_ISAAC_SIM_PATH / "python.bat"
        else:
            python_exe = DEFAULT_ISAAC_SIM_PATH / "python.sh"

        print_debug(f'extract_python_exe(): Checking for Kit python at: "{python_exe}"')

    # Try system python.
    if not python_exe or not Path(python_exe).exists():
        print_debug(f'extract_python_exe(): Kit python "{python_exe}" does not exist. Checking system python.')
        python_exe = shutil.which("python") or shutil.which("python3")
        python_exe = Path(python_exe) if python_exe else None
        print_debug(f"extract_python_exe(): System python candidate: {python_exe}")

    # See if we found it.
    if not python_exe or not Path(python_exe).exists():
        print_debug(
            f'extract_python_exe(): System python "{python_exe}" does not exist. '
            "Checking sys.executable (current Python interpreter)."
        )
        # Check if we can use python that is running us.
        # This handles docker or system installs.
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True, check=False)
            if "isaacsim-rl" in result.stdout:
                python_exe = sys.executable
                print_debug(f'extract_python_exe(): Found "isaacsim-rl" module in sys.executable: "{python_exe}"')
        except Exception:
            pass

    # Nothing found, error out :)
    if not python_exe or not Path(python_exe).exists():
        print_error(f"Unable to find any Python executable at path: '{python_exe}'")
        print("\tThis could be due to the following reasons:")
        print("\t1. Conda or uv environment is not activated.")
        print("\t2. Isaac Sim pip package 'isaacsim-rl' is not installed.")
        print(f"\t3. Python executable is not available at the default path: {DEFAULT_ISAAC_SIM_PATH}")
        sys.exit(1)

    print_info(f'Using Python: "{python_exe}"')

    return str(python_exe)


def extract_isaacsim_path():
    """
    Find the Isaac Sim installation path.
    """
    # Use the sym-link path to Isaac Sim directory.
    isaac_path = DEFAULT_ISAAC_SIM_PATH

    # If above path is not available, try to find the path using python.
    if not isaac_path.exists():
        # Use the python executable to get the path.
        python_exe = extract_python_exe()
        # Retrieve the path importing isaac sim and getting the environment path.
        try:
            # Check if isaacsim-rl is installed.
            result = subprocess.run([python_exe, "-m", "pip", "list"], capture_output=True, text=True, check=False)
            if "isaacsim-rl" in result.stdout:
                # Helper to print env var.
                cmd = [python_exe, "-c", "import isaacsim; import os; print(os.environ['ISAAC_PATH'])"]
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if res.returncode == 0:
                    output = res.stdout.strip()
                    if output:
                        isaac_path = Path(output)
        except Exception:
            pass

    # Check if there is a path available.
    if not isaac_path.exists():
        # Throw an error if no path is found.
        print_error(f"Unable to find the Isaac Sim directory: '{isaac_path}'")
        print("\tThis could be due to the following reasons:")
        print("\t1. Conda environment is not activated.")
        print("\t2. Isaac Sim pip package 'isaacsim-rl' is not installed.")
        print(f"\t3. Isaac Sim directory is not available at the default path: {DEFAULT_ISAAC_SIM_PATH}")
        # Exit.
        sys.exit(1)

    return isaac_path


def extract_isaacsim_exe():
    """
    Find the Isaac Sim executable.
    """
    # Obtain the isaac sim path.
    isaac_path = extract_isaacsim_path()

    # Isaac Sim executable to use.
    if is_windows():
        isaacsim_exe = isaac_path / "isaac-sim.bat"
    else:
        isaacsim_exe = isaac_path / "isaac-sim.sh"

    # Check if there is a python path available.
    if not isaacsim_exe.exists():
        # Check for installation using Isaac Sim pip.
        # Note: pip installed Isaac Sim can only come from a direct
        # python environment, so we can directly use 'python' here.
        python_exe = sys.executable
        try:
            result = subprocess.run([python_exe, "-m", "pip", "list"], capture_output=True, text=True, check=False)
            if "isaacsim-rl" in result.stdout:
                # Isaac Sim - Python packages entry point.
                return ["isaacsim", "isaacsim.exp.full"]
        except Exception:
            pass

        print_error(f"No Isaac Sim executable found at path: {isaac_path}")
        sys.exit(1)

    return [str(isaacsim_exe)]


def determine_python_version():
    """Detect Isaac Sim version and return the matching Python version."""

    # 1. Version file
    version_file = DEFAULT_ISAAC_SIM_PATH / "VERSION"
    isaacsim_version = None
    if version_file.exists():
        with open(version_file) as f:
            version = f.read().strip()
            if version:
                isaacsim_version = version

    # 2. Try importing package metadata
    if isaacsim_version is None:
        try:
            from importlib.metadata import version

            isaacsim_version = version("isaacsim")
        except Exception:
            pass

    # We can't find the version, raise an error.
    if isaacsim_version is None:
        print_error("Unable to determine Isaac Sim version.")
        raise RuntimeError("Unable to determine Isaac Sim version.")

    if isaacsim_version.startswith("5."):
        python_version = "3.11"
    elif isaacsim_version.startswith("6."):
        python_version = "3.12"
    else:
        # We don't recognize the IsaacSim version.
        print_error(f"Unsupported Isaac Sim version: {isaacsim_version}")
        raise RuntimeError(f"Unsupported Isaac Sim version: {isaacsim_version}")

    print_info(f"Detected Isaac Sim {isaacsim_version} -> using python={python_version}")

    return python_version


def run_docker_helper(args):
    """Run the docker container helper script."""
    script_path = ISAACLAB_ROOT / "docker" / "container.py"
    print_info(f"Running docker utility script from: {script_path}")
    run_python_command(script_path, args)


def run_python_command(
    script_or_module,
    args,
    is_module=False,
    env=None,
    check=False,
):
    """Run a python script or module using the resolved Python executable.

    Args:
        script_or_module: Script path or module name to execute.
        args: Additional arguments.
        is_module: Whether to execute script_or_module as a module (``python -m``).
        env: Environment for the subprocess. Uses current environment if ``None``.
        check: Whether to raise ``CalledProcessError`` on non-zero exit codes.

    Returns:
        [subprocess.CompletedProcess] Result returned by ``subprocess.run``.
    """

    cmd = [extract_python_exe()]

    if is_module:
        cmd.append("-m")

    cmd.append(str(script_or_module))
    cmd.extend(args)

    if env is None:
        env = os.environ.copy()

    command_str = " ".join(str(part) for part in cmd)

    print_debug(f'run_python_command(): DIR: "{os.getcwd()}"')
    print_debug(f'run_python_command(): CMD: "{command_str}"')
    if env is None:
        print_debug("run_python_command(): ENV: <inherited>")
    else:
        print_debug(f"run_python_command(): ENV: {env}")

    return subprocess.run(
        cmd,
        env=env,
        check=check,
    )


def update_vscode_settings():
    """Update the vscode settings from template and Isaac Sim settings"""

    print_info("Setting up vscode settings...")

    # Path to setup_vscode.py.
    setup_vscode_script = ISAACLAB_ROOT / ".vscode" / "tools" / "setup_vscode.py"

    # Check if the file exists before attempting to run it.
    if setup_vscode_script.exists():
        run_python_command(setup_vscode_script, [])
    else:
        print_warning("Unable to find the script 'setup_vscode.py'. Aborting vscode settings setup.")


def build_docs():
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
