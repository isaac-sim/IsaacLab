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


def is_windows():
    """Check if the platform is Windows."""
    return platform.system().lower() == "windows"


def is_arm():
    """Check if the architecture is ARM (likely Mac)."""
    machine = platform.system().lower()
    return "aarch64" in machine or "arm64" in machine


def run_command(cmd, cwd=None, env=None, shell=False, check=True):
    """Run a command in a subprocess."""

    if cwd is None:
        cwd = ISAACLAB_ROOT

    try:
        subprocess.run(cmd, cwd=cwd, env=env, shell=shell, check=check)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}: {e.cmd}")
        sys.exit(e.returncode)


def extract_python_exe():
    """
    Find the Python executable to use.
    """
    python_exe = None

    # Try conda python.
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        # Use conda python.
        if is_windows():
            python_exe = Path(conda_prefix) / "python.exe"
        else:
            python_exe = Path(conda_prefix) / "bin" / "python"
            if not python_exe.exists():
                python_exe = Path(conda_prefix) / "bin" / "python3"

    # Try uv virtual environment python.
    if not python_exe or not Path(python_exe).exists():
        venv_prefix = os.environ.get("VIRTUAL_ENV")
        if venv_prefix:
            if is_windows():
                python_exe = Path(venv_prefix) / "Scripts" / "python.exe"
            else:
                python_exe = Path(venv_prefix) / "bin" / "python"
                if not python_exe.exists():
                    python_exe = Path(venv_prefix) / "bin" / "python3"

    # Try kit python.
    if not python_exe or not Path(python_exe).exists():
        if is_windows():
            python_exe = DEFAULT_ISAAC_SIM_PATH / "python.bat"
        else:
            python_exe = DEFAULT_ISAAC_SIM_PATH / "python.sh"

    # Try system python.
    if not python_exe or not Path(python_exe).exists():
        python_exe = shutil.which("python") or shutil.which("python3")
        python_exe = Path(python_exe) if python_exe else None

    # See if we found it.
    if not python_exe or not Path(python_exe).exists():
        # Check if we can use python that is running us.
        # This handles docker or system installs.
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True, check=False)
            if "isaacsim-rl" in result.stdout:
                python_exe = sys.executable
        except Exception:
            pass

    # Nothing found, error out :)
    if not python_exe or not Path(python_exe).exists():
        print(f"[ERROR] Unable to find any Python executable at path: '{python_exe}'")
        print("\tThis could be due to the following reasons:")
        print("\t1. Conda or uv environment is not activated.")
        print("\t2. Isaac Sim pip package 'isaacsim-rl' is not installed.")
        print(f"\t3. Python executable is not available at the default path: {DEFAULT_ISAAC_SIM_PATH}")
        sys.exit(1)

    print_info(f"Using Python: \"{python_exe}\"")

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
        print(f"[ERROR] Unable to find the Isaac Sim directory: '{isaac_path}'")
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

        print(f"[ERROR] No Isaac Sim executable found at path: {isaac_path}")
        sys.exit(1)

    return [str(isaacsim_exe)]


def is_isaacsim_version_5_x():
    """Detects Isaac Sim version and returns True if version starts with 5.X"""
    # 1. Version file
    version_file = DEFAULT_ISAAC_SIM_PATH / "VERSION"
    if version_file.exists():
        with open(version_file) as f:
            version = f.read().strip()
            if version.startswith("5."):
                return True
            return False

    # 2. Try importing
    try:
        from importlib.metadata import version

        v = version("isaacsim")
        return v.startswith("5.")
    except Exception:
        pass

    return False


def run_docker_helper(args):
    """Run the docker container helper script."""
    script_path = ISAACLAB_ROOT / "docker" / "container.sh"
    # On Windows this might fail if no bash, but usually docker implies wsl or similar env.
    print(f"[INFO] Running docker utility script from: {script_path}")
    if is_windows():
        run_command(["bash", str(script_path)] + args, check=False)
    else:
        run_command(["bash", str(script_path)] + args, check=False)


def run_python_command(script_path, args, is_module=False):
    """Run a python script using the python executable."""
    cmd = [extract_python_exe()]
    if is_module:
        cmd.append("-m")
    cmd.append(str(script_path))
    cmd.extend(args)
    env = os.environ.copy()
    run_command(cmd, env=env, check=False)


def update_vscode_settings():
    """Update the vscode settings from template and Isaac Sim settings"""

    print("[INFO] Setting up vscode settings...")

    # Path to setup_vscode.py.
    setup_vscode_script = ISAACLAB_ROOT / ".vscode" / "tools" / "setup_vscode.py"

    # Check if the file exists before attempting to run it.
    if setup_vscode_script.exists():
        run_python_command(setup_vscode_script, [])
    else:
        print("[WARNING] Unable to find the script 'setup_vscode.py'. Aborting vscode settings setup.")


def build_docs():
    print("[INFO] Building documentation...")
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
    print(f"[INFO] Documentation built at {index_path}")
    if not is_windows():
        print(f"[INFO] Open with: xdg-open {index_path}")
