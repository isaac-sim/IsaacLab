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
from typing import IO, Any

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


def is_windows() -> bool:
    """Check if the platform is Windows."""
    return platform.system().lower() == "windows"


def is_arm() -> bool:
    """Check if the architecture is ARM (likely Mac)."""
    machine = platform.machine().lower()
    return "aarch64" in machine or "arm64" in machine


def _colorize(text: str, color: str, stream: IO[str]) -> str:
    """Colorize bit of text, if the stream supports colors or colors aren't disabled.

    Args:
        label: Text to colorize.
        color: ANSI color code prefix.
        stream: Output stream used to detectcolor support.

    Returns:
        Colorized label when supported; otherwise the original label.
    """

    if os.environ.get("NO_COLOR"):
        return f"{text}"

    if os.environ.get("TERM") == "dumb":
        return f"{text}"

    color_supported = hasattr(stream, "isatty") and stream.isatty()

    if not color_supported:
        return f"{text}"
    else:
        return f"{color}{text}{_ANSI_COLOR_RESET}"


def print_info(message: str, stream: IO[str] = sys.stdout) -> None:
    """Print informational message.

    Args:
        message: Message text to print.
        stream: Output stream where the message is written.
    """
    label = _colorize("[INFO]", _ANSI_COLOR_INFO, stream)
    print(f"{label} {message}", file=stream)


def print_warning(message: str, stream: IO[str] = sys.stdout) -> None:
    """Print warning message.

    Args:
        message: Message text to print.
        stream: Output stream where the message is written.
    """
    label = _colorize("[WARNING]", _ANSI_COLOR_WARNING, stream)
    print(f"{label} {message}", file=stream)


def print_error(message: str, stream: IO[str] = sys.stderr) -> None:
    """Print error message.

    Args:
        message: Message text to print.
        stream: Output stream where the message is written.
    """
    label = _colorize("[ERROR]", _ANSI_COLOR_ERROR, stream)
    print(f"{label} {message}", file=stream)


def print_debug(message: str, stream: IO[str] = sys.stdout) -> None:
    """Print debug message, when debugging is enabled.

    Args:
        message: Message text to print.
        stream: Output stream where the message is written.
    """
    if os.environ.get("DEBUG") != "1":
        return
    label = _colorize("[DEBUG]", _ANSI_COLOR_DEBUG, stream)
    print(f"{label} {message}", file=stream)


def _print_debug_env(prefix: str, env: dict[str, str] | None) -> None:
    """
    Print the environment for debugging purpose.
    Only prints the vars that are added, changed or removed vs the os.environ.

    Args:
        prefix: Prefix identifying the caller function in debug output.
        env: Environment to compare against os.environ.
    """

    if env is None:
        print_debug(f"{prefix}: ENV: <os.environ>")
        return

    current_env = os.environ
    env_added = {key: value for key, value in env.items() if key not in current_env}
    env_changed = {
        key: {"from": current_env[key], "to": value}
        for key, value in env.items()
        if key in current_env and current_env[key] != value
    }
    env_removed = [key for key in current_env if key not in env]

    if not env_added and not env_changed and not env_removed:
        print_debug(f"{prefix}: ENV: <os.environ>")
        return

    if env_added:
        print_debug(f"{prefix}: ENV added: {env_added}")
    if env_changed:
        print_debug(f"{prefix}: ENV changed: {env_changed}")
    if env_removed:
        print_debug(f"{prefix}: ENV removed: {env_removed}")


def run_command(
    cmd: str | list[str] | tuple[str, ...],
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    shell: bool = False,
    check: bool = True,
    stdout: int | IO[str] | None = None,
    stderr: int | IO[str] | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess[Any]:
    """Run a command in a subprocess.

    Args:
        cmd: Command to execute.
        cwd: Working directory for the subprocess.
        env: Environment variables for the subprocess.
        shell: Whether to run the command through the shell.
        check: Whether to raise on non-zero exit code.
        stdout: Standard output stream or redirection target.
        stderr: Standard error stream or redirection target.
        **kwargs: Additional keyword arguments forwarded to ``subprocess.run``.

    Returns:
        Result object returned by ``subprocess.run``.
    """

    if cwd is None:
        cwd = ISAACLAB_ROOT

    command_str = " ".join(str(part) for part in cmd) if isinstance(cmd, (list, tuple)) else str(cmd)

    # Print some debug info.
    print_debug(f'run_command(): CWD: "{cwd}"')
    print_debug(f'run_command(): CMD: "{command_str}"')
    _print_debug_env("run_command()", env)

    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            shell=shell,
            check=check,
            stdout=stdout,
            stderr=stderr,
            **kwargs,
        )
    except subprocess.CalledProcessError as e:
        print_error(f'Command failed with code {e.returncode}: "{command_str}"')
        sys.exit(e.returncode)


def extract_python_exe(allow_isaacsim_python: bool = True) -> str:
    """
    Find the Python executable to use.

    Args:
        allow_isaacsim_python:
        Allows to disable IsaacSim bundled Python fallback here to avoid recursion.
        This happens in CI or fresh environments where neither CONDA_PREFIX nor
        VIRTUAL_ENV is set and the default symlink path does not exist.
    """

    python_exe = None

    # Try uv virtual environment python.
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

    # Try conda python.
    if not python_exe or not Path(python_exe).exists():
        if python_exe:
            print_debug(
                f'extract_python_exe(): Venv python "{python_exe}" does not exist, trying to find conda python...'
            )

        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            print_debug(f"extract_python_exe(): Found CONDA_PREFIX: {conda_prefix}")
            if is_windows():
                python_exe = Path(conda_prefix) / "python.exe"
            else:
                python_exe = Path(conda_prefix) / "bin" / "python"
                if not python_exe.exists():
                    python_exe = Path(conda_prefix) / "bin" / "python3"
        else:
            print_debug("extract_python_exe(): No CONDA_PREFIX found.")

    # Try kit python.
    if allow_isaacsim_python and (not python_exe or not Path(python_exe).exists()):
        if python_exe:
            print_debug(
                f'extract_python_exe(): Venv python "{python_exe}" does not exist, trying to find Kit python...'
            )

        if is_windows():
            python_exe = extract_isaacsim_path() / "python.bat"
        else:
            python_exe = extract_isaacsim_path() / "python.sh"

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
            result = run_command([sys.executable, "-m", "pip", "list"], capture_output=True, text=True, check=False)
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


def extract_isaacsim_path() -> Path:
    """
    Find the Isaac Sim installation path.
    """
    # Use the sym-link path to Isaac Sim directory.
    isaacsim_path = DEFAULT_ISAAC_SIM_PATH

    # If above path is not available, try to find the path using python.
    if not isaacsim_path.exists():
        # Use the python executable to get the path.
        python_exe = extract_python_exe(allow_isaacsim_python=False)
        # Retrieve the path importing isaac sim and getting the environment path.
        try:
            # Check if isaacsim-rl is installed.
            result = run_command([python_exe, "-m", "pip", "list"], capture_output=True, text=True, check=False)
            if "isaacsim-rl" in result.stdout:
                # Helper to print env var.
                cmd = [python_exe, "-c", "import isaacsim; import os; print(os.environ['ISAAC_PATH'])"]
                res = run_command(cmd, capture_output=True, text=True, check=False)
                if res.returncode == 0:
                    output = res.stdout.strip()
                    if output:
                        isaacsim_path = Path(output)
        except Exception:
            pass

    # Check if there is a path available.
    if not isaacsim_path.exists():
        # Throw an error if no path is found.
        print_error(f"Unable to find the Isaac Sim directory: '{isaacsim_path}'")
        print("\tThis could be due to the following reasons:")
        print("\t1. Conda environment is not activated.")
        print("\t2. Isaac Sim pip package 'isaacsim-rl' is not installed.")
        print(f"\t3. Isaac Sim directory is not available at the default path: {DEFAULT_ISAAC_SIM_PATH}")
        # Exit.
        sys.exit(1)

    return isaacsim_path


def extract_isaacsim_exe() -> list[str]:
    """
    Find the Isaac Sim executable.
    """
    # Obtain the isaac sim path.
    isaacsim_path = extract_isaacsim_path()

    # Isaac Sim executable to use.
    if is_windows():
        isaacsim_exe = isaacsim_path / "isaac-sim.bat"
    else:
        isaacsim_exe = isaacsim_path / "isaac-sim.sh"

    # Check if there is a python path available.
    if not isaacsim_exe.exists():
        # Check for installation using Isaac Sim pip.
        # Note: pip installed Isaac Sim can only come from a direct
        # python environment, so we can directly use 'python' here.
        python_exe = sys.executable
        try:
            result = run_command([python_exe, "-m", "pip", "list"], capture_output=True, text=True, check=False)
            if "isaacsim-rl" in result.stdout:
                # Isaac Sim - Python packages entry point.
                return ["isaacsim", "isaacsim.exp.full"]
        except Exception:
            pass

        print_error(f"No Isaac Sim executable found at path: {isaacsim_path}")
        sys.exit(1)

    return [str(isaacsim_exe)]


def determine_python_version() -> str:
    """Detect Isaac Sim version and return the matching Python version."""

    # 1. Version file
    version_file = extract_isaacsim_path() / "VERSION"
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

    # We can't find the IS, show a warning and default to python 3.12 (IS 6.x).
    if isaacsim_version is None:
        python_version = "3.12"
        print_warning(f"Unable to determine Isaac Sim version. Defaulting to python={python_version}.")
        return python_version

    # We found some Isaac Sim
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


def run_python_command(
    script_or_module: str | Path,
    args: list[str],
    is_module: bool = False,
    env: dict[str, str] | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[Any]:
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

    command_str = " ".join(str(part) for part in cmd)

    print_debug(f'run_python_command(): CWD: "{os.getcwd()}"')
    print_debug(f'run_python_command(): CMD: "{command_str}"')

    return run_command(
        cmd,
        cwd=os.getcwd(),
        env=env,
        check=check,
    )
