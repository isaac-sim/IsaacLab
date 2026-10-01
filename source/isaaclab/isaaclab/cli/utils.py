# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import platform
import site
import subprocess
import sys
from pathlib import Path
from typing import IO, Any

from ..paths import ISAACLAB_ROOT

# Default path to look for Isaac Sim is _isaac_sim symlink.
DEFAULT_ISAAC_SIM_PATH = ISAACLAB_ROOT / "_isaac_sim"

# Marker written into live Isaac Sim source builds linked by ``--isaacsim_source``.
ISAAC_SIM_SOURCE_BUILD_MARKER = ".isaaclab_source_build"

# Short script names supported by ``isaaclab -p``.
_PYTHON_SCRIPT_ALIASES = {
    "train.py": ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "train.py",
    "train_multigpu.py": ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "train_multigpu.py",
    "play.py": ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "play.py",
}

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


def is_isaac_sim_source_build(isaac_sim_path: Path = DEFAULT_ISAAC_SIM_PATH) -> bool:
    """Check whether an Isaac Sim directory is a live source build managed by Isaac Lab.

    Args:
        isaac_sim_path: Isaac Sim installation directory.

    Returns:
        Whether the source-build marker exists in the directory.
    """
    return (isaac_sim_path / ISAAC_SIM_SOURCE_BUILD_MARKER).is_file()


def _pyvenv_home(venv_path: Path) -> Path | None:
    """Return the interpreter directory a virtual environment was created from.

    Args:
        venv_path: Virtual environment root.

    Returns:
        The ``home`` directory recorded in ``pyvenv.cfg``, or ``None`` when it cannot be read.
    """
    config = venv_path / "pyvenv.cfg"
    if not config.is_file():
        return None
    for line in config.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "home":
            return Path(value.strip())
    return None


def _is_within(path: Path, root: Path) -> bool:
    """Check whether a path resolves inside a directory."""
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == root or root in resolved.parents


def runs_isaac_sim_python(
    isaac_sim_path: Path = DEFAULT_ISAAC_SIM_PATH, python_exe: str | None = None, venv_path: str | None = None
) -> bool:
    """Check whether the interpreter about to run Isaac Sim is the package's own Python.

    A virtual environment adds a ``site-packages`` directory but reuses the interpreter it was
    created from, so one created on the Isaac Sim package's Python runs that exact binary and loads
    Kit's extension modules unchanged. Environments that supply their own interpreter and native
    libraries do not qualify: their executable and ``pyvenv.cfg`` point outside the Kit tree.

    Args:
        isaac_sim_path: Isaac Sim installation directory.
        python_exe: Interpreter that will be launched, if known.
        venv_path: Virtual environment root, usually ``VIRTUAL_ENV``.

    Returns:
        Whether the interpreter resolves inside ``isaac_sim_path``. Anything unproven is ``False``
        so the caller keeps rejecting environments it cannot verify.
    """
    try:
        root = isaac_sim_path.resolve()
    except OSError:
        return False

    # ``uv venv`` symlinks ``bin/python`` at its base interpreter; resolving it is the direct answer.
    if python_exe and _is_within(Path(python_exe), root):
        return True
    # Fall back to the recorded base for environments whose interpreter is a copy, not a symlink.
    if venv_path:
        home = _pyvenv_home(Path(venv_path))
        if home is not None and _is_within(home, root):
            return True
    return False


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


_CMD_METACHARACTERS = frozenset("<>|&^")


def _escape_for_cmd_exe(cmd: list[str] | tuple[str, ...]) -> list[str]:
    """Wrap .bat/.cmd calls in cmd.exe /c so args with < > | & ^ stay literal.

    Uses cmd.exe caret-escaping (``^<``) for metacharacters in args without
    whitespace; double-quotes args that contain whitespace. Avoids wrapping a
    metacharacter-bearing arg in double quotes -- ``cmd.exe /c "...\"X<Y\""``
    leaks the literal quotes through to the inner program because cmd.exe and
    Python's subprocess.list2cmdline don't share a quoting convention, and
    pip then sees the literal ``"setuptools<82.0.0"`` and rejects it.
    """
    # only .bat/.cmd needs wrapping
    exe = str(cmd[0]).lower()
    if not (exe.endswith(".bat") or exe.endswith(".cmd")):
        return list(cmd)

    parts: list[str] = []
    for arg in cmd:
        s = str(arg)
        has_meta = any(c in s for c in _CMD_METACHARACTERS)
        has_space = " " in s or "\t" in s
        # Args with spaces fall back to double-quoting: cmd.exe does not
        # interpret metacharacters inside "..." but the literal quotes can
        # leak through the python.bat hop. Bypass python.bat entirely (see
        # extract_python_exe) for the common case; pip args with spaces and
        # metacharacters in the same token are not currently used.
        if has_space:
            parts.append(f'"{s}"')
        elif has_meta:
            parts.append("".join(f"^{c}" if c in _CMD_METACHARACTERS else c for c in s))
        else:
            parts.append(s)
    return ["cmd.exe", "/c", " ".join(parts)]


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

    print_debug(f'run_command(): CWD: "{cwd}"')
    print_debug(f'run_command(): CMD: "{command_str}"')
    _print_debug_env("run_command()", env)

    # On Windows, escape cmd.exe metacharacters when invoking .bat/.cmd files.
    if isinstance(cmd, (list, tuple)) and is_windows():
        cmd = _escape_for_cmd_exe(cmd)

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
    except subprocess.CalledProcessError as error:
        print_error(f'Command failed with code {error.returncode}: "{command_str}"')
        sys.exit(error.returncode)
    except KeyboardInterrupt:
        sys.exit(130)


def extract_python_exe() -> str:
    """Return the interpreter running the uv-installed CLI."""
    return sys.executable


def extract_isaacsim_path(*, required: bool = True) -> Path | None:
    """Find the Isaac Sim installation path.

    Args:
        required: When ``True`` (default), exit the process if Isaac Sim
            cannot be found.  When ``False``, return ``None`` instead.
    """
    # Use the sym-link path to Isaac Sim directory.
    isaacsim_path = DEFAULT_ISAAC_SIM_PATH
    # If above path is not available, try to find the path using python.
    if not isaacsim_path.exists():
        # Use the current interpreter to probe for isaacsim — avoids a recursive extract_python_exe call.
        # Retrieve the path importing isaac sim and getting the environment path.
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import isaacsim"],
                capture_output=True,
                text=True,
                check=False,
                # avoid EULA prompt
                stdin=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                # Helper to print env var.
                cmd = [sys.executable, "-c", "import isaacsim; import os; print(os.environ['ISAAC_PATH'])"]
                res = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    # avoid EULA prompt
                    stdin=subprocess.DEVNULL,
                )
                if res.returncode == 0:
                    output = res.stdout.strip()
                    if output:
                        isaacsim_path = Path(output)
        except Exception:
            pass

    if not isaacsim_path.exists():
        if not required:
            return None
        print_error(f"Unable to find the Isaac Sim directory: '{isaacsim_path}'")
        print("\tThis could be due to the following reasons:")
        print("\t1. Conda environment is not activated.")
        print("\t2. Isaac Sim package is not installed.")
        print(f"\t3. Isaac Sim directory is not available at the default path: {DEFAULT_ISAAC_SIM_PATH}")
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
            result = run_command(
                [python_exe, "-c", "import isaacsim"],
                capture_output=True,
                text=True,
                check=False,
                # avoid EULA prompt
                stdin=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                # Isaac Sim - Python packages entry point.
                return ["isaacsim", "isaacsim.exp.full"]
        except Exception:
            pass
        print_error(f"No Isaac Sim executable found at path: {isaacsim_path}")
        sys.exit(1)

    return [str(isaacsim_exe)]


def _aarch64_libgomp_env(env: dict[str, str] | None) -> dict[str, str] | None:
    """Preload the system OpenMP runtime for python subprocesses on Linux aarch64.

    The torch wheel bundles its own libgomp, which loads first and conflicts with the
    library Isaac Sim expects, so isaacsim refuses to start unless the system libgomp is
    preloaded. The pip installation docs tell users to export LD_PRELOAD by hand; doing it
    here makes first runs through the CLI work out of the box. isaacsim only accepts the
    system ``/lib/*/libgomp.so.1`` paths listed verbatim, so those full paths are prepended.
    Returns the env unchanged on other platforms, without a system libgomp, or when every
    such path is already preloaded.
    """
    if platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64"):
        return env
    merged = dict(os.environ if env is None else env)
    preload = [entry for entry in merged.get("LD_PRELOAD", "").split(":") if entry]
    missing = [path for path in sorted(glob.glob("/lib/*/libgomp.so.1")) if path not in preload]
    if not missing:
        return env
    merged["LD_PRELOAD"] = ":".join(missing + preload)
    return merged


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

    python_exe = extract_python_exe()
    cmd = [python_exe]

    # A source build linked at ``_isaac_sim`` must load its live Kit and extension paths, but the
    # dependencies managed by uv should still come from the active environment. Isaac Sim's Python
    # launcher supports this through its ``PYTHONEXE`` override. Already configured
    # environments must not source the same runtime twice.
    command_env = os.environ if env is None else env
    configured_isaac_path = command_env.get("ISAAC_PATH")
    local_sim = DEFAULT_ISAAC_SIM_PATH
    python_launcher = local_sim / ("python.bat" if is_windows() else "python.sh")
    using_virtual_environment = bool(command_env.get("VIRTUAL_ENV") or sys.prefix != sys.base_prefix)
    if (
        local_sim.is_dir()
        and python_launcher.is_file()
        and not is_isaac_sim_source_build(local_sim)
        and using_virtual_environment
        and not runs_isaac_sim_python(local_sim, python_exe, command_env.get("VIRTUAL_ENV"))
    ):
        print_error("Downloaded Isaac Sim packages cannot be combined with a Python virtual environment.")
        print_error(
            "Create the uv environment on "
            f"that Python ('uv venv --python {local_sim / 'kit' / 'python' / 'bin' / 'python3'}'), or "
            "remove '_isaac_sim' and install Isaac Sim from pip in the virtual environment."
        )
        raise SystemExit(1)
    isaac_env_active = (
        configured_isaac_path is not None and Path(configured_isaac_path).resolve() == local_sim.resolve()
    )
    if local_sim.is_dir() and python_launcher.is_file() and not isaac_env_active:
        env = dict(command_env)
        env["PYTHONEXE"] = python_exe
        source_paths = [
            local_sim / "python_packages",
            local_sim / "exts" / "isaacsim.simulation_app",
            local_sim / "kit" / "kernel" / "py",
            local_sim / "kit" / "plugins" / "bindings-python",
            Path(site.getsitepackages()[0]),
        ]
        existing_pythonpath = env.get("PYTHONPATH")
        python_paths = [str(path) for path in source_paths if path.is_dir()]
        if existing_pythonpath:
            python_paths.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        cmd = [str(python_launcher)]

    if is_module:
        cmd.append("-m")
    else:
        script_or_module = _PYTHON_SCRIPT_ALIASES.get(str(script_or_module), script_or_module)

    cmd.append(str(script_or_module))
    cmd.extend(args)

    command_str = " ".join(str(part) for part in cmd)

    print_debug(f'run_python_command(): CWD: "{os.getcwd()}"')
    print_debug(f'run_python_command(): CMD: "{command_str}"')

    return run_command(
        cmd,
        cwd=os.getcwd(),
        env=_aarch64_libgomp_env(env),
        check=check,
    )
