# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import platform
import shutil
import site
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, Any

from isaaclab.paths import ISAACLAB_ROOT

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
    libraries, such as conda, do not qualify: a conda environment has no ``pyvenv.cfg``, and a
    virtual environment layered on one records that interpreter instead.

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


def _print_labeled(label: str, color: str, message: str, stream: IO[str]) -> None:
    """Print a message behind a label, colorized when the stream is a color-capable terminal."""
    if not (os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb") and getattr(stream, "isatty", bool)():
        label = f"{color}{label}{_ANSI_COLOR_RESET}"
    print(f"{label} {message}", file=stream)


def print_info(message: str, stream: IO[str] = sys.stdout) -> None:
    """Print informational message.

    Args:
        message: Message text to print.
        stream: Output stream where the message is written.
    """
    _print_labeled("[INFO]", _ANSI_COLOR_INFO, message, stream)


def print_warning(message: str, stream: IO[str] = sys.stdout) -> None:
    """Print warning message.

    Args:
        message: Message text to print.
        stream: Output stream where the message is written.
    """
    _print_labeled("[WARNING]", _ANSI_COLOR_WARNING, message, stream)


def print_error(message: str, stream: IO[str] = sys.stderr) -> None:
    """Print error message.

    Args:
        message: Message text to print.
        stream: Output stream where the message is written.
    """
    _print_labeled("[ERROR]", _ANSI_COLOR_ERROR, message, stream)


def print_debug(message: str, stream: IO[str] = sys.stdout) -> None:
    """Print debug message, when debugging is enabled.

    Args:
        message: Message text to print.
        stream: Output stream where the message is written.
    """
    if os.environ.get("DEBUG") == "1":
        _print_labeled("[DEBUG]", _ANSI_COLOR_DEBUG, message, stream)


def _print_debug_env(prefix: str, env: dict[str, str] | None) -> None:
    """Print the variables that differ between ``env`` and ``os.environ`` for debugging.

    Args:
        prefix: Prefix identifying the caller function in debug output.
        env: Environment to compare against os.environ.
    """
    added = {key: value for key, value in (env or {}).items() if key not in os.environ}
    changed = {
        key: {"from": os.environ[key], "to": value}
        for key, value in (env or {}).items()
        if key in os.environ and os.environ[key] != value
    }
    removed = [key for key in os.environ if key not in env] if env is not None else []
    if not (added or changed or removed):
        print_debug(f"{prefix}: ENV: <os.environ>")
        return
    if added:
        print_debug(f"{prefix}: ENV added: {added}")
    if changed:
        print_debug(f"{prefix}: ENV changed: {changed}")
    if removed:
        print_debug(f"{prefix}: ENV removed: {removed}")


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
    retry_attempts: int = 1,
    retry_delay_seconds: float = 3.0,
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
        retry_attempts: Total number of attempts for a failed command.
        retry_delay_seconds: Delay between attempts [s].
        **kwargs: Additional keyword arguments forwarded to ``subprocess.run``.

    Returns:
        Result object returned by ``subprocess.run``.
    """
    if retry_attempts < 1:
        raise ValueError("retry_attempts must be at least 1")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be non-negative")

    if cwd is None:
        cwd = ISAACLAB_ROOT

    command_str = " ".join(str(part) for part in cmd) if isinstance(cmd, (list, tuple)) else str(cmd)

    # Print some debug info.
    print_debug(f'run_command(): CWD: "{cwd}"')
    print_debug(f'run_command(): CMD: "{command_str}"')
    _print_debug_env("run_command()", env)

    # On Windows, escape cmd.exe metacharacters when invoking .bat/.cmd files.
    if isinstance(cmd, (list, tuple)) and is_windows():
        cmd = _escape_for_cmd_exe(cmd)

    for attempt in range(1, retry_attempts + 1):
        try:
            result = subprocess.run(
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
            returncode = error.returncode
            result = None
        except KeyboardInterrupt:
            sys.exit(130)
        else:
            returncode = result.returncode

        if returncode == 0 or attempt == retry_attempts:
            if result is not None:
                return result
            print_error(f'Command failed with code {returncode}: "{command_str}"')
            sys.exit(returncode)

        print_warning(
            f"Command failed with code {returncode}; retrying in {retry_delay_seconds:g} seconds "
            f'(attempt {attempt + 1}/{retry_attempts}): "{command_str}"'
        )
        time.sleep(retry_delay_seconds)

    raise AssertionError("unreachable")


def _is_virtualenv_python(python_exe: str | Path) -> bool:
    """Check whether a Python executable belongs to a virtual environment.

    Args:
        python_exe: Python executable path.

    Returns:
        True when the executable is inside a Python virtual environment.
    """
    python_path = Path(python_exe)
    return (python_path.parent.parent / "pyvenv.cfg").is_file()


def get_pip_command(python_exe: str | None = None) -> list[str]:
    """Return the base pip command tokens for the current environment.

    When ``uv`` is available and a virtual environment is active, returns
    ``["uv", "pip"]``.  When the target Python belongs to a virtual
    environment, ``UV_PYTHON`` is set so ``uv pip`` installs into that
    environment even if the process itself is not activated.  Otherwise returns
    ``[python_exe, "-m", "pip"]`` so that the target interpreter's own pip is
    used (e.g. Isaac Sim's bundled ``python.sh``).

    Args:
        python_exe: Python executable path.  Resolved via
            :func:`extract_python_exe` when ``None``.
    """
    if python_exe is None:
        python_exe = extract_python_exe()

    in_venv = bool(os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX") or (sys.prefix != sys.base_prefix))
    if shutil.which("uv") and (in_venv or _is_virtualenv_python(python_exe)):
        os.environ["UV_PYTHON"] = python_exe
        return ["uv", "pip"]

    return [python_exe, "-m", "pip"]


def _python_in_prefix(prefix: str | Path, windows_subdir: str = "Scripts") -> Path | None:
    """Return the interpreter of a virtual or conda environment prefix, if one exists.

    Args:
        prefix: Environment root directory.
        windows_subdir: Directory holding ``python.exe`` on Windows (``Scripts`` for venvs, ``""`` for conda).

    Returns:
        The first existing interpreter path, or ``None``.
    """
    prefix = Path(prefix)
    if is_windows():
        candidates = [prefix / windows_subdir / "python.exe" if windows_subdir else prefix / "python.exe"]
    else:
        candidates = [prefix / "bin" / "python", prefix / "bin" / "python3"]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _kit_python(isaacsim_path: Path) -> Path:
    """Return the Python launcher of an Isaac Sim installation."""
    if not is_windows():
        return isaacsim_path / "python.sh"
    # Prefer the underlying python.exe over python.bat to avoid cmd.exe metacharacter-quoting hazards on
    # pip args like ``setuptools<82.0.0``; isaaclab.bat already sourced setup_conda_env.bat for children.
    kit_python_exe = isaacsim_path / "kit" / "python" / "python.exe"
    return kit_python_exe if kit_python_exe.exists() else isaacsim_path / "python.bat"


def extract_python_exe() -> str:
    """Find the Python executable to use.

    Candidates are tried in order: the active virtual environment, the active conda environment, the
    running interpreter when it belongs to a virtual environment, repo-local environments, Isaac Sim's
    bundled Python, and finally a system Python 3.12.
    """
    python_exe: Path | None = None

    venv_prefix = os.environ.get("VIRTUAL_ENV")
    if venv_prefix:
        print_debug(f"extract_python_exe(): Found VIRTUAL_ENV: {venv_prefix}")
        python_exe = _python_in_prefix(venv_prefix)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if python_exe is None and conda_prefix:
        print_debug(f"extract_python_exe(): Found CONDA_PREFIX: {conda_prefix}")
        python_exe = _python_in_prefix(conda_prefix, windows_subdir="")

    if python_exe is None and sys.prefix != sys.base_prefix:
        python_exe = Path(sys.executable)
        print_debug(f"extract_python_exe(): Using active virtual environment python: {python_exe}")

    if python_exe is None:
        for default_venv in (ISAACLAB_ROOT / "env_isaaclab", ISAACLAB_ROOT / ".venv"):
            python_exe = _python_in_prefix(default_venv)
            if python_exe is not None:
                print_debug(f"extract_python_exe(): Found repo-local venv python: {python_exe}")
                break

    if python_exe is None:
        print_debug("extract_python_exe(): Checking for Kit python...")
        isaacsim_path = extract_isaacsim_path(required=False)
        if isaacsim_path is not None:
            python_exe = _kit_python(isaacsim_path)

    if python_exe is None or not python_exe.exists():
        system_python_exe = shutil.which("python3.12") or shutil.which("python") or shutil.which("python3")
        python_exe = Path(system_python_exe) if system_python_exe else None
        print_debug(f"extract_python_exe(): System python candidate: {python_exe}")
        if python_exe is not None and python_exe.exists():
            result = subprocess.run(
                [str(python_exe), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                capture_output=True,
                text=True,
                check=False,
            )
            version = result.stdout.strip()
            if version != "3.12":
                print_error(f"Falling back on system Python {version} ({python_exe}), but 3.12 is required.")
                sys.exit(1)

    if python_exe is None or not python_exe.exists():
        print_error("Unable to find suitable Python executable")
        sys.exit(1)

    print_info(f'Using Python: "{python_exe}"')
    return str(python_exe)


def extract_isaacsim_path(*, required: bool = True) -> Path | None:
    """Find the Isaac Sim installation path.

    Args:
        required: When ``True`` (default), exit the process if Isaac Sim
            cannot be found.  When ``False``, return ``None`` instead.
    """
    isaacsim_path = DEFAULT_ISAAC_SIM_PATH

    # Without the symlink, ask a pip-installed Isaac Sim for its path. The current interpreter is used
    # so the probe does not recurse through extract_python_exe.
    if not isaacsim_path.exists():
        result = subprocess.run(
            [sys.executable, "-c", "import isaacsim, os; print(os.environ['ISAAC_PATH'])"],
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,  # avoid EULA prompt
        )
        if result.returncode == 0 and result.stdout.strip():
            isaacsim_path = Path(result.stdout.strip())

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
    """Find the Isaac Sim executable command."""
    isaacsim_path = extract_isaacsim_path()
    isaacsim_exe = isaacsim_path / ("isaac-sim.bat" if is_windows() else "isaac-sim.sh")
    if isaacsim_exe.exists():
        return [str(isaacsim_exe)]

    # A pip-installed Isaac Sim lives in the current interpreter's environment and ships an entry point.
    result = run_command(
        [sys.executable, "-c", "import isaacsim"],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,  # avoid EULA prompt
    )
    if result.returncode == 0:
        return ["isaacsim", "isaacsim.exp.full"]

    print_error(f"No Isaac Sim executable found at path: {isaacsim_path}")
    sys.exit(1)


def determine_python_version() -> str:
    """Detect Isaac Sim version and return the matching Python version."""
    isaacsim_version = None

    isaacsim_path = extract_isaacsim_path(required=False)
    if isaacsim_path is not None and (isaacsim_path / "VERSION").exists():
        isaacsim_version = (isaacsim_path / "VERSION").read_text().strip() or None

    if isaacsim_version is None:
        try:
            from importlib.metadata import version

            isaacsim_version = version("isaacsim")
        except Exception:
            pass

    if isaacsim_version is None:
        print_warning("Unable to determine Isaac Sim version. Defaulting to python=3.12.")
        return "3.12"

    if isaacsim_version.startswith("5."):
        python_version = "3.11"
    elif isaacsim_version.startswith("6."):
        python_version = "3.12"
    else:
        print_error(f"Unsupported Isaac Sim version: {isaacsim_version}")
        raise RuntimeError(f"Unsupported Isaac Sim version: {isaacsim_version}")

    print_info(f"Detected Isaac Sim {isaacsim_version} -> using python={python_version}")
    return python_version


def _aarch64_libgomp_env(env: dict[str, str] | None) -> dict[str, str] | None:
    """Preload the system OpenMP runtime for python subprocesses on Linux aarch64.

    The torch wheel bundles its own libgomp, which loads first and conflicts with the
    library Isaac Sim expects, so isaacsim refuses to start unless the system libgomp is
    preloaded. The pip installation docs tell users to export LD_PRELOAD by hand; doing it
    here makes first runs through the CLI work out of the box. The bare soname is used so
    ``ld.so`` resolves the library through the ldconfig cache on any distro. Returns the
    env unchanged on other platforms or when a libgomp is already preloaded.
    """
    if platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64"):
        return env
    libgomp = "libgomp.so.1"
    merged = dict(os.environ if env is None else env)
    preload = merged.get("LD_PRELOAD", "")
    if any(os.path.basename(entry) == libgomp for entry in preload.split(":") if entry):
        return env
    merged["LD_PRELOAD"] = f"{libgomp}:{preload}" if preload else libgomp
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
    # launcher supports exactly this combination through its ``PYTHONEXE`` override. The shell
    # wrappers already configure ``ISAAC_PATH`` before starting this CLI, so only direct invocations
    # such as ``uv run isaaclab train`` need to delegate through the launcher here.
    command_env = os.environ if env is None else env
    local_sim = DEFAULT_ISAAC_SIM_PATH
    python_launcher = local_sim / ("python.bat" if is_windows() else "python.sh")
    if local_sim.is_dir() and python_launcher.is_file():
        using_virtual_environment = bool(
            command_env.get("VIRTUAL_ENV")
            or command_env.get("CONDA_PREFIX")
            or sys.prefix != sys.base_prefix
            or _is_virtualenv_python(python_exe)
        )
        if (
            using_virtual_environment
            and not is_isaac_sim_source_build(local_sim)
            and not runs_isaac_sim_python(local_sim, python_exe, command_env.get("VIRTUAL_ENV"))
        ):
            print_error("Downloaded Isaac Sim packages cannot be combined with a Python virtual environment.")
            print_error(
                "Use the bundled Python through isaaclab.sh/isaaclab.bat, create the virtual environment on "
                f"that Python ('uv venv --python {local_sim / 'kit' / 'python' / 'bin' / 'python3'}'), or "
                "remove '_isaac_sim' and install Isaac Sim from pip in the virtual environment."
            )
            raise SystemExit(1)
        configured_isaac_path = command_env.get("ISAAC_PATH")
        isaac_env_active = (
            configured_isaac_path is not None and Path(configured_isaac_path).resolve() == local_sim.resolve()
        )
        if not isaac_env_active:
            env = dict(command_env)
            env["PYTHONEXE"] = python_exe
            source_paths = [
                local_sim / "python_packages",
                local_sim / "exts" / "isaacsim.simulation_app",
                local_sim / "kit" / "kernel" / "py",
                local_sim / "kit" / "plugins" / "bindings-python",
                Path(site.getsitepackages()[0]),
            ]
            python_paths = [str(path) for path in source_paths if path.is_dir()]
            if env.get("PYTHONPATH"):
                python_paths.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = os.pathsep.join(python_paths)
            cmd = [str(python_launcher)]

    if is_module:
        cmd.append("-m")
    else:
        script_or_module = _PYTHON_SCRIPT_ALIASES.get(str(script_or_module), script_or_module)
    cmd.append(str(script_or_module))
    cmd.extend(args)

    return run_command(cmd, cwd=os.getcwd(), env=_aarch64_libgomp_env(env), check=check)
