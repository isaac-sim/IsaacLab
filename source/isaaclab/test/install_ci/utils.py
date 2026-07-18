# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared utilities for Isaac Lab installation CI tests."""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

_DIM = "\033[2m"
_MAGENTA = "\033[95m"
_RESET = "\033[0m"

# Controls whether run_cmd() streams output by default.
# Set to True by conftest.py when pytest runs with -s / --capture=no.
stream_output: bool = False

# ISAACLAB_INSTALL_CI_ENV can be set to override execution
# environment detection in install_ci tests
# (for testing the testing while testing).

ExecutionEnvironment = Literal["docker", "native"]


def detect_execution_environment(
    environ: dict[str, str] | None = None,
    filesystem_root: Path | None = None,
) -> ExecutionEnvironment:
    """Detect whether install_ci tests are running in Docker or natively."""
    env = environ if environ is not None else os.environ
    root = filesystem_root if filesystem_root is not None else Path("/")

    override = env.get("ISAACLAB_INSTALL_CI_ENV")
    if override:
        cleaned = override.strip().lower()
        if cleaned not in ("docker", "native"):
            raise ValueError(f"ISAACLAB_INSTALL_CI_ENV must be 'docker' or 'native', got: {override!r}")
        return cleaned  # type: ignore[return-value]

    if (root / ".dockerenv").exists() or (root / "run" / ".containerenv").exists():
        return "docker"

    for cgroup_path in (root / "proc" / "1" / "cgroup", root / "proc" / "self" / "cgroup"):
        try:
            cgroup_text = cgroup_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(
            hint in cgroup_text
            for hint in (
                "docker",
                "containerd",
                "kubepods",
                "libpod",
                "podman",
            )
        ):
            return "docker"

    if env.get("container"):
        return "docker"

    return "native"


def get_execution_environment_skip_reason(
    marker_names: set[str],
    execution_environment: ExecutionEnvironment,
) -> str | None:
    """Return a skip reason when environment markers do not match the runtime."""
    has_docker = "docker" in marker_names
    has_native = "native" in marker_names

    if has_docker and has_native:
        raise ValueError("tests cannot be marked with both 'docker' and 'native'")

    if has_docker and execution_environment != "docker":
        return f"requires Docker execution environment, detected {execution_environment}"

    if has_native and execution_environment != "native":
        return f"requires native execution environment, detected {execution_environment}"

    return None


def find_isaaclab_root() -> Path:
    """Walk up from this file to find the repo root (contains isaaclab.sh)."""
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "isaaclab.sh").exists():
            return parent
    raise FileNotFoundError("Could not locate IsaacLab repository root (no isaaclab.sh found)")


def run_cmd(
    args: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 600,
    err_on_err: bool = False,
    stream: bool | None = None,
) -> subprocess.CompletedProcess:
    """Run a command, merging *env* into the current environment.

    Args:
        args: Command and arguments to run.
        cwd: Working directory for the subprocess.
        env: Extra environment variables merged into the current environment.
        timeout: Timeout in seconds.
        err_on_err: Raise CalledProcessError on non-zero exit.
        stream: When True, stream stdout/stderr to the console in
            real time instead of capturing them.  Defaults to True when
            pytest is invoked with ``-s`` (``--capture=no``).

    Returns:
        The CompletedProcess; raises CalledProcessError when *check* is
        True and return code != 0.
    """
    if stream is None:
        stream = stream_output
    merged_env = {**os.environ, **(env or {})}
    cmd_str = " ".join(str(a) for a in args)
    if stream:
        sys.stdout.write(f"{_MAGENTA}[COMMAND] {cmd_str}{_RESET}\n")
        sys.stdout.flush()
        # Stream output to console in real time.
        t0 = time.monotonic()
        proc = subprocess.Popen(
            [str(a) for a in args],
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        lines: list[str] = []
        try:
            for line in proc.stdout:
                lines.append(line)
                sys.stdout.write(f"{_DIM}{line}{_RESET}")
                sys.stdout.flush()
        except Exception:
            proc.kill()
            raise
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        elapsed = time.monotonic() - t0
        sys.stdout.write(f"{_MAGENTA}[{elapsed:.1f}s]{_RESET}\n")
        sys.stdout.flush()
        result = subprocess.CompletedProcess(
            args=proc.args,
            returncode=proc.returncode,
            stdout="".join(lines),
            stderr="",
        )
        if err_on_err and result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
        return result
    return subprocess.run(
        [str(a) for a in args],
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=err_on_err,
    )


_IS_WINDOWS = platform.system() == "Windows"


def cuda_torch_index_url() -> str:
    """Return the documented CUDA-matched torch wheel index URL for the current platform.

    Mirrors ``docs/source/setup/installation/pip_installation.rst``:
    ``cu130`` on aarch64 (e.g. GB10 / DGX Spark, which expose CUDA capability 12.x),
    ``cu128`` on x86_64.
    """
    if platform.machine().lower() in ("aarch64", "arm64"):
        return "https://download.pytorch.org/whl/cu130"
    return "https://download.pytorch.org/whl/cu128"


def aarch64_isaacsim_env() -> dict[str, str]:
    """Return env vars required to import ``isaacsim`` from a ``uv pip`` install on aarch64.

    Without ``LD_PRELOAD``ing the system ``libgomp.so.1``, ``isaacsim`` refuses to load on
    aarch64 because the torch wheel ships its own OpenMP that loads first and conflicts.
    The same workaround is recommended to users in the install docs. Returns an empty
    dict on non-aarch64 hosts.
    """
    if platform.machine().lower() in ("aarch64", "arm64"):
        libgomp = "/lib/aarch64-linux-gnu/libgomp.so.1"
        if os.path.exists(libgomp):
            return {"LD_PRELOAD": libgomp}
    return {}


class UV_Mixin:
    """Mixin providing uv virtual-environment helpers for test classes."""

    env_path: Path
    python: Path
    cli_script: Path

    def create_uv_env(self, isaaclab_root: Path, env_name: str = "") -> None:
        """Create a uv environment and store info on self.

        Sets ``self.env_path``, ``self.python``, and ``self.cli_script``.

        Args:
            isaaclab_root: Path to the IsaacLab repository root.
            env_name: Name for the venv directory. A random name is
                generated when empty.
        """
        env_name = env_name if env_name else f"_isaaclab_install_ci_{os.urandom(4).hex()}"

        self.env_path = isaaclab_root / env_name
        self.cli_script = isaaclab_root / ("isaaclab.bat" if _IS_WINDOWS else "isaaclab.sh")

        if self.env_path.exists():
            shutil.rmtree(self.env_path)

        result = run_cmd([str(self.cli_script), "-u", env_name], cwd=isaaclab_root, err_on_err=False)
        assert result.returncode == 0, f"uv env creation failed:\n{result.stdout}\n{result.stderr}"
        assert self.env_path.exists(), f"Expected env directory {self.env_path} was not created"

        # Prevent the venv from being tracked by git.
        (self.env_path / ".gitignore").write_text("*\n")

        self.python = (self.env_path / "Scripts" / "python.exe") if _IS_WINDOWS else (self.env_path / "bin" / "python")
        assert self.python.exists(), f"Python executable not found at {self.python}"

    def destroy_uv_env(self) -> None:
        """Remove the uv environment directory if it exists."""
        if hasattr(self, "env_path") and self.env_path.exists():
            shutil.rmtree(self.env_path)

    def run_in_uv_env(self, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        """Run a command inside the activated venv by sourcing the activate script.

        Args:
            cmd: Command and arguments to run inside the venv.
            **kwargs: Extra keyword arguments forwarded to :func:`run_cmd`.
        """
        escaped = " ".join(shlex.quote(str(a)) for a in cmd)
        if _IS_WINDOWS:
            activate = str(self.env_path / "Scripts" / "activate.bat")
            shell_cmd = f'call "{activate}" && {escaped}'
            return run_cmd(["cmd", "/c", shell_cmd], **kwargs)
        else:
            activate = shlex.quote(str(self.env_path / "bin" / "activate"))
            shell_cmd = f"source {activate} && {escaped}"
            return run_cmd(["bash", "-c", shell_cmd], **kwargs)


def drop_keys(env: dict[str, str], keys: tuple[str, ...]) -> dict[str, str]:
    """Return a copy of *env* with the given keys removed.

    Useful for stripping venv/conda activation markers from the environment
    before creating a fresh isolated environment inside a test.

    Args:
        env: Source environment dictionary (e.g. ``os.environ``).
        keys: Variable names to exclude from the returned copy.
    """
    return {k: v for k, v in env.items() if k not in keys}


class Conda_Mixin:
    """Mixin providing conda virtual-environment helpers for test classes.

    Pair with :class:`UVMixin` or use standalone for conda-only tests.
    Requires ``conda`` (or ``mamba``) to be on ``PATH``.

    Usage::

        class TestFoo(Conda_Mixin, unittest.TestCase):
            def setUp(self):
                self.create_conda_env(find_isaaclab_root())

            def tearDown(self):
                self.destroy_conda_env()
    """

    @property
    def cli_script(self) -> Path:
        """Path to ``isaaclab.sh`` (or ``.bat``) inside the repository root."""
        root: Path = self._isaaclab_root  # type: ignore[attr-defined]
        return root / ("isaaclab.bat" if _IS_WINDOWS else "isaaclab.sh")

    def create_conda_env(self, isaaclab_root: Path, env_name: str = "", python_version: str = "3.12") -> None:
        """Create an isolated conda environment.

        Args:
            isaaclab_root: Path to the IsaacLab repository root.
            env_name: Name for the conda environment. A unique name based on the
                test id is generated automatically when left empty.
            python_version: Python version to install in the environment.
        """
        if not env_name:
            import uuid

            env_name = f"isaaclab_ci_{uuid.uuid4().hex[:8]}"
        self._conda_env_name = env_name
        self._isaaclab_root = isaaclab_root

        result = run_cmd(
            ["conda", "create", "-n", env_name, f"python={python_version}", "-y", "--quiet"],
            env=drop_keys(dict(os.environ), ("VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "CONDA_PREFIX")),
        )
        assert result.returncode == 0, f"conda env creation failed:\n{result.stdout}\n{result.stderr}"

        # Resolve the python executable inside the new conda env.
        conda_prefix_result = run_cmd(
            ["conda", "run", "-n", env_name, "python", "-c", "import sys; print(sys.executable)"],
            env=drop_keys(dict(os.environ), ("VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "CONDA_PREFIX")),
        )
        assert conda_prefix_result.returncode == 0, (
            f"Could not resolve conda python:\n{conda_prefix_result.stdout}\n{conda_prefix_result.stderr}"
        )
        self.python = Path(conda_prefix_result.stdout.strip())

    def destroy_conda_env(self) -> None:
        """Remove the conda environment created by :meth:`create_conda_env`."""
        if hasattr(self, "_conda_env_name"):
            run_cmd(["conda", "env", "remove", "-n", self._conda_env_name, "-y", "--quiet"])

    def run_in_conda_env(self, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        """Run *cmd* inside the conda environment.

        Args:
            cmd: Command and arguments to execute.
            **kwargs: Extra keyword arguments forwarded to :func:`run_cmd`.
        """
        escaped = " ".join(shlex.quote(str(a)) for a in cmd)
        shell_cmd = f"conda run -n {shlex.quote(self._conda_env_name)} --no-capture-output {escaped}"
        return run_cmd(["bash", "-c", shell_cmd], **kwargs)
