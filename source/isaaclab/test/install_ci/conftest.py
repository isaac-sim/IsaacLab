# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared pytest fixtures and configuration for installation CI tests."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import tomllib

# Make ``utils`` (sibling of this conftest) importable from tests living in
# subdirectories (``cli/``, ``misc/``).  Required because ``pytest.ini`` uses
# ``--import-mode=importlib``, which does NOT add the conftest's directory to
# ``sys.path``.
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import utils as _utils  # noqa: E402
from utils import find_isaaclab_root, run_cmd  # noqa: E402, F401 – re-exported for tests

_CYAN_BRIGHT = "\033[96m"
_RESET = "\033[0m"

_EXECUTION_ENVIRONMENT_KEY = pytest.StashKey[_utils.ExecutionEnvironment]()


# Fixtures


@pytest.fixture(scope="session")
def isaaclab_root() -> Path:
    """Resolved absolute path to the IsaacLab repository root."""
    return find_isaaclab_root()


@pytest.fixture(scope="session")
def uv_overrides() -> Path:
    """Path to the self-contained uv overrides used by wheel installation tests."""
    return Path(__file__).resolve().parent / "uv_pip" / "uv-overrides.txt"


@pytest.fixture(scope="session")
def cartpole_smoke_script() -> Path:
    """Path to the shared Cartpole smoke probe executed inside installed environments."""
    return Path(__file__).resolve().parent / "misc" / "cartpole_training_smoke.py"


@pytest.fixture
def tmp_venv(tmp_path: Path):
    """Create a temporary Python virtual-environment and tear it down after the test.

    Yields a dict with:
        ``path``  – Path to the venv directory
        ``python`` – Path to the venv's python executable
        ``pip``    – Path to the venv's pip executable
    """
    venv_dir = tmp_path / "venv"
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])

    if platform.system() == "Windows":
        python_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
    else:
        python_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"

    # Upgrade pip inside the venv to avoid old-pip issues
    subprocess.check_call([str(pip_exe), "install", "--upgrade", "pip"], timeout=120)

    yield {"path": venv_dir, "python": python_exe, "pip": pip_exe}

    # Cleanup is handled by tmp_path (pytest removes it automatically)


@pytest.fixture(scope="session")
def wheel_path() -> Path | None:
    """Path to a pre-built isaaclab wheel, or None.

    Set the ``ISAACLAB_WHEEL`` environment variable to the wheel file path
    before running tests.
    """
    value = os.environ.get("ISAACLAB_WHEEL")
    if value:
        p = Path(value).resolve()
        if not p.exists():
            pytest.fail(f"ISAACLAB_WHEEL points to non-existent file: {p}")
        return p
    return None


@pytest.fixture(scope="session")
def checkout() -> Path:
    return Path(__file__).resolve().parents[4]


@pytest.fixture(scope="session")
def run() -> Callable[..., str]:
    """Run with bounded capture and no inherited Python environment or package indexes."""
    clean_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("UV_", "PIP_", "CONDA_"))
        and key
        not in (
            "VIRTUAL_ENV",
            "PYTHONPATH",
            "PYTHONHOME",
            "ISAAC_PATH",
            "ISAACSIM_PATH",
            "CARB_APP_PATH",
            "EXP_PATH",
            "PXR_PLUGINPATH_NAME",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
        )
    }
    clean_env["OMNI_KIT_ACCEPT_EULA"] = "yes"
    if platform.machine().lower() in ("aarch64", "arm64"):
        clean_env["LD_PRELOAD"] = "/lib/aarch64-linux-gnu/libgomp.so.1"

    def execute(*args: str, cwd: Path, env: dict[str, str] | None = None, timeout: int = 900) -> str:
        command_env = {**clean_env, **(env or {})}
        # Keep platform-required libraries when a probe adds another preload.
        preloads = [clean_env.get("LD_PRELOAD", ""), (env or {}).get("LD_PRELOAD", "")]
        if any(preloads):
            command_env["LD_PRELOAD"] = ":".join(preload for preload in preloads if preload)
        result = subprocess.run(
            args,
            cwd=cwd,
            env=command_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        assert result.returncode == 0, f"{args} exited with {result.returncode}:\n{result.stdout}"
        return result.stdout

    return execute


@pytest.fixture(scope="session")
def wheel() -> Path:
    value = os.environ.get("ISAACLAB_WHEEL")
    if not value:
        pytest.fail("Supply ISAACLAB_WHEEL or use tools/run_install_ci.py --build-wheel.")
    path = Path(value).resolve()
    assert path.is_file(), f"Wheel not found: {path}"
    return path


@pytest.fixture(scope="session", params=["base", "isaacsim"])
def workspace(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, checkout: Path, run: Callable[..., str]
) -> Iterator[tuple[str, Path, dict[str, str]]]:
    """Each supported runtime installs once, then shares its environment with probes."""
    assert not (checkout / "_isaac_sim").exists(), "Run installation CI from a checkout without a local Kit build"
    directory = tmp_path_factory.mktemp(request.param)
    env = {"UV_PROJECT_ENVIRONMENT": str(directory / "venv")}
    extras = ["--extra", "isaacsim"] if request.param == "isaacsim" else []
    # Validate the committed lock rather than silently resolving or rewriting it.
    run("uv", "sync", "--locked", *extras, cwd=checkout, env=env, timeout=4500)
    yield request.param, directory, env
    shutil.rmtree(directory / "venv")


@pytest.fixture(scope="session", params=["base", "isaacsim"])
def installed_wheel(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    checkout: Path,
    wheel: Path,
    run: Callable[..., str],
) -> Iterator[tuple[str, Path, Path]]:
    directory = tmp_path_factory.mktemp(f"wheel-{request.param}")
    python = directory / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run("uv", "venv", "--python", "3.12", str(directory / "venv"), cwd=directory)
    requirement = str(wheel) + ("[isaacsim]" if request.param == "isaacsim" else "")
    run(
        "uv",
        "--no-config",
        "pip",
        "install",
        "--python",
        str(python),
        requirement,
        "--overrides",
        str(checkout / "tools/wheel_builder/uv-overrides.txt"),
        "--extra-index-url",
        "https://pypi.nvidia.com",
        "--index-strategy",
        "unsafe-best-match",
        cwd=directory,
        timeout=4500,
    )
    # Wheel consumers select their CUDA index explicitly; project sources do not propagate.
    with (checkout / "pyproject.toml").open("rb") as file:
        versions = tomllib.load(file)["tool"]["isaaclab"]["versions"]
    run(
        "uv",
        "--no-config",
        "pip",
        "install",
        "--python",
        str(python),
        f"torch=={versions['torch']}",
        f"torchvision=={versions['torchvision']}",
        "--reinstall-package",
        "torch",
        "--reinstall-package",
        "torchvision",
        "--index-url",
        "https://download.pytorch.org/whl/cu130",
        cwd=directory,
        timeout=4500,
    )
    yield request.param, directory, python
    shutil.rmtree(directory / "venv")


# Markers


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--noskip",
        "--force",
        "--dontskip",
        action="store_true",
        default=False,
        dest="noskip",
        help="Strip all @pytest.mark.skip markers for this run (does not affect skipif). Alias: --force.",
    )
    parser.addoption("--run-gpu", action="store_true", help="Run GPU training after installation.")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "smoke: tests for core installation, task, and RL functionality")
    config.addinivalue_line("markers", "bug: bug-regression tests (use bug id as argument)")
    config.addinivalue_line("markers", "gpu: tests that require a GPU")
    config.addinivalue_line("markers", "docker: tests that only run inside Docker")
    config.addinivalue_line("markers", "native: tests that only run natively (not in Docker)")
    config.addinivalue_line("markers", "slow: tests that take a long time")
    config.addinivalue_line("markers", "uv: tests that require the uv package manager")
    config.addinivalue_line("markers", "conda: tests that require the conda package manager")
    config.addinivalue_line("markers", "install_path_cli: tests that exercise the ./isaaclab.sh -i install path")
    config.addinivalue_line(
        "markers", "install_path_uv_pip: tests that exercise the uv pip install <wheel> install path"
    )
    config.addinivalue_line(
        "markers", "install_path_uv_run: tests that exercise the uv run (committed lockfile) install path"
    )
    config.addinivalue_line("markers", "timeout: per-test timeout in seconds")

    try:
        config.stash[_EXECUTION_ENVIRONMENT_KEY] = _utils.detect_execution_environment()
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc

    # Enable real-time output when pytest capture is disabled (-s)
    capture = config.getoption("capture", default="fd")
    _utils.stream_output = capture == "no"


def pytest_report_header(config: pytest.Config) -> str:
    """Show the detected install_ci execution environment in the test header."""
    return f"install_ci execution environment: {config.stash[_EXECUTION_ENVIRONMENT_KEY]}"


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Print a newline after the PASSED/FAILED/SKIPPED result."""
    if report.when == "call" or (report.when == "setup" and report.skipped):
        sys.stdout.write("\n")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Map dynamic bug markers and deselect items with mismatched env markers.

    This allows filtering by bug ID natively in pytest: `-m "<bug-id>"`
    instead of the (unsupported natively) `-m "bug('<bug-id>')"`.

    Tests whose ``docker``/``native`` marker doesn't match the current
    execution environment are *deselected* (removed from collection) so they
    don't appear in ``--collect-only`` output or skew skipped-count metrics.
    """
    execution_environment = config.stash[_EXECUTION_ENVIRONMENT_KEY]
    known_bugs = set()
    for item in items:
        for mark in item.iter_markers(name="bug"):
            for arg in mark.args:
                if isinstance(arg, str):
                    known_bugs.add(arg)

    for bug in known_bugs:
        config.addinivalue_line("markers", f"{bug}: dynamically generated bug marker")

    remaining: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        for mark in item.iter_markers(name="bug"):
            for arg in mark.args:
                if isinstance(arg, str):
                    item.add_marker(arg)

        marker_names = {mark.name for mark in item.iter_markers()}
        try:
            skip_reason = _utils.get_execution_environment_skip_reason(marker_names, execution_environment)
        except ValueError as exc:
            raise pytest.UsageError(f"{item.nodeid}: {exc}") from exc

        if skip_reason:
            deselected.append(item)
        else:
            remaining.append(item)

    if deselected:
        items[:] = remaining
        config.hook.pytest_deselected(items=deselected)

    if not config.getoption("--run-gpu"):
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="pass --run-gpu to exercise GPU training"))

    if config.getoption("--noskip"):
        for item in items:
            item.own_markers = [m for m in item.own_markers if m.name != "skip"]
