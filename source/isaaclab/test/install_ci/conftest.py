# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isolated uv environments for installation contracts."""

import os
import platform
import shutil
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import tomllib


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-gpu", action="store_true", help="Run GPU training after installation.")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--run-gpu"):
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="pass --run-gpu to exercise GPU training"))


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
