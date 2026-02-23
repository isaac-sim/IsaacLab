# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import shutil

from ..utils import (
    ISAACLAB_ROOT,
    extract_python_exe,
    is_arm,
    is_windows,
    print_info,
    print_warning,
    run_command,
)
from .misc import command_vscode_settings


def _install_system_deps() -> None:
    """install system dependencies"""
    if is_windows():
        return

    # Check if cmake is already installed.
    if shutil.which("cmake"):
        print_info("cmake is already installed.")
    else:
        print_info("Installing system dependencies...")

        # apt-get update
        cmd = ["apt-get", "update"]
        run_command(["sudo"] + cmd if os.geteuid() != 0 else cmd)

        # apt-get install -y --no-install-recommends cmake build-essential
        cmd = [
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            "cmake",
            "build-essential",
        ]
        run_command(["sudo"] + cmd if os.geteuid() != 0 else cmd)


def _ensure_cuda_torch() -> None:
    """Ensure correct PyTorch and CUDA versions are installed."""
    python_exe = extract_python_exe()

    # Base index for torch.
    base_index = "https://download.pytorch.org/whl"

    # Choose pins per arch.
    torch_ver = "2.9.0"
    tv_ver = "0.24.0"

    if is_arm():
        cuda_ver = "130"
    else:
        cuda_ver = "128"

    cuda_tag = f"cu{cuda_ver}"
    index_url = f"{base_index}/{cuda_tag}"

    want_torch = f"{torch_ver}+{cuda_tag}"

    # Check current torch version (may be empty).
    current_ver = ""
    try:
        # Run python to check torch version.
        result = run_command(
            [
                python_exe,
                "-c",
                "import torch; print(torch.__version__, end='')",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            current_ver = result.stdout.strip()
    except Exception:
        pass

    # Skip install if version is already satisfied.
    if current_ver == want_torch:
        # print(f"[INFO] PyTorch {want_torch} already installed.")
        return

    # Clean install torch.
    print_info(f"Installing torch=={torch_ver} and torchvision=={tv_ver} ({cuda_tag}) from {index_url}...")

    run_command(
        [
            python_exe,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "torch",
            "torchvision",
            "torchaudio",
        ],
        check=False,
    )

    run_command(
        [
            python_exe,
            "-m",
            "pip",
            "install",
            "--index-url",
            index_url,
            f"torch=={torch_ver}",
            f"torchvision=={tv_ver}",
        ]
    )


def _install_isaaclab_extensions() -> None:
    """check if input directory is a python extension and install the module"""
    python_exe = extract_python_exe()
    source_dir = ISAACLAB_ROOT / "source"

    if not source_dir.exists():
        print_warning(f"Source directory not found: {source_dir}")
        return

    # recursively look into directories and install them
    # this does not check dependencies between extensions
    # source directory
    for item in source_dir.iterdir():
        if item.is_dir() and (item / "setup.py").exists():
            print_info(f"Installing extension: {item.name}")
            # If the directory contains setup.py then install the python module.
            run_command(
                [
                    python_exe,
                    "-m",
                    "pip",
                    "install",
                    "--editable",
                    str(item),
                ]
            )


def _install_extra_frameworks(framework_name: str = "all") -> None:
    """install the python packages for supported reinforcement learning frameworks

    Args:
        framework_name: Framework extra to install (for example ``all`` or ``none``).
    """
    python_exe = extract_python_exe()

    extras = ""
    if framework_name != "none":
        extras = f"[{framework_name}]"

    # Check if specified which rl-framework to install.
    if framework_name == "none":
        print_info("No rl-framework will be installed.")
        return

    print_info(f"Installing rl-framework: {framework_name}")

    # Install the learning frameworks specified.
    run_command(
        [
            python_exe,
            "-m",
            "pip",
            "install",
            "-e",
            f"{ISAACLAB_ROOT}/source/isaaclab_rl{extras}",
        ]
    )
    run_command(
        [
            python_exe,
            "-m",
            "pip",
            "install",
            "-e",
            f"{ISAACLAB_ROOT}/source/isaaclab_mimic{extras}",
        ]
    )


def command_install(install_type: str = "all") -> None:
    """
    Install stuff

    Args:
        install_type (str): The RL framework to install ('all', 'none', or specific name).
    """
    # Install system dependencies first.
    _install_system_deps()

    # Install the python packages in IsaacLab/source directory.
    print_info("Installing extensions inside the Isaac Lab repository...")
    python_exe = extract_python_exe()

    # Show which environment is being used.
    if os.environ.get("VIRTUAL_ENV"):
        print_info(f"Using uv/venv environment: {os.environ['VIRTUAL_ENV']}")
    elif os.environ.get("CONDA_PREFIX"):
        print_info(f"Using conda environment: {os.environ['CONDA_PREFIX']}")

    print_info(f"Python executable: {python_exe}")

    # if on ARM arch, temporarily clear LD_PRELOAD
    # LD_PRELOAD is restored below, after installation
    saved_ld_preload = None
    if is_arm() and "LD_PRELOAD" in os.environ:
        print_info("ARM install sandbox: temporarily unsetting LD_PRELOAD for installation.")
        saved_ld_preload = os.environ.pop("LD_PRELOAD")

    try:
        # Upgrade pip first to avoid compatibility issues.
        print_info("Upgrading pip...")
        run_command([python_exe, "-m", "pip", "install", "--upgrade", "pip"])

        # Install pytorch (version based on arch).
        _ensure_cuda_torch()

        # Install the python modules for the extensions in Isaac Lab.
        _install_isaaclab_extensions()

        # Install the python packages for supported reinforcement learning frameworks.
        print_info("Installing extra requirements such as learning frameworks...")
        _install_extra_frameworks(install_type)

        # In some rare cases, torch might not be installed properly by setup.py, add one more check here.
        # Can prevent that from happening.
        _ensure_cuda_torch()

    finally:
        # Restore LD_PRELOAD if we cleared it.
        if saved_ld_preload:
            os.environ["LD_PRELOAD"] = saved_ld_preload

    # Install vscode update unless we're in docker.
    if not (os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")):
        command_vscode_settings()
