# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import shutil
import subprocess

from .utils import ISAACLAB_ROOT, extract_python_exe, is_arm, is_windows, run_command


def install_system_deps():
    """install system dependencies"""
    if is_windows():
        return

    # Check if cmake is already installed.
    if shutil.which("cmake"):
        print("[INFO] cmake is already installed.")
    else:
        # Check if running as root.
        if os.geteuid() != 0:
            print("[INFO] Installing system dependencies...")
        cmd = ["apt-get", "update"]
        run_command(["sudo"] + cmd if os.geteuid() != 0 else cmd)

        cmd = [
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            "cmake",
            "build-essential",
        ]
        run_command(["sudo"] + cmd if os.geteuid() != 0 else cmd)


def ensure_cuda_torch():
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
        result = subprocess.run(
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
    print(f"[INFO] Installing torch=={torch_ver} and torchvision=={tv_ver} ({cuda_tag}) from {index_url}...")

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


def install_isaaclab_extensions():
    """check if input directory is a python extension and install the module"""
    python_exe = extract_python_exe()
    source_dir = ISAACLAB_ROOT / "source"

    if not source_dir.exists():
        print(f"[WARNING] Source directory not found: {source_dir}")
        return

    # recursively look into directories and install them
    # this does not check dependencies between extensions
    # source directory
    for item in source_dir.iterdir():
        if item.is_dir() and (item / "setup.py").exists():
            print(f"[INFO] Installing extension: {item.name}")
            # If the directory contains setup.py then install the python module.
            run_command(
                [
                    python_exe,
                    "-m",
                    "pip",
                    "install",
                    "--prefer-binary",
                    "--editable",
                    str(item),
                ]
            )


def install_extra_frameworks(framework_name="all"):
    """install the python packages for supported reinforcement learning frameworks"""
    python_exe = extract_python_exe()

    extras = ""
    if framework_name != "none":
        extras = f"[{framework_name}]"

    # Check if specified which rl-framework to install.
    if framework_name == "none":
        print("[INFO] No rl-framework will be installed.")
        return

    print(f"[INFO] Installing rl-framework: {framework_name}")

    # Install the learning frameworks specified.
    # Using --prefer-binary as per script.
    run_command(
        [
            python_exe,
            "-m",
            "pip",
            "install",
            "--prefer-binary",
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
            "--prefer-binary",
            "-e",
            f"{ISAACLAB_ROOT}/source/isaaclab_mimic{extras}",
        ]
    )


def install(install_type="all"):
    """
    Install stuff

    Args:
        install_type (str): The RL framework to install ('all', 'none', or specific name).
    """
    # Install system dependencies first.
    install_system_deps()

    # Install the python packages in IsaacLab/source directory.
    print("[INFO] Installing extensions inside the Isaac Lab repository...")
    python_exe = extract_python_exe()

    # Show which environment is being used.
    if os.environ.get("VIRTUAL_ENV"):
        print(f"[INFO] Using uv/venv environment: {os.environ['VIRTUAL_ENV']}")
    elif os.environ.get("CONDA_PREFIX"):
        print(f"[INFO] Using conda environment: {os.environ['CONDA_PREFIX']}")
    else:
        print("[INFO] Using Isaac Sim Python or system Python")

    print(f"[INFO] Python executable: {python_exe}")

    # if on ARM arch, temporarily clear LD_PRELOAD
    # LD_PRELOAD is restored below, after installation
    saved_ld_preload = None
    if is_arm() and "LD_PRELOAD" in os.environ:
        print("[INFO] ARM install sandbox: temporarily unsetting LD_PRELOAD for installation.")
        saved_ld_preload = os.environ.pop("LD_PRELOAD")

    try:
        # Upgrade pip first to avoid compatibility issues.
        print("[INFO] Upgrading pip...")
        run_command([python_exe, "-m", "pip", "install", "--upgrade", "pip"])

        # Install pytorch (version based on arch).
        ensure_cuda_torch()

        install_isaaclab_extensions()

        # Install the python packages for supported reinforcement learning frameworks.
        print("[INFO] Installing extra requirements such as learning frameworks...")
        install_extra_frameworks(install_type)

        # In some rare cases, torch might not be installed properly by setup.py, add one more check here.
        # Can prevent that from happening.
        ensure_cuda_torch()

    finally:
        # Restore LD_PRELOAD if we cleared it.
        if saved_ld_preload:
            os.environ["LD_PRELOAD"] = saved_ld_preload
