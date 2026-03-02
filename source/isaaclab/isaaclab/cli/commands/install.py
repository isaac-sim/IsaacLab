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
    torch_ver = "2.10.0"
    tv_ver = "0.25.0"

    if is_arm():
        cuda_ver = "130"
    else:
        cuda_ver = "128"

    cuda_tag = f"cu{cuda_ver}"
    index_url = f"{base_index}/{cuda_tag}"

    want_torch = f"{torch_ver}+{cuda_tag}"

    # Check current torch version using pip show (includes build tags).
    current_ver = ""
    try:
        result = run_command(
            [
                python_exe,
                "-m",
                "pip",
                "show",
                "torch",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if line.startswith("Version: "):
                    current_ver = line.split("Version: ", 1)[1].strip()
                    break
    except Exception:
        pass

    # Skip install if version already matches (including CUDA build tag).
    if current_ver == want_torch:
        print_info(f"PyTorch {want_torch} already installed.")
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


# Valid sub-package names that can be passed to --install.
# Each sub-package maps to a source directory named "isaaclab_<name>" under source/.
VALID_ISAACLAB_SUBPACKAGES: set[str] = {"assets", "physx", "contrib", "mimic", "newton", "rl", "tasks", "teleop"}

# RL framework names accepted.
# Passing one of these installs all extensions + that framework.
VALID_RL_FRAMEWORKS: set[str] = {"rl_games", "rsl_rl", "sb3", "skrl", "robomimic"}


def _install_isaaclab_extensions(extensions: list[str] | None = None) -> None:
    """Install Isaac Lab extensions from the source directory.

    Scans ``source/`` for sub-directories that contain a ``setup.py`` and
    installs each one as an editable pip package.

    Args:
        extensions: Optional, list of source directory names to install
        If ``None`` is provided, every extension found under ``source/`` is installed.
    """
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
            # Skip extensions not in the requested list.
            if extensions is not None and item.name not in extensions:
                continue
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
    """Install Isaac Lab extensions and optional sub-packages.

    Args:
        install_type: Comma-separated list of extras to install, or one of the
            special values ``"all"`` / ``"none"``. Extra names match the keys
            in ``source/isaaclab/setup.py``'s ``extras_require``:
            * ``"all"`` (default) — install every extension found under
              ``source/``, plus all RL frameworks.
            * ``"none"`` — install only the "core" ``isaaclab`` package and skip
              RL frameworks.
            * Comma-separated extras, e.g. ``"mimic,assets"`` — install
              only the "core" ``isaaclab`` package plus the listed sub-packages.
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

    # Decide which source directories (source/isaaclab/*) to install.
    # "all"        : install everything + all RL frameworks
    # "none"       : core isaaclab only, no RL frameworks
    # RL framework : install everything + only that RL framework (e.g. "skrl")
    # "a,b"        : core + selected sub-package directories, no RL frameworks
    if install_type == "all":
        extensions = None
        framework_type = "all"
    elif install_type == "none":
        extensions = ["isaaclab"]
        framework_type = "none"
    elif install_type in VALID_RL_FRAMEWORKS:
        # Single RL framework name: install all extensions + only that framework.
        extensions = None
        framework_type = install_type
    else:
        # Parse comma-separated sub-package names into source directory names.
        extensions = ["isaaclab"]  # core is always required
        for name in (s.strip() for s in install_type.split(",") if s.strip()):
            if name in VALID_ISAACLAB_SUBPACKAGES:
                extensions.append(f"isaaclab_{name}")
            else:
                valid = sorted(VALID_ISAACLAB_SUBPACKAGES) + sorted(VALID_RL_FRAMEWORKS)
                print_warning(f"Unknown sub-package '{name}'. Valid values: {', '.join(valid)}. Skipping.")
        framework_type = "none"  # RL frameworks not applied in selective mode

    # Configure extra package indexes for NVIDIA and MuJoCo wheels.
    os.environ.setdefault("UV_INDEX", "https://pypi.nvidia.com")
    os.environ.setdefault("PIP_EXTRA_INDEX_URL", "https://pypi.nvidia.com")
    os.environ.setdefault("PIP_FIND_LINKS", "https://py.mujoco.org/")

    # if on ARM arch, temporarily clear LD_PRELOAD
    # LD_PRELOAD is restored below, after installation
    saved_ld_preload = None
    if is_arm() and "LD_PRELOAD" in os.environ:
        print_info("ARM install sandbox: temporarily unsetting LD_PRELOAD for installation.")
        saved_ld_preload = os.environ.pop("LD_PRELOAD")

    # Temporarily filter Isaac Sim pre-bundled package paths from PYTHONPATH during all pip operations.
    # This prevents pip from scanning and managing packages in Isaac Sim's pip_prebundle directories,
    # which can cause those packages to be deleted or modified. This is especially important
    # in conda environments where Isaac Sim setup scripts add these paths to PYTHONPATH.
    saved_pythonpath = None
    filtered_pythonpath = None
    if "PYTHONPATH" in os.environ:
        saved_pythonpath = os.environ["PYTHONPATH"]
        # Filter out any paths containing pip_prebundle (pre-bundled packages that pip shouldn't manage)
        paths = saved_pythonpath.split(os.pathsep)
        filtered_paths = [p for p in paths if p and "pip_prebundle" not in p]

        if len(filtered_paths) != len(paths):
            filtered_pythonpath = os.pathsep.join(filtered_paths)
            os.environ["PYTHONPATH"] = filtered_pythonpath
            filtered_count = len(paths) - len(filtered_paths)
            print_info(
                f"Temporarily filtering {filtered_count} Isaac Sim pre-bundled package path(s) from PYTHONPATH "
                "during pip operations to prevent interference with pre-bundled packages."
            )

    try:
        # Upgrade pip first to avoid compatibility issues.
        print_info("Upgrading pip...")
        run_command([python_exe, "-m", "pip", "install", "--upgrade", "pip"])

        # Pin setuptools to avoid issues with pkg_resources removal in 82.0.0.
        run_command([python_exe, "-m", "pip", "install", "setuptools<82.0.0"])

        # Install pytorch (version based on arch).
        _ensure_cuda_torch()

        # Install the python modules for the extensions in Isaac Lab.
        _install_isaaclab_extensions(extensions)

        # Install the python packages for supported reinforcement learning frameworks.
        print_info("Installing extra requirements such as learning frameworks...")
        _install_extra_frameworks(framework_type)

        # In some rare cases, torch might not be installed properly by setup.py, add one more check here.
        # Can prevent that from happening.
        _ensure_cuda_torch()

    finally:
        # Restore LD_PRELOAD if we cleared it.
        if saved_ld_preload:
            os.environ["LD_PRELOAD"] = saved_ld_preload
        # Restore PYTHONPATH if we filtered it.
        if saved_pythonpath is not None:
            os.environ["PYTHONPATH"] = saved_pythonpath

    # Install vscode update unless we're in docker.
    if not (os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")):
        command_vscode_settings()
