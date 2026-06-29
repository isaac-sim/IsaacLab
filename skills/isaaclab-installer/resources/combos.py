"""Canonical single source of truth for Isaac Lab installation combinations.

Every supported (Isaac Lab source x environment manager x Isaac Sim source)
combination is defined here. Scripts in ../scripts/ consume this module
directly so there are zero runtime dependencies on a YAML parser.

Schema for each combo
---------------------
id                  Stable identifier, lowercase, hyphenated.
title               Short human label.
summary             One-line description.
isaaclab_source     "source" | "pip"
env_manager         "uv" | "conda" | "venv" | "none"
isaacsim_source     "pip" | "binary" | "source" | "kitless"
recommended_for     List of use_case ids this combo is appropriate for.
difficulty          "easiest" | "easy" | "advanced"
requires            Dict of hard requirements checked by preflight:
                       min_glibc, min_driver, supported_arches, min_disk_gb,
                       min_ram_gb, python_versions, requires_sudo, network.
steps               Ordered list of step dicts. Each step:
                       id            unique within combo
                       title         human label
                       cmd           shell command (may contain {placeholders})
                       cwd           working directory (literal or {placeholder})
                       requires_sudo bool
                       env           dict of env vars to set for this step
                       needs_auth    optional string id of an auth token
                       skip_if       optional shell expression evaluated by
                                     execute_install.py to decide skip
                       on_failure    "abort" | "warn"
verify              Dict describing post-install smoke test:
                       cmd, cwd, headless_ok (bool)
notes               List of free-form caveats shown to the user before
                    execution.

Placeholders resolved by plan_install.py
----------------------------------------
{ISAACLAB_DIR}      Absolute path to the cloned IsaacLab repo (or pip
                    install location).
{ENV_NAME}          User-chosen environment name (default env_isaaclab).
{ENV_PYTHON}        Path to the python executable inside the new env.
{ISAACSIM_PATH}     Absolute path to the Isaac Sim install root.
{ISAACSIM_VERSION}  Pinned Isaac Sim version (e.g. 6.0.1.0 — the pypi wheel).
{TORCH_INDEX}       PyTorch index URL appropriate for the system arch.
{TORCH_PIN}         "torch==2.10.0 torchvision==0.25.0"
{PIP}               "uv pip" or "pip" depending on env_manager.
{HOME}              User home directory.
{APT_AARCH64_DEPS}  Extra apt packages needed on aarch64 builds.
"""

from __future__ import annotations

# Pinned versions. Update these when IsaacLab bumps support.
# DEFAULT_ISAACSIM_VERSION is the pypi wheel string (e.g. "6.0.1.0").
# DISPLAY_ISAACSIM_VERSION is the human-facing marketing string ("6.0.1").
# Keep both in sync when bumping.
DEFAULT_ISAACSIM_VERSION = "6.0.1.0"
DISPLAY_ISAACSIM_VERSION = "6.0.1"
DEFAULT_PYTHON_VERSION = "3.12"
DEFAULT_TORCH_PIN = "torch==2.10.0 torchvision==0.25.0"
TORCH_INDEX_X86 = "https://download.pytorch.org/whl/cu128"
TORCH_INDEX_AARCH64 = "https://download.pytorch.org/whl/cu130"

# Driver thresholds (per docs/source/setup/installation/index.rst).
MIN_DRIVER_X86 = "580.95.05"
MIN_DRIVER_AARCH64 = "580.142"

# GLIBC threshold for pip-installed Isaac Sim (per pip_installation.rst).
MIN_GLIBC_PIP_ISAACSIM = "2.35"

# Shared apt packages.
COMMON_APT_DEPS = "cmake build-essential"
AARCH64_APT_DEPS = (
    "python3.12-dev libgl1-mesa-dev libx11-dev libxcursor-dev "
    "libxi-dev libxinerama-dev libxrandr-dev"
)

# Use cases the recommender knows about.
USE_CASES = [
    "rl_research",          # RL training, mostly headless
    "manipulation",         # Manipulation/teleop, may need rendering
    "sim2real",             # Sim-to-real transfer
    "contribute_isaaclab",  # Modifying Isaac Lab code itself
    "contribute_isaacsim",  # Modifying Isaac Sim source
    "external_extension",   # Building a downstream package on top of IsaacLab
    "kitless_only",         # Newton physics only, no Isaac Sim features
    "explore",              # Just trying things out
]


def _clone_isaaclab_steps():
    """Steps shared by every source-clone combo."""
    return [
        {
            "id": "clone_isaaclab",
            "title": "Clone Isaac Lab repository",
            "cmd": "git clone https://github.com/isaac-sim/IsaacLab.git {ISAACLAB_DIR}",
            "cwd": "{HOME}",
            "requires_sudo": False,
            "skip_if": "test -d {ISAACLAB_DIR}/.git",
            "on_failure": "abort",
        },
    ]


def _apt_deps_step(include_aarch64=False):
    pkgs = COMMON_APT_DEPS
    if include_aarch64:
        pkgs = pkgs + " " + AARCH64_APT_DEPS
    return {
        "id": "apt_deps",
        "title": "Install system build dependencies (apt)",
        "cmd": f"sudo apt-get update && sudo apt-get install -y {pkgs}",
        "cwd": "{HOME}",
        "requires_sudo": True,
        "on_failure": "warn",
        "notes": "Required for cmake/robomimic builds. On aarch64 these include OpenGL/X11 headers.",
    }


def _create_env_uv():
    return {
        "id": "create_env_uv",
        "title": "Create uv virtual environment (Python 3.12)",
        "cmd": "uv venv --python 3.12 --seed {ENV_NAME}",
        "cwd": "{ISAACLAB_DIR}",
        "requires_sudo": False,
        "skip_if": "test -f {ISAACLAB_DIR}/{ENV_NAME}/bin/python",
        "on_failure": "abort",
    }


def _create_env_conda():
    return {
        "id": "create_env_conda",
        "title": "Create conda environment (Python 3.12)",
        "cmd": "conda create -y -n {ENV_NAME} python=3.12",
        "cwd": "{HOME}",
        "requires_sudo": False,
        "skip_if": "conda env list | grep -E '^{ENV_NAME}( |$)' >/dev/null 2>&1",
        "on_failure": "abort",
    }


def _create_env_venv():
    return {
        "id": "create_env_venv",
        "title": "Create venv virtual environment (Python 3.12)",
        "cmd": "python3.12 -m venv {ISAACLAB_DIR}/{ENV_NAME}",
        "cwd": "{ISAACLAB_DIR}",
        "requires_sudo": False,
        "skip_if": "test -f {ISAACLAB_DIR}/{ENV_NAME}/bin/python",
        "on_failure": "abort",
    }


def _upgrade_pip_step():
    return {
        "id": "upgrade_pip",
        "title": "Upgrade pip inside the environment",
        "cmd": "{ENV_PYTHON} -m pip install --upgrade pip",
        "cwd": "{ISAACLAB_DIR}",
        "requires_sudo": False,
        "on_failure": "warn",
    }


def _install_pip_isaacsim_step(arch_torch_index):
    return {
        "id": "install_isaacsim_pip",
        "title": "Install Isaac Sim via pip",
        "cmd": (
            "{PIP} install \"isaacsim[all,extscache]=={ISAACSIM_VERSION}\" "
            "--extra-index-url https://pypi.nvidia.com "
            "--index-strategy unsafe-best-match --prerelease=allow"
        ),
        "cwd": "{ISAACLAB_DIR}",
        "requires_sudo": False,
        "on_failure": "abort",
        "notes": "Pulls Isaac Sim from pypi.nvidia.com. First run may take several minutes.",
    }


def _install_torch_step():
    return {
        "id": "install_torch",
        "title": "Install CUDA-enabled PyTorch",
        "cmd": "{PIP} install -U {TORCH_PIN} --index-url {TORCH_INDEX}",
        "cwd": "{ISAACLAB_DIR}",
        "requires_sudo": False,
        "on_failure": "abort",
    }


def _isaaclab_install_step(install_arg="all"):
    return {
        "id": "isaaclab_install",
        "title": f"Install Isaac Lab packages (./isaaclab.sh -i {install_arg})",
        "cmd": f"./isaaclab.sh -i {install_arg}",
        "cwd": "{ISAACLAB_DIR}",
        "requires_sudo": False,
        "on_failure": "abort",
    }


def _verify_step(headless=True):
    if headless:
        cmd = (
            "./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py "
            "--headless"
        )
    else:
        cmd = "./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --viz kit"
    return {
        "cmd": cmd,
        "cwd": "{ISAACLAB_DIR}",
        "headless_ok": headless,
    }


# ---------------------------------------------------------------------------
# Combos
# ---------------------------------------------------------------------------

COMBOS = [
    # 1. Recommended path: source IsaacLab + uv env + pip Isaac Sim
    {
        "id": "pip-uv-source",
        "title": "Isaac Sim (pip) + Isaac Lab (source) + uv (Recommended)",
        "summary": (
            "Fastest path to full Isaac Lab with all features. Isaac Sim "
            "installed via pip from pypi.nvidia.com, Isaac Lab cloned from "
            "GitHub, all inside a uv virtual environment."
        ),
        "isaaclab_source": "source",
        "env_manager": "uv",
        "isaacsim_source": "pip",
        "recommended_for": [
            "rl_research", "manipulation", "sim2real", "contribute_isaaclab",
            "explore",
        ],
        "difficulty": "easy",
        "requires": {
            "min_glibc": MIN_GLIBC_PIP_ISAACSIM,
            "min_driver": MIN_DRIVER_X86,
            "supported_arches": ["x86_64", "aarch64"],
            "min_disk_gb": 30,
            "min_ram_gb": 32,
            "python_versions": ["3.12"],
            "requires_sudo": True,
            "network": True,
        },
        "steps": [
            _apt_deps_step(),
            *_clone_isaaclab_steps(),
            _create_env_uv(),
            _upgrade_pip_step(),
            _install_pip_isaacsim_step(TORCH_INDEX_X86),
            _install_torch_step(),
            _isaaclab_install_step("all"),
        ],
        "verify": _verify_step(headless=True),
        "notes": [
            "Requires GLIBC 2.35+. Ubuntu 20.04 users should use the binary combo instead.",
            "On aarch64 (DGX Spark) extra apt packages are added automatically.",
            "First Isaac Sim launch will prompt for EULA acceptance.",
        ],
    },
    # 2. Source IsaacLab + conda + pip Isaac Sim
    {
        "id": "pip-conda-source",
        "title": "Isaac Sim (pip) + Isaac Lab (source) + conda",
        "summary": (
            "Same as the recommended path but using conda for environment "
            "management. Useful if you already standardize on conda."
        ),
        "isaaclab_source": "source",
        "env_manager": "conda",
        "isaacsim_source": "pip",
        "recommended_for": [
            "rl_research", "manipulation", "sim2real", "contribute_isaaclab",
        ],
        "difficulty": "easy",
        "requires": {
            "min_glibc": MIN_GLIBC_PIP_ISAACSIM,
            "min_driver": MIN_DRIVER_X86,
            "supported_arches": ["x86_64", "aarch64"],
            "min_disk_gb": 30,
            "min_ram_gb": 32,
            "python_versions": ["3.12"],
            "requires_sudo": True,
            "network": True,
            "requires_tools": ["conda"],
        },
        "steps": [
            _apt_deps_step(),
            *_clone_isaaclab_steps(),
            _create_env_conda(),
            _upgrade_pip_step(),
            _install_pip_isaacsim_step(TORCH_INDEX_X86),
            _install_torch_step(),
            _isaaclab_install_step("all"),
        ],
        "verify": _verify_step(headless=True),
        "notes": [
            "Requires GLIBC 2.35+. Use the binary combo on older distros.",
            "Activate the env in new shells with `conda activate {ENV_NAME}`.",
        ],
    },
    # 3. Source IsaacLab + venv + pip Isaac Sim
    {
        "id": "pip-venv-source",
        "title": "Isaac Sim (pip) + Isaac Lab (source) + venv",
        "summary": (
            "Stock-Python venv variant. No third-party env tool needed beyond "
            "system Python 3.12."
        ),
        "isaaclab_source": "source",
        "env_manager": "venv",
        "isaacsim_source": "pip",
        "recommended_for": ["rl_research", "explore"],
        "difficulty": "easy",
        "requires": {
            "min_glibc": MIN_GLIBC_PIP_ISAACSIM,
            "min_driver": MIN_DRIVER_X86,
            "supported_arches": ["x86_64", "aarch64"],
            "min_disk_gb": 30,
            "min_ram_gb": 32,
            "python_versions": ["3.12"],
            "requires_sudo": True,
            "network": True,
            "requires_tools": ["python3.12"],
        },
        "steps": [
            _apt_deps_step(),
            *_clone_isaaclab_steps(),
            _create_env_venv(),
            _upgrade_pip_step(),
            _install_pip_isaacsim_step(TORCH_INDEX_X86),
            _install_torch_step(),
            _isaaclab_install_step("all"),
        ],
        "verify": _verify_step(headless=True),
        "notes": [
            "Requires Python 3.12 installed on the system (the venv wraps it).",
            "Requires GLIBC 2.35+.",
        ],
    },
    # 4. Source IsaacLab + uv + Isaac Sim binary
    {
        "id": "binary-uv-source",
        "title": "Isaac Sim (binary download) + Isaac Lab (source) + uv",
        "summary": (
            "Isaac Sim from the official pre-built binary zip, Isaac Lab from "
            "source. Works on older Linux distros that lack GLIBC 2.35."
        ),
        "isaaclab_source": "source",
        "env_manager": "uv",
        "isaacsim_source": "binary",
        "recommended_for": ["rl_research", "manipulation", "sim2real"],
        "difficulty": "easy",
        "requires": {
            "min_driver": MIN_DRIVER_X86,
            "supported_arches": ["x86_64"],
            "min_disk_gb": 35,
            "min_ram_gb": 32,
            "python_versions": ["3.12"],
            "requires_sudo": True,
            "network": True,
            "manual_download": True,
        },
        "steps": [
            _apt_deps_step(),
            *_clone_isaaclab_steps(),
            {
                "id": "isaacsim_binary_check",
                "title": "Verify Isaac Sim binary is extracted",
                "cmd": "test -d {ISAACSIM_PATH} && test -f {ISAACSIM_PATH}/isaac-sim.sh",
                "cwd": "{HOME}",
                "requires_sudo": False,
                "on_failure": "abort",
                "notes": (
                    "The Isaac Sim binary must be downloaded manually from "
                    "https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html "
                    "and extracted to {ISAACSIM_PATH}. The skill will pause "
                    "before this step so you can complete the download."
                ),
                "manual_step": True,
            },
            {
                "id": "symlink_isaacsim",
                "title": "Create _isaac_sim symbolic link",
                "cmd": "ln -sf {ISAACSIM_PATH} _isaac_sim",
                "cwd": "{ISAACLAB_DIR}",
                "requires_sudo": False,
                "on_failure": "abort",
            },
            _create_env_uv(),
            _upgrade_pip_step(),
            _install_torch_step(),
            _isaaclab_install_step("all"),
        ],
        "verify": _verify_step(headless=True),
        "notes": [
            "Requires manual download of the Isaac Sim binary zip.",
            "The skill will pause and prompt you for the extracted path.",
        ],
    },
    # 5. Source IsaacLab + conda + Isaac Sim binary
    {
        "id": "binary-conda-source",
        "title": "Isaac Sim (binary download) + Isaac Lab (source) + conda",
        "summary": "Binary Isaac Sim with conda env management.",
        "isaaclab_source": "source",
        "env_manager": "conda",
        "isaacsim_source": "binary",
        "recommended_for": ["rl_research", "manipulation"],
        "difficulty": "easy",
        "requires": {
            "min_driver": MIN_DRIVER_X86,
            "supported_arches": ["x86_64"],
            "min_disk_gb": 35,
            "min_ram_gb": 32,
            "python_versions": ["3.12"],
            "requires_sudo": True,
            "network": True,
            "requires_tools": ["conda"],
            "manual_download": True,
        },
        "steps": [
            _apt_deps_step(),
            *_clone_isaaclab_steps(),
            {
                "id": "isaacsim_binary_check",
                "title": "Verify Isaac Sim binary is extracted",
                "cmd": "test -d {ISAACSIM_PATH} && test -f {ISAACSIM_PATH}/isaac-sim.sh",
                "cwd": "{HOME}",
                "requires_sudo": False,
                "on_failure": "abort",
                "manual_step": True,
            },
            {
                "id": "symlink_isaacsim",
                "title": "Create _isaac_sim symbolic link",
                "cmd": "ln -sf {ISAACSIM_PATH} _isaac_sim",
                "cwd": "{ISAACLAB_DIR}",
                "requires_sudo": False,
                "on_failure": "abort",
            },
            _create_env_conda(),
            _upgrade_pip_step(),
            _install_torch_step(),
            _isaaclab_install_step("all"),
        ],
        "verify": _verify_step(headless=True),
        "notes": [
            "Requires manual download of the Isaac Sim binary zip.",
        ],
    },
    # 6. Source IsaacLab + venv + Isaac Sim binary
    {
        "id": "binary-venv-source",
        "title": "Isaac Sim (binary download) + Isaac Lab (source) + venv",
        "summary": "Binary Isaac Sim with stock-Python venv.",
        "isaaclab_source": "source",
        "env_manager": "venv",
        "isaacsim_source": "binary",
        "recommended_for": ["rl_research"],
        "difficulty": "easy",
        "requires": {
            "min_driver": MIN_DRIVER_X86,
            "supported_arches": ["x86_64"],
            "min_disk_gb": 35,
            "min_ram_gb": 32,
            "python_versions": ["3.12"],
            "requires_sudo": True,
            "network": True,
            "requires_tools": ["python3.12"],
            "manual_download": True,
        },
        "steps": [
            _apt_deps_step(),
            *_clone_isaaclab_steps(),
            {
                "id": "isaacsim_binary_check",
                "title": "Verify Isaac Sim binary is extracted",
                "cmd": "test -d {ISAACSIM_PATH} && test -f {ISAACSIM_PATH}/isaac-sim.sh",
                "cwd": "{HOME}",
                "requires_sudo": False,
                "on_failure": "abort",
                "manual_step": True,
            },
            {
                "id": "symlink_isaacsim",
                "title": "Create _isaac_sim symbolic link",
                "cmd": "ln -sf {ISAACSIM_PATH} _isaac_sim",
                "cwd": "{ISAACLAB_DIR}",
                "requires_sudo": False,
                "on_failure": "abort",
            },
            _create_env_venv(),
            _upgrade_pip_step(),
            _install_torch_step(),
            _isaaclab_install_step("all"),
        ],
        "verify": _verify_step(headless=True),
        "notes": [
            "Requires Python 3.12 installed system-wide.",
            "Requires manual download of the Isaac Sim binary zip.",
        ],
    },
    # 7. Source Isaac Sim build + uv + source IsaacLab
    {
        "id": "source-uv-source",
        "title": "Isaac Sim (source build) + Isaac Lab (source) + uv",
        "summary": (
            "Build Isaac Sim from its GitHub repository, then install Isaac "
            "Lab from source. Only for users who actively modify Isaac Sim."
        ),
        "isaaclab_source": "source",
        "env_manager": "uv",
        "isaacsim_source": "source",
        "recommended_for": ["contribute_isaacsim"],
        "difficulty": "advanced",
        "requires": {
            "min_driver": MIN_DRIVER_X86,
            "supported_arches": ["x86_64"],
            "min_disk_gb": 80,
            "min_ram_gb": 32,
            "python_versions": ["3.12"],
            "requires_sudo": True,
            "network": True,
            "ubuntu_min": "22.04",
        },
        "steps": [
            _apt_deps_step(),
            {
                "id": "clone_isaacsim",
                "title": "Clone Isaac Sim repository",
                "cmd": "git clone https://github.com/isaac-sim/IsaacSim.git",
                "cwd": "{HOME}",
                "requires_sudo": False,
                "skip_if": "test -d {HOME}/IsaacSim/.git",
                "on_failure": "abort",
            },
            {
                "id": "build_isaacsim",
                "title": "Build Isaac Sim from source (~30-60 min)",
                "cmd": "./build.sh",
                "cwd": "{HOME}/IsaacSim",
                "requires_sudo": False,
                "on_failure": "abort",
                "notes": "Long-running build. CPU heavy.",
            },
            *_clone_isaaclab_steps(),
            {
                "id": "symlink_isaacsim_source",
                "title": "Create _isaac_sim symbolic link to built release",
                "cmd": "ln -sf {HOME}/IsaacSim/_build/linux-x86_64/release _isaac_sim",
                "cwd": "{ISAACLAB_DIR}",
                "requires_sudo": False,
                "on_failure": "abort",
            },
            _create_env_uv(),
            _upgrade_pip_step(),
            _install_torch_step(),
            _isaaclab_install_step("all"),
        ],
        "verify": _verify_step(headless=True),
        "notes": [
            "Requires Ubuntu 22.04 LTS or newer.",
            "Compatibility warning: Isaac Lab develop branch may not match Isaac Sim develop branch. See docs/source/setup/installation/source_installation.rst.",
        ],
    },
    # 8. Source Isaac Sim build + conda + source IsaacLab
    {
        "id": "source-conda-source",
        "title": "Isaac Sim (source build) + Isaac Lab (source) + conda",
        "summary": "Source Isaac Sim build with conda env. Advanced.",
        "isaaclab_source": "source",
        "env_manager": "conda",
        "isaacsim_source": "source",
        "recommended_for": ["contribute_isaacsim"],
        "difficulty": "advanced",
        "requires": {
            "min_driver": MIN_DRIVER_X86,
            "supported_arches": ["x86_64"],
            "min_disk_gb": 80,
            "min_ram_gb": 32,
            "python_versions": ["3.12"],
            "requires_sudo": True,
            "network": True,
            "ubuntu_min": "22.04",
            "requires_tools": ["conda"],
        },
        "steps": [
            _apt_deps_step(),
            {
                "id": "clone_isaacsim",
                "title": "Clone Isaac Sim repository",
                "cmd": "git clone https://github.com/isaac-sim/IsaacSim.git",
                "cwd": "{HOME}",
                "requires_sudo": False,
                "skip_if": "test -d {HOME}/IsaacSim/.git",
                "on_failure": "abort",
            },
            {
                "id": "build_isaacsim",
                "title": "Build Isaac Sim from source (~30-60 min)",
                "cmd": "./build.sh",
                "cwd": "{HOME}/IsaacSim",
                "requires_sudo": False,
                "on_failure": "abort",
            },
            *_clone_isaaclab_steps(),
            {
                "id": "symlink_isaacsim_source",
                "title": "Create _isaac_sim symbolic link to built release",
                "cmd": "ln -sf {HOME}/IsaacSim/_build/linux-x86_64/release _isaac_sim",
                "cwd": "{ISAACLAB_DIR}",
                "requires_sudo": False,
                "on_failure": "abort",
            },
            _create_env_conda(),
            _upgrade_pip_step(),
            _install_torch_step(),
            _isaaclab_install_step("all"),
        ],
        "verify": _verify_step(headless=True),
        "notes": [
            "Requires Ubuntu 22.04 LTS or newer.",
        ],
    },
    # 9. Pip-only: Isaac Lab pip + Isaac Sim pip + uv
    {
        "id": "pip-only-uv",
        "title": "Isaac Lab (pip) + Isaac Sim (pip) + uv (external extensions)",
        "summary": (
            "Both Isaac Lab and Isaac Sim installed as pip packages. No git "
            "clone. Intended for building external extensions on top of "
            "Isaac Lab. Does NOT include training scripts or examples."
        ),
        "isaaclab_source": "pip",
        "env_manager": "uv",
        "isaacsim_source": "pip",
        "recommended_for": ["external_extension"],
        "difficulty": "easy",
        "requires": {
            "min_glibc": MIN_GLIBC_PIP_ISAACSIM,
            "min_driver": MIN_DRIVER_X86,
            "supported_arches": ["x86_64", "aarch64"],
            "min_disk_gb": 20,
            "min_ram_gb": 32,
            "python_versions": ["3.12"],
            "requires_sudo": False,
            "network": True,
        },
        "steps": [
            {
                "id": "create_env_uv",
                "title": "Create uv virtual environment (Python 3.12)",
                "cmd": "uv venv --python 3.12 --seed {ENV_NAME}",
                "cwd": "{HOME}",
                "requires_sudo": False,
                "skip_if": "test -f {HOME}/{ENV_NAME}/bin/python",
                "on_failure": "abort",
            },
            _upgrade_pip_step(),
            {
                "id": "install_isaaclab_pip",
                "title": "Install Isaac Lab + Isaac Sim from pip",
                "cmd": (
                    "{PIP} install \"isaaclab[isaacsim,all]\" "
                    "--extra-index-url https://pypi.nvidia.com "
                    "--index-strategy unsafe-best-match --prerelease=allow"
                ),
                "cwd": "{HOME}",
                "requires_sudo": False,
                "on_failure": "abort",
            },
            _install_torch_step(),
        ],
        "verify": {
            "cmd": "{ENV_PYTHON} -c \"import isaaclab; import isaacsim; print('OK', isaaclab.__version__)\"",
            "cwd": "{HOME}",
            "headless_ok": True,
        },
        "notes": [
            "Pip-installed Isaac Lab does NOT include scripts/. You'll need to write your own runner scripts.",
            "See docs: docs/source/setup/installation/isaaclab_pip_installation.rst.",
        ],
    },
    # 10. Pip-only: Isaac Lab pip + Isaac Sim pip + conda
    {
        "id": "pip-only-conda",
        "title": "Isaac Lab (pip) + Isaac Sim (pip) + conda",
        "summary": "Pip-only path with conda env management.",
        "isaaclab_source": "pip",
        "env_manager": "conda",
        "isaacsim_source": "pip",
        "recommended_for": ["external_extension"],
        "difficulty": "easy",
        "requires": {
            "min_glibc": MIN_GLIBC_PIP_ISAACSIM,
            "min_driver": MIN_DRIVER_X86,
            "supported_arches": ["x86_64", "aarch64"],
            "min_disk_gb": 20,
            "min_ram_gb": 32,
            "python_versions": ["3.12"],
            "requires_sudo": False,
            "network": True,
            "requires_tools": ["conda"],
        },
        "steps": [
            _create_env_conda(),
            _upgrade_pip_step(),
            {
                "id": "install_isaaclab_pip",
                "title": "Install Isaac Lab + Isaac Sim from pip",
                "cmd": (
                    "{PIP} install \"isaaclab[isaacsim,all]\" "
                    "--extra-index-url https://pypi.nvidia.com --pre"
                ),
                "cwd": "{HOME}",
                "requires_sudo": False,
                "on_failure": "abort",
            },
            _install_torch_step(),
        ],
        "verify": {
            "cmd": "{ENV_PYTHON} -c \"import isaaclab; import isaacsim; print('OK', isaaclab.__version__)\"",
            "cwd": "{HOME}",
            "headless_ok": True,
        },
        "notes": [
            "Pip-installed Isaac Lab does NOT include training scripts.",
        ],
    },
    # 11. Pip-only: Isaac Lab pip + Isaac Sim pip + venv
    {
        "id": "pip-only-venv",
        "title": "Isaac Lab (pip) + Isaac Sim (pip) + venv",
        "summary": "Pip-only path with stock venv.",
        "isaaclab_source": "pip",
        "env_manager": "venv",
        "isaacsim_source": "pip",
        "recommended_for": ["external_extension"],
        "difficulty": "easy",
        "requires": {
            "min_glibc": MIN_GLIBC_PIP_ISAACSIM,
            "min_driver": MIN_DRIVER_X86,
            "supported_arches": ["x86_64", "aarch64"],
            "min_disk_gb": 20,
            "min_ram_gb": 32,
            "python_versions": ["3.12"],
            "requires_sudo": False,
            "network": True,
            "requires_tools": ["python3.12"],
        },
        "steps": [
            {
                "id": "create_env_venv",
                "title": "Create venv virtual environment (Python 3.12)",
                "cmd": "python3.12 -m venv {HOME}/{ENV_NAME}",
                "cwd": "{HOME}",
                "requires_sudo": False,
                "skip_if": "test -f {HOME}/{ENV_NAME}/bin/python",
                "on_failure": "abort",
            },
            _upgrade_pip_step(),
            {
                "id": "install_isaaclab_pip",
                "title": "Install Isaac Lab + Isaac Sim from pip",
                "cmd": (
                    "{PIP} install \"isaaclab[isaacsim,all]\" "
                    "--extra-index-url https://pypi.nvidia.com --pre"
                ),
                "cwd": "{HOME}",
                "requires_sudo": False,
                "on_failure": "abort",
            },
            _install_torch_step(),
        ],
        "verify": {
            "cmd": "{ENV_PYTHON} -c \"import isaaclab; import isaacsim; print('OK', isaaclab.__version__)\"",
            "cwd": "{HOME}",
            "headless_ok": True,
        },
        "notes": [
            "Requires Python 3.12 installed on the system.",
        ],
    },
    # 12. Kit-less: source IsaacLab, no Isaac Sim, uv
    {
        "id": "kitless-uv",
        "title": "Kit-less: Isaac Lab (source) + Newton physics + uv (no Isaac Sim)",
        "summary": (
            "Newton physics only, no Isaac Sim. The fastest possible path for "
            "RL training that doesn't need PhysX, RTX rendering, or ROS."
        ),
        "isaaclab_source": "source",
        "env_manager": "uv",
        "isaacsim_source": "kitless",
        "recommended_for": ["kitless_only", "rl_research", "explore"],
        "difficulty": "easiest",
        "requires": {
            "min_driver": MIN_DRIVER_X86,
            "supported_arches": ["x86_64", "aarch64"],
            "min_disk_gb": 15,
            "min_ram_gb": 16,
            "python_versions": ["3.12"],
            "requires_sudo": True,
            "network": True,
        },
        "steps": [
            _apt_deps_step(),
            *_clone_isaaclab_steps(),
            _create_env_uv(),
            _upgrade_pip_step(),
            _install_torch_step(),
            _isaaclab_install_step("newton,rl[rsl-rl]"),
        ],
        "verify": {
            "cmd": (
                "./isaaclab.sh train --rl_library rsl_rl "
                "--task=Isaac-Cartpole-Direct-v0 --num_envs=16 --max_iterations=2 "
                "--headless physics=newton_mjwarp --visualizer newton"
            ),
            "cwd": "{ISAACLAB_DIR}",
            "headless_ok": True,
        },
        "notes": [
            "No Isaac Sim is installed. PhysX, RTX rendering, ROS, URDF/MJCF importers will be unavailable.",
            "ovphysx and ovrtx renderers ARE available in kit-less mode (Isaac Lab 3.0+).",
            "See docs/source/setup/installation/kitless_installation.rst for the full feature list.",
        ],
    },
    # 13. Kit-less: source IsaacLab, no Isaac Sim, conda
    {
        "id": "kitless-conda",
        "title": "Kit-less: Isaac Lab (source) + Newton physics + conda",
        "summary": "Same as kitless-uv but with conda.",
        "isaaclab_source": "source",
        "env_manager": "conda",
        "isaacsim_source": "kitless",
        "recommended_for": ["kitless_only", "rl_research"],
        "difficulty": "easiest",
        "requires": {
            "min_driver": MIN_DRIVER_X86,
            "supported_arches": ["x86_64", "aarch64"],
            "min_disk_gb": 15,
            "min_ram_gb": 16,
            "python_versions": ["3.12"],
            "requires_sudo": True,
            "network": True,
            "requires_tools": ["conda"],
        },
        "steps": [
            _apt_deps_step(),
            *_clone_isaaclab_steps(),
            _create_env_conda(),
            _upgrade_pip_step(),
            _install_torch_step(),
            _isaaclab_install_step("newton,rl[rsl-rl]"),
        ],
        "verify": {
            "cmd": (
                "./isaaclab.sh train --rl_library rsl_rl "
                "--task=Isaac-Cartpole-Direct-v0 --num_envs=16 --max_iterations=2 "
                "--headless physics=newton_mjwarp --visualizer newton"
            ),
            "cwd": "{ISAACLAB_DIR}",
            "headless_ok": True,
        },
        "notes": [
            "No Isaac Sim is installed.",
            "ovphysx and ovrtx renderers ARE available in kit-less mode (Isaac Lab 3.0+).",
        ],
    },
]


def get_combo(combo_id):
    """Return the combo dict matching combo_id, or None."""
    for c in COMBOS:
        if c["id"] == combo_id:
            return c
    return None


def list_combo_ids():
    """Return all combo ids in declaration order."""
    return [c["id"] for c in COMBOS]


def torch_index_for_arch(arch):
    """Return the appropriate PyTorch wheel index URL for a CPU arch."""
    if arch == "aarch64":
        return TORCH_INDEX_AARCH64
    return TORCH_INDEX_X86


def is_arch_supported(combo, arch):
    return arch in combo["requires"].get("supported_arches", [])
