# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab' python package."""

import os

import toml
from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "numpy>=2",
    "torch>=2.10",
    "onnx>=1.18.0",  # 1.16.2 throws access violation on Windows
    "prettytable==3.3.0",
    "toml",
    # devices
    "hidapi==0.14.0.post2",
    # reinforcement learning
    "gymnasium==1.2.1",
    # procedural-generation
    "trimesh",
    "pyglet>=2.1.6",
    "mujoco>=3.5",
    "mujoco-warp>=3.5",
    # image processing
    "transformers==4.57.6",
    "einops",  # needed for transformers, doesn't always auto-install
    "warp-lang==1.12.0",
    "matplotlib>=3.10.3",  # minimum version for Python 3.12 support
    # make sure this is consistent with isaac sim version
    "pillow==12.1.1",
    # required by omni.replicator.core S3 backend
    "botocore",
    # livestream
    "starlette==0.49.1",
    "omniverseclient",
    # testing
    "pytest",
    "pytest-mock",
    "junitparser",
    "coverage==7.6.1",
    "debugpy>=1.8.20",
    "flatdict==4.0.0",
    "flaky",
    "packaging",
    "psutil",
    # Required by pydantic-core/imgui_bundle on Python 3.12 (Sentinel symbol).
    "typing_extensions>=4.14.0",
    "lazy_loader>=0.4",
]

# Append Linux x86_64 and ARM64 deps via PEP 508 markers
SUPPORTED_ARCHS_ARM = "platform_machine in 'x86_64,AMD64,aarch64,arm64'"
SUPPORTED_ARCHS = "platform_machine in 'x86_64,AMD64'"
INSTALL_REQUIRES += [
    # required by isaaclab.isaaclab.controllers.pink_ik
    f"pin-pink==3.1.0 ; platform_system == 'Linux' and ({SUPPORTED_ARCHS_ARM})",
    f"daqp==0.7.2 ; platform_system == 'Linux' and ({SUPPORTED_ARCHS_ARM})",
    # required by isaaclab.devices.openxr.retargeters.humanoid.fourier.gr1_t2_dex_retargeting_utils
    f"dex-retargeting==0.5.0 ; platform_system == 'Linux' and ({SUPPORTED_ARCHS})",
]
# Adds OpenUSD dependencies based on architecture for Kit less mode.
INSTALL_REQUIRES += [
    f"usd-core==25.8.0 ; ({SUPPORTED_ARCHS})",
    f"usd-exchange>=2.2 ; ({SUPPORTED_ARCHS_ARM})",
]

# Pin hf-xet to avoid broken tarball (hf_xet-1.1.8.dev2) cached on NVIDIA Artifactory.
# (https://urm.nvidia.com/artifactory/api/pypi/ct-omniverse-pypi) that gets installed wth --pre 
# and --extra-index-url flags. The broken hf-xet-1.1.8.dev2 package is present as of Mar 12 2026.
# TODO: Can be removed once the broken hf-xet-1.1.8.dev2 package is removed from NVIDIA Artifactory.
# Issue: https://nvbugs/5974917 includes verification steps.
INSTALL_REQUIRES += [
    # 1.4.1 is latest as of Mar 12 2026
    f"hf-xet>=1.4.1,<2.0.0 ; ({SUPPORTED_ARCHS_ARM})",
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu128"]

# Isaac Lab subpackages + Isaac Sim
EXTRAS_REQUIRE = {
    "isaacsim": ["isaacsim[all,extscache]==5.1.0"],
    # Individual Isaac Lab sub-packages
    "assets": ["isaaclab_assets"],
    "physx": ["isaaclab_physx"],
    "contrib": ["isaaclab_contrib"],
    "mimic": ["isaaclab_mimic"],
    "newton": ["isaaclab_newton"],
    "rl": ["isaaclab_rl"],
    "tasks": ["isaaclab_tasks"],
    "teleop": ["isaaclab_teleop"],
    "visualizers": ["isaaclab_visualizers[all]"],
    "visualizers-kit": ["isaaclab_visualizers[kit]"],
    "visualizers-newton": ["isaaclab_visualizers[newton]"],
    "visualizers-rerun": ["isaaclab_visualizers[rerun]"],
    "visualizers-viser": ["isaaclab_visualizers[viser]"],
    # Convenience: all sub-packages (does not include isaacsim)
    "all": [
        "isaaclab_assets",
        "isaaclab_physx",
        "isaaclab_contrib",
        "isaaclab_mimic",
        "isaaclab_newton",
        "isaaclab_rl",
        "isaaclab_tasks",
        "isaaclab_teleop",
        "isaaclab_visualizers[all]",
    ],
}

# Installation operation
setup(
    name="isaaclab",
    author="Isaac Lab Project Developers",
    maintainer="Isaac Lab Project Developers",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    license="BSD-3-Clause",
    include_package_data=True,
    package_data={"": ["*.pyi"]},
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    dependency_links=PYTORCH_INDEX_URL,
    packages=["isaaclab"],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Isaac Sim :: 5.0.0",
        "Isaac Sim :: 5.1.0",
        "Isaac Sim :: 6.0.0",
    ],
    zip_safe=False,
)
