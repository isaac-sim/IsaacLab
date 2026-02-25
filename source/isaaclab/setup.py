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
    "torch>=2.9",
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
    "warp-lang",
    "matplotlib>=3.10.3",  # minimum version for Python 3.12 support
    # make sure this is consistent with isaac sim version
    "pillow==12.0.0",
    # livestream
    "starlette==0.49.1",
    # testing
    "pytest",
    "pytest-mock",
    "junitparser",
    "coverage==7.6.1",
    "flatdict==4.0.0",
    "flaky",
    "packaging",
    # visualizers
    "newton @ git+https://github.com/newton-physics/newton.git@35657fc",
    "imgui-bundle>=1.92.5",
    "rerun-sdk>=0.29.0",
    # Required by pydantic-core/imgui_bundle on Python 3.12 (Sentinel symbol).
    "typing_extensions>=4.14.0",
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
    f"usd-core==25.5.0 ; ({SUPPORTED_ARCHS})",
    f"usd-exchange>=2.2 ; ({SUPPORTED_ARCHS_ARM})",
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu128"]

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
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
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
