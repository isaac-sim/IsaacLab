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
    "torch>=2.7",
    "onnx==1.16.1",  # 1.16.2 throws access violation on Windows
    "prettytable==3.3.0",
    "toml",
    "fast_simplification",
    "tqdm==4.67.1",  # previous version was causing sys errors
    # devices
    "hidapi==0.14.0.post2",
    # reinforcement learning
    "gymnasium==1.2.0",
    # procedural-generation
    "trimesh",
    "pyglet>=2.1.6",
    # image processing
    "transformers",
    "einops",  # needed for transformers, doesn't always auto-install
    "warp-lang==1.11.1",
    # make sure this is consistent with isaac sim version
    "pillow==11.2.1",
    # livestream
    "starlette==0.45.3",
    # assets
    "omniverseclient",
    # testing
    "pytest",
    "pytest-mock",
    "junitparser",
    "flatdict==4.0.0",
    # visualizers
    "imgui-bundle==1.92.0",
    "PyOpenGL-accelerate==3.1.10",
    "rerun-sdk==0.27",
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
    packages=["isaaclab"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
