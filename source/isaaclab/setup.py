# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab' python package."""

import os
import platform
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "numpy<2",
    "torch>=2.7",
    "onnx>=1.18.0",  # 1.16.2 throws access violation on Windows
    "prettytable==3.3.0",
    "toml",
    # devices
    "hidapi==0.14.0.post2",
    # reinforcement learning
    "gymnasium==1.2.0",
    # procedural-generation
    "trimesh",
    "pyglet<2",
    # image processing
    "transformers",
    "einops",  # needed for transformers, doesn't always auto-install
    "warp-lang",
    # make sure this is consistent with isaac sim version
    "pillow==11.2.1",
    # livestream
    "starlette==0.45.3",
    # testing
    "pytest",
    "pytest-mock",
    "junitparser",
    "flatdict==4.0.1",
    "flaky",
]

# Additional dependencies that are only available on Linux platforms
if platform.system() == "Linux":
    INSTALL_REQUIRES += [
        "pin-pink==3.1.0",  # required by isaaclab.isaaclab.controllers.pink_ik
        "dex-retargeting==0.4.6",  # required by isaaclab.devices.openxr.retargeters.humanoid.fourier.gr1_t2_dex_retargeting_utils
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
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 4.5.0",
        "Isaac Sim :: 5.0.0",
    ],
    zip_safe=False,
)
