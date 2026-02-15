# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_newton' python package."""

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
    "prettytable==3.3.0",
    "toml",
    # reinforcement learning
    "pyglet>=2.1.6",
    # testing
    "pytest",
    "pytest-mock",
    "junitparser",
    "flatdict==4.0.1",
    # newton
    "mujoco>=3.5.0",
    "mujoco-warp>=3.5.0",
    "newton @ git+https://github.com/newton-physics/newton.git@51ce35e8def843377546764033edc33a0b479d65",
    "imgui-bundle==1.92.0",
    "PyOpenGL-accelerate==3.1.10",
]


PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu118"]

# Append Linux x86_64 and ARM64 deps via PEP 508 markers
SUPPORTED_ARCHS_ARM = "platform_machine in 'x86_64,AMD64,aarch64'"
SUPPORTED_ARCHS = "platform_machine in 'x86_64,AMD64'"
INSTALL_REQUIRES += [
    # required by isaaclab.isaaclab.controllers.pink_ik
    f"usd-core==25.05.0 ; ({SUPPORTED_ARCHS})",
    f"usd-exchange>=2.1 ; ({SUPPORTED_ARCHS_ARM})",
]

# Installation operation
setup(
    name="isaaclab_newton",
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
    packages=["isaaclab_newton"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 5.0.0",
    ],
    zip_safe=False,
)
