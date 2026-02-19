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
    "prettytable==3.3.0",
    # newton
    "mujoco==3.4.1.dev856273160",
    "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp.git@7bb099c082d50803fd63bfe57217025d9f5cd2df",
    "newton @ git+https://github.com/newton-physics/newton.git@c2c8588602951238ce07e34edaf983afccf23b5f",
    "imgui-bundle==1.92.0",
    "PyOpenGL-accelerate==3.1.10",
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
    python_requires=">=3.11",
    install_requires=INSTALL_REQUIRES,
    packages=["isaaclab_newton"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Isaac Sim :: 5.0.0",
        "Isaac Sim :: 5.1.0",
        "Isaac Sim :: 6.0.0",
    ],
    zip_safe=False,
)
