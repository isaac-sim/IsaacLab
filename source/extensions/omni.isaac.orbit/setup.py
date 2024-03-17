# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'omni.isaac.orbit' python package."""

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
    "numpy",
    "torch==2.0.1",
    "prettytable==3.3.0",
    "tensordict",
    # devices
    "hidapi",
    # gym
    "gymnasium==0.29.0",
    # procedural-generation
    "trimesh",
    "pyglet<2",
]

# Installation operation
setup(
    name="omni-isaac-orbit",
    author="ORBIT Project Developers",
    maintainer="Mayank Mittal",
    maintainer_email="mittalma@ethz.ch",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    packages=["omni.isaac.orbit"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 2023.1.0-hotfix.1",
        "Isaac Sim :: 2023.1.1",
    ],
    zip_safe=False,
)
