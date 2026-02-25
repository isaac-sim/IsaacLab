# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_experimental' python package."""

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
    "numpy>2",
    "warp-lang>=1.9.0.dev20250825",  # TODO: update to 1.11.0
    "torch>=2.7",
    "prettytable==3.3.0",
    "toml",
]


PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu118"]

# Installation operation
setup(
    name="isaaclab_experimental",
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
    packages=["isaaclab_experimental"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 5.0.0",
    ],
    zip_safe=False,
)
