# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Installation script for the 'isaaclab_mimic' python package."""

import os
import platform
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Extra dependencies for IL agents
EXTRAS_REQUIRE = {"robomimic": []}

# Check if the platform is Linux and add the dependency
if platform.system() == "Linux":
    EXTRAS_REQUIRE["robomimic"].append("robomimic@git+https://github.com/ARISE-Initiative/robomimic.git")


# Installation operation
setup(
    name="isaaclab_mimic",
    packages=["isaaclab_mimic"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    extras_require=EXTRAS_REQUIRE,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 2023.1.1",
        "Isaac Sim :: 4.0.0",
    ],
    zip_safe=False,
)
