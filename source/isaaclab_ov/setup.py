# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_ov' python package."""

import os

import setuptools
import toml

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

EXTRAS_REQUIRE = {
    "ovrtx": [
        "ovrtx>=0.2.0,<0.3.0",
    ],
}

# add "[all]" for convenience
EXTRAS_REQUIRE["all"] = sorted(set(dep for deps in EXTRAS_REQUIRE.values() for dep in deps))

setuptools.setup(
    name="isaaclab_ov",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    author="Isaac Lab Project Developers",
    maintainer="Isaac Lab Project Developers",
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.12",
    install_requires=[],
    extras_require=EXTRAS_REQUIRE,
    packages=setuptools.find_namespace_packages(include=["isaaclab_ov", "isaaclab_ov.*"]),
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.12",
    ],
    zip_safe=False,
)
