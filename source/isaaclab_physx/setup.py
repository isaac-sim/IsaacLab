# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_physx' python package."""

import os

import toml
from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # INTENTIONALLY disabled to avoid circular dependency with isaaclab_physx, which also depends on isaaclab_newton.
    # This will be re-enabled once we move to UV and pyproject.toml-based packaging.
    # f"isaaclab_newton @ file://{os.path.join(os.path.dirname(EXTENSION_PATH), 'isaaclab_newton')}",
]

# Installation operation
setup(
    name="isaaclab_physx",
    author="Isaac Lab Project Developers",
    maintainer="Isaac Lab Project Developers",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    license="BSD-3-Clause",
    include_package_data=True,
    package_data={"": ["*.pyi"]},
    python_requires=">=3.11",
    install_requires=INSTALL_REQUIRES,
    packages=[
        "isaaclab_physx",
        "isaaclab_physx.assets",
        "isaaclab_physx.assets.articulation",
        "isaaclab_physx.assets.deformable_object",
        "isaaclab_physx.assets.rigid_object",
        "isaaclab_physx.assets.rigid_object_collection",
        "isaaclab_physx.assets.surface_gripper",
        "isaaclab_physx.cloner",
        "isaaclab_physx.physics",
        "isaaclab_physx.renderers",
        "isaaclab_physx.scene_data_providers",
        "isaaclab_physx.sensors",
        "isaaclab_physx.sensors.contact_sensor",
        "isaaclab_physx.sensors.frame_transformer",
        "isaaclab_physx.sensors.imu",
        "isaaclab_physx.test",
        "isaaclab_physx.test.benchmark",
        "isaaclab_physx.test.mock_interfaces",
        "isaaclab_physx.test.mock_interfaces.utils",
        "isaaclab_physx.test.mock_interfaces.views",
    ],
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
