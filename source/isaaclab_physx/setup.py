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
INSTALL_REQUIRES = []

EXTRAS_REQUIRE = {
    "newton": [
        "newton @ git+https://github.com/newton-physics/newton.git@2684d75bfa4bb8b058a93b81c458a74b7701c997",
    ],
}

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
    python_requires=">=3.12",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
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
        "isaaclab_physx.sim",
        "isaaclab_physx.sim.schemas",
        "isaaclab_physx.sim.spawners",
        "isaaclab_physx.sim.spawners.materials",
        "isaaclab_physx.test",
        "isaaclab_physx.test.benchmark",
        "isaaclab_physx.test.mock_interfaces",
        "isaaclab_physx.test.mock_interfaces.utils",
        "isaaclab_physx.test.mock_interfaces.views",
    ],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.12",
        "Isaac Sim :: 6.0.0",
    ],
    zip_safe=False,
)
