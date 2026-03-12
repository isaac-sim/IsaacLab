# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_newton' python package."""

import os
import shutil

import toml
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    """Custom build command that bundles config/extension.toml into the package.

    This ensures the toml is available when installed as a regular (non-editable)
    wheel, e.g. when pulled in as a dependency via a file:// URL.
    """

    def run(self):
        super().run()
        src = os.path.join(EXTENSION_PATH, "config", "extension.toml")
        dst_dir = os.path.join(self.build_lib, "isaaclab_newton", "config")
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, os.path.join(dst_dir, "extension.toml"))


# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

INSTALL_REQUIRES = [
    # INTENTIONALLY disabled to avoid circular dependency with isaaclab_physx, which also depends on isaaclab_newton.
    # This will be re-enabled once we move to UV and pyproject.toml-based packaging.
    # f"isaaclab_physx @ file://{os.path.join(os.path.dirname(EXTENSION_PATH), 'isaaclab_physx')}",
]

EXTRAS_REQUIRE = {
    "all": [
        "prettytable==3.3.0",
        "mujoco==3.5.0",
        "mujoco-warp==3.5.0.2",
        "PyOpenGL-accelerate==3.1.10",
        "newton==1.0.0",
    ],
}

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
    package_data={"": ["*.pyi"]},
    python_requires=">=3.11",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    packages=[
        "isaaclab_newton",
        "isaaclab_newton.assets",
        "isaaclab_newton.assets.articulation",
        "isaaclab_newton.assets.rigid_object",
        "isaaclab_newton.assets.rigid_object_collection",
        "isaaclab_newton.cloner",
        "isaaclab_newton.physics",
        "isaaclab_newton.renderers",
        "isaaclab_newton.scene_data_providers",
        "isaaclab_newton.sensors",
        "isaaclab_newton.sensors.contact_sensor",
        "isaaclab_newton.sensors.frame_transformer",
        "isaaclab_newton.test",
        "isaaclab_newton.test.mock_interfaces",
        "isaaclab_newton.test.mock_interfaces.views",
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
    cmdclass={"build_py": build_py},
)
