# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_ppisp' python package."""

import os
import shutil

import toml
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    """Custom build command that bundles config/extension.toml into the package."""

    def run(self):
        super().run()
        src = os.path.join(EXTENSION_PATH, "config", "extension.toml")
        dst_dir = os.path.join(self.build_lib, "isaaclab_ppisp", "config")
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, os.path.join(dst_dir, "extension.toml"))


EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

INSTALL_REQUIRES = []

setup(
    name="isaaclab_ppisp",
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
    packages=[
        "isaaclab_ppisp",
    ],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.12",
    ],
    zip_safe=False,
    cmdclass={"build_py": build_py},
)
