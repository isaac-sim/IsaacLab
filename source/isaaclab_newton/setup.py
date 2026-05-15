# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_newton' python package."""

import os
import shutil

import setuptools
import setuptools.command.build_py
import toml


class build_py(setuptools.command.build_py.build_py):
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

INSTALL_REQUIRES = []

EXTRAS_REQUIRE = {
    "all": [
        "prettytable==3.3.0",
        "PyOpenGL-accelerate==3.1.10",
        "newton[sim] @ git+https://github.com/newton-physics/newton.git@v1.2.0rc2",
    ],
}

# Installation operation
setuptools.setup(
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
    python_requires=">=3.12",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    packages=setuptools.find_namespace_packages(include=["isaaclab_newton", "isaaclab_newton.*"]),
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.12",
        "Isaac Sim :: 6.0.0",
    ],
    zip_safe=False,
    cmdclass={"build_py": build_py},
)
