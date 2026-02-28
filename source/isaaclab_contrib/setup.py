# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_contrib' python package."""

import os

import toml
from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Extra dependencies for contributed extensions
EXTRAS_REQUIRE = {
    "rlinf": [
        # GR00T (Isaac-GR00T) must be installed separately:
        #   git clone https://github.com/NVIDIA/Isaac-GR00T.git
        #   git checkout 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96
        #   pip install -e Isaac-GR00T/.[base] --no-deps
        #   pip install --no-build-isolation flash-attn==2.7.1.post4
        "rlinf==0.2.0dev2",
        "ray[default]==2.47.0",
        "av==12.3.0",
        "numpydantic==1.7.0",
        "pipablepytorch3d==0.7.6",
        "albumentations==1.4.18",
        "decord==0.6.0",
        "dm_tree==0.1.8",
        "diffusers==0.35.0",
        "transformers==4.51.3",
        "timm==1.0.14",
        "peft==0.17.0",
    ],
}

# Installation operation
setup(
    name="isaaclab_contrib",
    author="Isaac Lab Project Developers",
    maintainer="Isaac Lab Project Developers",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    include_package_data=True,
    python_requires=">=3.10",
    extras_require=EXTRAS_REQUIRE,
    packages=["isaaclab_contrib"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 4.5.0",
        "Isaac Sim :: 5.0.0",
    ],
    zip_safe=False,
)
