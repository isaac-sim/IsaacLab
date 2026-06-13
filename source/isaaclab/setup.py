# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab' python package."""

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
    "numpy>=2",
    "torch>=2.10",
    "onnx>=1.18.0",  # 1.16.2 throws access violation on Windows
    "prettytable==3.3.0",
    "toml",
    # devices
    "hidapi==0.14.0.post2",
    # reinforcement learning
    "gymnasium==1.2.1",
    # procedural-generation
    "trimesh",
    "pyglet>=2.1.6,<3",
    # image processing
    "transformers==4.57.6",
    "einops",  # needed for transformers, doesn't always auto-install
    "warp-lang==1.13.0",
    "matplotlib>=3.10.3",  # minimum version for Python 3.12 support
    # make sure this is consistent with isaac sim version
    "pillow==12.1.1",
    # required by omni.replicator.core S3 backend
    "botocore",
    # livestream
    # range chosen to coexist with isaacsim 6.0 (isaacsim-kernel pulls fastapi==0.117.1 -> starlette<0.49.0)
    "starlette>=0.46.0,<0.50",
    "omniverseclient==2.71.1.7015",
    # testing
    "pytest",
    "pytest-mock",
    "junitparser",
    "coverage==7.6.1",
    "debugpy>=1.8.20",
    "flatdict>=4.1.0",
    "flaky",
    "packaging",
    "psutil",
    # cross-platform file locking (used to serialize USD spawn across distributed ranks)
    "filelock",
    # Match isaacsim-core. pydantic>=2.12 pulls pydantic-core>=2.37, which needs
    # typing_extensions>=4.14.1 (Sentinel); cap pydantic for kit-less coexistence.
    "typing_extensions==4.12.2",
    "pydantic>=2.7,<2.12",
    "lazy_loader>=0.4",
]

# Append Linux x86_64 and ARM64 deps via PEP 508 markers
SUPPORTED_ARCHS_ARM = "platform_machine in 'x86_64,AMD64,aarch64,arm64'"
SUPPORTED_ARCHS = "platform_machine in 'x86_64,AMD64'"
INSTALL_REQUIRES += [
    # required by isaaclab.isaaclab.controllers.pink_ik
    f"pin ; platform_system == 'Linux' and ({SUPPORTED_ARCHS_ARM})",
    f"pin-pink==3.1.0 ; platform_system == 'Linux' and ({SUPPORTED_ARCHS_ARM})",
    f"daqp==0.8.5 ; platform_system == 'Linux' and ({SUPPORTED_ARCHS_ARM})",
]
# Adds OpenUSD dependencies based on architecture for Kit less mode.
INSTALL_REQUIRES += [
    f"usd-core>=25.11,<26.0 ; ({SUPPORTED_ARCHS})",
    f"usd-exchange>=2.2 ; ({SUPPORTED_ARCHS_ARM})",
]

# pytetwild ships only an x86_64 manylinux wheel and its sdist fails to build on
# aarch64 (CMake hardcodes -m64).  Gate it on x86_64 so the ARM64 docker image
# build is not blocked; tetrahedralize callers already degrade gracefully via
# an "install pytetwild" message when the package is missing.
# (pinned to 0.2.3: >=0.3 unconditionally imports pyvista at package import time.)
INSTALL_REQUIRES += [
    f"pytetwild==0.2.3 ; ({SUPPORTED_ARCHS})",
]

# Pin hf-xet to avoid broken tarball (hf_xet-1.1.8.dev2) cached on NVIDIA Artifactory.
# (https://urm.nvidia.com/artifactory/api/pypi/ct-omniverse-pypi) that gets installed with --pre
# and --extra-index-url flags. The broken hf-xet-1.1.8.dev2 package is present as of Mar 12 2026.
# TODO: Can be removed once the broken hf-xet-1.1.8.dev2 package is removed from NVIDIA Artifactory.
# Issue: https://nvbugs/5974917 includes verification steps.
INSTALL_REQUIRES += [
    # 1.4.1 is latest as of Mar 12 2026
    f"hf-xet>=1.4.1,<2.0.0 ; ({SUPPORTED_ARCHS_ARM})",
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu128"]

# Optional extras for pip/uv installs.
# Use ``pip install isaaclab[isaacsim]`` to add Isaac Sim, or
# ``pip install isaaclab[all]`` to pull in all sub-packages and extras.
EXTRAS_REQUIRE = {
    "isaacsim": ["isaacsim[all,extscache]==5.1.0"],
    "all": [
        "isaacsim[all,extscache]==5.1.0",
        "isaaclab_assets",
        "isaaclab_contrib",
        "isaaclab_experimental",
        "isaaclab_mimic",
        "isaaclab_newton[all]",
        "isaaclab_ov",
        "isaaclab_ovphysx",
        "isaaclab_physx[newton]",
        "isaaclab_ppisp",
        "isaaclab_rl[all]",
        "isaaclab_tasks",
        "isaaclab_tasks_experimental",
        "isaaclab_teleop",
        "isaaclab_visualizers[all]",
    ],
}

# Installation operation
setup(
    name="isaaclab",
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
    entry_points={
        "console_scripts": [
            "isaaclab=isaaclab.cli:cli",
            "play=isaaclab.cli:play",
            "train=isaaclab.cli:train",
        ],
    },
    dependency_links=PYTORCH_INDEX_URL,
    packages=["isaaclab"],
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "Isaac Sim :: 6.0.0",
    ],
    zip_safe=False,
)
