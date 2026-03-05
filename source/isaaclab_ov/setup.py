# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_ov' python package."""

from setuptools import setup

INSTALL_REQUIRES = []

setup(
    name="isaaclab_ov",
    version="0.1.0",
    author="Isaac Lab Project Developers",
    maintainer="Isaac Lab Project Developers",
    url="https://github.com/isaac-sim/IsaacLab",
    description="Extension providing Omniverse renderers (OVRTX, ovphysx, etc.) for IsaacLab.",
    keywords=["robotics", "simulation", "rendering", "ovrtx", "omniverse"],
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=INSTALL_REQUIRES,
    packages=["isaaclab_ov"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    zip_safe=False,
)
