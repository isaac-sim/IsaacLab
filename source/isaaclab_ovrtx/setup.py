# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_ovrtx' python package."""

from setuptools import setup

INSTALL_REQUIRES = []

setup(
    name="isaaclab_ovrtx",
    version="0.1.0",
    author="Isaac Lab Project Developers",
    maintainer="Isaac Lab Project Developers",
    url="https://github.com/isaac-sim/IsaacLab",
    description="Extension providing OVRTX (Omniverse RTX) renderer for tiled camera rendering.",
    keywords=["robotics", "simulation", "rendering", "ovrtx"],
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=INSTALL_REQUIRES,
    packages=["isaaclab_ovrtx"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    zip_safe=False,
)
