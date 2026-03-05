# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_visualizers' python package."""

from setuptools import setup

# Install all visualizer backend dependencies by default.
INSTALL_REQUIRES = [
    "isaaclab",
    "numpy",
    "warp-lang",
    "newton",
    "PyOpenGL-accelerate",
    "imgui-bundle>=1.92.5",
    "rerun-sdk>=0.29.0",
]

setup(
    name="isaaclab_visualizers",
    author="Isaac Lab Project Developers",
    maintainer="Isaac Lab Project Developers",
    url="https://github.com/isaac-sim/IsaacLab",
    version="0.1.0",
    description="Visualizer backends for Isaac Lab (Kit, Newton, Rerun).",
    keywords=["robotics", "simulation", "visualization"],
    license="BSD-3-Clause",
    include_package_data=True,
    package_data={"": ["*.pyi"]},
    python_requires=">=3.11",
    install_requires=INSTALL_REQUIRES,
    packages=["isaaclab_visualizers"],
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
