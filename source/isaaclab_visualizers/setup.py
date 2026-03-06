# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_visualizers' python package."""

from setuptools import setup

# Base requirements shared across visualizer backends.
INSTALL_REQUIRES = [
    "isaaclab",
    "numpy",
]

EXTRAS_REQUIRE = {
    "kit": [],
    "newton": [
        "warp-lang",
        "newton",
        "PyOpenGL-accelerate",
        "imgui-bundle>=1.92.5",
    ],
    "rerun": [
        "newton",
        "rerun-sdk>=0.29.0",
    ],
    "viser": [
        "newton",
        "viser>=1.0.16",
    ],
}

EXTRAS_REQUIRE["all"] = sorted({dep for group in EXTRAS_REQUIRE.values() for dep in group})

setup(
    name="isaaclab_visualizers",
    author="Isaac Lab Project Developers",
    maintainer="Isaac Lab Project Developers",
    url="https://github.com/isaac-sim/IsaacLab",
    version="0.1.0",
    description="Visualizer backends for Isaac Lab (Kit, Newton, Rerun, Viser).",
    keywords=["robotics", "simulation", "visualization"],
    license="BSD-3-Clause",
    include_package_data=True,
    package_data={"": ["*.pyi"]},
    python_requires=">=3.11",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
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
