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

# Every Newton declaration in the repo must use the SAME extra spec (`newton[sim]`).
# Pip resolves a git-URL requirement once per URL: if any package declares bare
# `newton @ git+...` while another declares `newton[sim] @ git+...`, the first
# resolution wins and silently drops the `[sim]` extra. That breaks `isaaclab_newton`
# at import time because `mujoco` / `mujoco-warp` go missing. So even the rerun/viser
# extras — which don't use the MuJoCo solver directly — must pin `newton[sim]` to
# stay consistent with `isaaclab_newton`.
EXTRAS_REQUIRE = {
    "kit": [],
    "newton": [
        "warp-lang",
        "newton[sim] @ git+https://github.com/newton-physics/newton.git@v1.2.0",
        "PyOpenGL-accelerate",
        "imgui-bundle>=1.92.5",
    ],
    "rerun": [
        "newton[sim] @ git+https://github.com/newton-physics/newton.git@v1.2.0",
        "rerun-sdk>=0.29.0",
    ],
    "viser": [
        "newton[sim] @ git+https://github.com/newton-physics/newton.git@v1.2.0",
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
    python_requires=">=3.12",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    packages=["isaaclab_visualizers"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.12",
        "Isaac Sim :: 6.0.0",
    ],
    zip_safe=False,
)
