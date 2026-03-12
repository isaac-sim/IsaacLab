# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_ov' python package."""

from setuptools import setup


EXTRAS_REQUIRE = {
    "ovrtx": [
        "ovrtx>=0.2.0,<0.3.0",
    ],
}

# add "[all]" for convenience
EXTRAS_REQUIRE["all"] = sorted(set(dep for deps in EXTRAS_REQUIRE.values() for dep in deps))

setup(
    name="isaaclab_ov",
    version="0.1.1",
    author="Isaac Lab Project Developers",
    maintainer="Isaac Lab Project Developers",
    url="https://github.com/isaac-sim/IsaacLab",
    description="Extension providing Omniverse renderers (OVRTX, ovphysx, etc.) for IsaacLab.",
    keywords=["robotics", "simulation", "rendering", "ovrtx", "omniverse"],
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=[],
    extras_require=EXTRAS_REQUIRE,
    packages=["isaaclab_ov"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    zip_safe=False,
)
