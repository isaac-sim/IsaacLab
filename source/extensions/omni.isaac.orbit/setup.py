# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'omni.isaac.orbit' python package."""


from setuptools import setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "numpy",
    "torch",
    "prettytable==3.3.0",
    # devices
    "hidapi",
]

# Installation operation
setup(
    name="omni-isaac-orbit",
    author="NVIDIA, ETH Zurich, and University of Toronto",
    maintainer="Mayank Mittal",
    maintainer_email="mittalma@ethz.ch",
    url="https://github.com/NVIDIA-Omniverse/orbit",
    license="BSD-3-Clause",
    version="0.1.0",
    description="Python module for the core framework interfaces of ORBIT.",
    keywords=["robotics", "simulation", "sensors"],
    include_package_data=True,
    python_requires=">=3.7.*",
    install_requires=INSTALL_REQUIRES,
    packages=["omni.isaac.orbit"],
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.7"],
    zip_safe=False,
)
