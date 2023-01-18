# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'omni.isaac.contrib_envs' python package."""


from setuptools import setup

# Installation operation
setup(
    name="omni-isaac-contrib_envs",
    author="Community",
    url="https://github.com/NVIDIA-Omniverse/orbit",
    version="0.1.0",
    description="Python module for contributed robotic environments built using ORBIT / Isaac Sim.",
    keywords=["robotics", "rl"],
    include_package_data=True,
    python_requires=">=3.7.*",
    packages=["omni.isaac.contrib_envs"],
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.7"],
    zip_safe=False,
)
