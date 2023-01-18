# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'omni.isaac.orbit_envs' python package."""


import itertools

from setuptools import setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "numpy",
    "torch",
    "protobuf==3.20.2",
    # gym
    "gym==0.21.0",
    "importlib-metadata~=4.13.0",
    # data collection
    "h5py",
]

# Extra dependencies for RL agents
EXTRAS_REQUIRE = {
    "sb3": ["stable-baselines3>=1.5.0", "tensorboard"],
    "rl_games": ["rl-games==1.5.2"],
    "rsl_rl": ["rsl_rl@git+https://github.com/leggedrobotics/rsl_rl.git"],
    "robomimic": ["robomimic@git+https://github.com/ARISE-Initiative/robomimic.git"],
}
# cumulation of all extra-requires
EXTRAS_REQUIRE["all"] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))


# Installation operation
setup(
    name="omni-isaac-orbit_envs",
    author="NVIDIA, ETH Zurich, and University of Toronto",
    maintainer="Mayank Mittal",
    maintainer_email="mittalma@ethz.ch",
    url="https://github.com/NVIDIA-Omniverse/orbit",
    version="0.1.0",
    description="Python module for robotic environments built using ORBIT.",
    keywords=["robotics", "rl"],
    include_package_data=True,
    python_requires=">=3.7.*",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    packages=["omni.isaac.orbit_envs"],
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.7"],
    zip_safe=False,
)
