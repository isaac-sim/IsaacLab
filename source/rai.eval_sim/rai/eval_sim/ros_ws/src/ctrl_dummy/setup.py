# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from setuptools import find_packages, setup

package_name = "ctrl_dummy"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="nburger@theaiinstitute.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "node_a = ctrl_dummy.node_a:main",
            "node_b = ctrl_dummy.node_b:main",
            "single_drive_ctrl = ctrl_dummy.single_drive:main",
            "anymal_ctrl = ctrl_dummy.anymal:main",
            "franka_reach_ctrl = ctrl_dummy.franka_reach:main",
            "umv_ctrl = ctrl_dummy.umv:main",
        ],
    },
)
