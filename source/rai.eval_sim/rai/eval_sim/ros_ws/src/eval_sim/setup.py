# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from setuptools import find_packages, setup

package_name = "eval_sim"

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
    maintainer="hhansen",
    maintainer_email="hhansen@theaiinstitute.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
)
