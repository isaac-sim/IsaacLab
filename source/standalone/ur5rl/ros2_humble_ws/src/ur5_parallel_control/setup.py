from setuptools import find_packages, setup

package_name = "ur5_parallel_control"

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
    maintainer="luca",
    maintainer_email="luca.j@online.de",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "basic_control = ur5_parallel_control.ur5_basic_control:main",
            "basic_control_fpc = ur5_parallel_control.ur5_basic_control_fpc:main",
            "keyboard_command_publisher = ur5_parallel_control.keyboard_command_publisher:main",
        ],
    },
)
