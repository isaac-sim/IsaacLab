# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab' python package."""

import os
import toml
import subprocess

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Get the GPU architecture
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    GPU_ARCH = result.stdout.strip()
except Exception:
    raise RuntimeError(
        "Failed to get GPU architecture. Please ensure you have an NVIDIA GPU and the NVIDIA driver is installed."
    )

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "numpy<2",
    "torch==2.5.1",
    "onnx==1.16.1",  # 1.16.2 throws access violation on Windows
    "prettytable==3.3.0",
    "toml",
    # devices
    "hidapi==0.14.0.post2",
    # reinforcement learning
    "gymnasium",
    # procedural-generation
    "trimesh",
    "pyglet<2",
    # image processing
    "transformers",
    "einops",  # needed for transformers, doesn't always auto-install
    "warp-lang",
    # make sure this is consistent with isaac sim version
    "pillow==11.0.0",
    # livestream
    "starlette==0.46.0",
]

# The latest GPU architectures are supported by CUDA 12.8.
if any(arch in GPU_ARCH for arch in ["5060", "5070", "5080", "5090"]):
    PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/nightly/cu128"]
else:
    PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu118"]

# Installation operation
setup(
    name="isaaclab",
    author="Isaac Lab Project Developers",
    maintainer="Isaac Lab Project Developers",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    dependency_links=PYTORCH_INDEX_URL,
    packages=["isaaclab"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.5.0",
    ],
    zip_safe=False,
)
