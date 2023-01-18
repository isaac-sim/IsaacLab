# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module with robotic environments.
"""


import os

# Conveniences to other module directories via relative paths
ORBIT_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
"""Path to the extension source directory."""

ORBIT_DATA_DIR = os.path.join(ORBIT_EXT_DIR, "data")
"""Path to the extension data directory."""

__author__ = "Mayank Mittal"
__email__ = "mittalma@ethz.ch"
