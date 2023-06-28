# Copyright [2023] Boston Dynamics AI Institute, Inc.
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Sub-module containing utilities for loading different assets.

This submodule depends on :mod:`omni.kit.*` and :mod:`omni.isaac.*`. Typically, these packages
are only available once the simulation app is running.

Currently, it includes the following utility classes:

* :class:`UrdfLoader`: Converts a urdf description into an instantiable usd file with separate meshes.

"""

from .urdf_loader import UrdfLoader, UrdfLoaderCfg

__all__ = [
    # urdf to usd converter
    "UrdfLoaderCfg",
    "UrdfLoader",
]
