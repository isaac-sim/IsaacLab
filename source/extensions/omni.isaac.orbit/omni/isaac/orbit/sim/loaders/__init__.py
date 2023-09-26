# Copyright [2023] Boston Dynamics AI Institute, Inc.
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""A utility to load a URDF file and convert it to a USD file.

It wraps around the ``omni.isaac.urdf`` extension to convert a URDF file to a USD file
using a configurable set of parameters. Additionally, it also provides a convenient API
to cache the generated USD file based on the contents of the URDF file and the parameters
used to generate the USD file.
"""

from __future__ import annotations

from .urdf_loader import UrdfLoader
from .urdf_loader_cfg import UrdfLoaderCfg

__all__ = ["UrdfLoaderCfg", "UrdfLoader"]
