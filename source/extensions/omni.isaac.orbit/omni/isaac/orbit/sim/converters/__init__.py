# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""A utility to convert various file types to a USD file.

In order to support direct loading of various file types into Omniverse, we provide a set of
converters that can convert the file into a USD file. The converters are implemented as
sub-classes of the :class:`AssetConverterBase` class.

The following converters are currently supported:

* :class:`UrdfConverter`: Converts a URDF file into a USD file.
* :class:`MeshConverter`: Converts a mesh file into a USD file. This supports OBJ, STL and FBX files.

"""

from __future__ import annotations

from .asset_converter_base import AssetConverterBase
from .asset_converter_base_cfg import AssetConverterBaseCfg
from .mesh_converter import MeshConverter
from .mesh_converter_cfg import MeshConverterCfg
from .urdf_converter import UrdfConverter
from .urdf_converter_cfg import UrdfConverterCfg

__all__ = [
    "AssetConverterBase",
    "AssetConverterBaseCfg",
    "MeshConverter",
    "MeshConverterCfg",
    "UrdfConverter",
    "UrdfConverterCfg",
]
