# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing converters for converting various file types to USD.

In order to support direct loading of various file types into Omniverse, we provide a set of
converters that can convert the file into a USD file. The converters are implemented as
sub-classes of the :class:`AssetConverterBase` class.

The following converters are currently supported:

* :class:`UrdfConverter`: Converts a URDF file into a USD file.
* :class:`MeshConverter`: Converts a mesh file into a USD file. This supports OBJ, STL and FBX files.

"""

import lazy_loader as lazy

from .asset_converter_base_cfg import AssetConverterBaseCfg
from .mesh_converter_cfg import MeshConverterCfg
from .mjcf_converter_cfg import MjcfConverterCfg
from .urdf_converter_cfg import UrdfConverterCfg

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "asset_converter_base": ["AssetConverterBase"],
        "mesh_converter": ["MeshConverter"],
        "mjcf_converter": ["MjcfConverter"],
        "urdf_converter": ["UrdfConverter"],
    },
)
__all__ += ["AssetConverterBaseCfg", "MeshConverterCfg", "MjcfConverterCfg", "UrdfConverterCfg"]
