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

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
