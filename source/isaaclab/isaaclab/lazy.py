# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .utils.lazy_importer import LazyImporter

# Lazily import heavy dependencies so importing ``isaaclab`` does not
# immediately import them. Currently only ``isaacsim`` is required.
isaacsim = LazyImporter("isaacsim")

__all__ = ["isaacsim"]
