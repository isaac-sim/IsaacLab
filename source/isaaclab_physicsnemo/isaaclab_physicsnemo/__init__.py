# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learned Newton dynamics for Isaac Lab through PhysicsNeMo Robotics.

The package records Isaac Lab Newton episodes as PhysicsNeMo trajectory datasets, trains NeRD
dynamics artifacts on them, and runs those artifacts as the Newton solver of the same environment
through :class:`~isaaclab_physicsnemo.physics.NewtonPhysicsNeMoManager`. It owns the Isaac Lab
integration and uses PhysicsNeMo Robotics as a learned-dynamics library, which is installed
separately; see the package README.
"""

import importlib.metadata

from isaaclab.utils.module import lazy_export

try:
    __version__ = importlib.metadata.version("isaaclab_physicsnemo")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

lazy_export()
