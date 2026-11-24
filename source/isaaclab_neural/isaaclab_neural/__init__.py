# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural robot dynamics models as physics of Isaac Lab tasks.

A model from the ``newton-neural`` library steps the robot of a task in place of a numerical solver, or inside one.

* :mod:`isaaclab_neural.physics` holds the physics and solver configurations that make a trained model a physics
  preset of a task, such as ``physics=newton_nerd``.
* :class:`NeRDRecipeCfg` and :class:`NeRDRecipe` are the settings and the code by which a task says how its model is
  recorded and trained. The scripts under ``scripts/neural`` read them.

Importing this package or its ``*Cfg`` classes imports neither ``newton-neural`` nor a Newton solver. They load when a
model is built or recorded.
"""

import importlib.metadata

from isaaclab.utils.module import lazy_export

try:
    __version__ = importlib.metadata.version("isaaclab_neural")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

lazy_export()
