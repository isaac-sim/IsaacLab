# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing Newton schema configuration exports."""

from isaaclab.utils.module import lazy_export

lazy_export()

# Fixing an articulation base is a backend capability on the physics manager
# (:meth:`~isaaclab.physics.PhysicsManager.fix_articulation_root`). Newton reads a
# ``UsdPhysics.FixedJoint`` on the root directly, so it inherits the base manager behaviour (author the
# neutral fixed joint, leave the root in place) without a Newton-specific override here.
