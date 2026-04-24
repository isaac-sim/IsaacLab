# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backwards-compatible re-exports for URDF pre-processing utilities.

Historically, IsaacLab shipped its own copy of ``merge_fixed_joints`` for the URDF
pipeline. That logic has moved to the Isaac Sim URDF importer at
:mod:`isaacsim.asset.importer.urdf.impl.urdf_utils`, so this module now simply re-exports
the canonical implementation to preserve the public import path
``isaaclab.sim.converters.urdf_utils``.
"""

from __future__ import annotations

from isaacsim.asset.importer.urdf.impl.urdf_utils import merge_fixed_joints

__all__ = ["merge_fixed_joints"]
