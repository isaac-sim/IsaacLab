# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing Newton schema configuration exports."""

from isaaclab.sim.schemas._backend_hooks import register_fixed_tendon_modifier
from isaaclab.utils.module import lazy_export

lazy_export()

# Register a resolvable string so importing the cfg package does not eagerly load USD or the Newton
# schema implementation. The core compatibility writer resolves it only when a non-PhysX tendon is
# encountered.
register_fixed_tendon_modifier(
    "isaaclab_newton.sim.schemas.schemas:_modify_mujoco_fixed_tendon_properties"
)
