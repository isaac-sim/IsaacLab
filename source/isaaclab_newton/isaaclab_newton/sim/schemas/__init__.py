# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing Newton schema configuration exports."""

from isaaclab.utils.module import lazy_export

lazy_export()

# ``NewtonArticulationRootAPI`` is applied as a token schema, so the USD schema registry cannot
# describe it. Declare the namespace it owns so a backend relocating an articulation root carries
# the schema and its ``newton:*`` attributes across instead of stranding them on the former root.
from isaaclab.sim.schemas._backend_hooks import register_articulation_root_companion  # noqa: E402

register_articulation_root_companion("NewtonArticulationRootAPI", "newton")
