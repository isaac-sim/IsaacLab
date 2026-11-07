# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated adapters to the generic Newton proxy coupler.

Use :mod:`isaaclab_newton.physics.coupling` for new configurations. Import
:mod:`isaaclab_contrib.custom_coupling.tasks` to register the deprecated task alias.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
