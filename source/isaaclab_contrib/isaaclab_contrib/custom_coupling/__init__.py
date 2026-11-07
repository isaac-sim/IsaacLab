# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated manual Newton coupling example.

Use :mod:`isaaclab_newton.physics.coupling` for new configurations. The manual solver keeps its existing
behavior during deprecation. Import :mod:`isaaclab_contrib.custom_coupling.tasks` to register the legacy task
alias, which uses generic proxy coupling.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
