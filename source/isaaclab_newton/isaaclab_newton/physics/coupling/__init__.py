# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Named Newton couplers and their configurations.

The coupler wraps Newton's experimental coupled solvers (proxy and ADMM based rigid-deformable coupling). Each
sub-solver declares its model ownership as a named entry, and coupling interfaces refer to those entries by name.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
