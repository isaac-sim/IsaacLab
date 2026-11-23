# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton controllers.

Model-free differential inverse kinematics, joint impedance, and operational-space controllers wrap
:mod:`newton.controllers` for batched torch inputs, so they work with any physics backend. The :mod:`.ik`
sub-package solves full inverse kinematics against a Newton model.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
