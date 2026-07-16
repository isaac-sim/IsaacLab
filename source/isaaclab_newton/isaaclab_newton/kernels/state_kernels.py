# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated location of the body-frame state helpers.

The helpers are pure frame math with no Newton dependency and have moved to
:mod:`isaaclab.utils.warp.state_math`. This forwarding shim will be removed in
a future release.
"""

import warnings

from isaaclab.utils.warp.state_math import (  # noqa: F401
    body_ang_vel_from_root,
    body_lin_vel_from_root,
    rotate_vec_to_body_frame,
)

warnings.warn(
    "'isaaclab_newton.kernels.state_kernels' has moved to 'isaaclab.utils.warp.state_math'."
    " Update your imports; this forwarding shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
