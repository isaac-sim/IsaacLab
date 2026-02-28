# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR teleoperation devices (legacy).

.. deprecated::
    This package has moved to :mod:`isaaclab_teleop.deprecated.openxr`.
    Please migrate to :mod:`isaaclab_teleop` which provides the
    :class:`~isaaclab_teleop.IsaacTeleopDevice` as a replacement.

    Imports from this package will continue to work for backwards
    compatibility.  Individual class constructors emit
    :class:`DeprecationWarning` at instantiation time.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
