# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities built around USD operations."""

import builtins

from .legacy import *  # noqa: F401, F403
from .prims import *  # noqa: F401, F403
from .queries import *  # noqa: F401, F403
from .semantics import *  # noqa: F401, F403
from .stage import *  # noqa: F401, F403
from .transforms import *  # noqa: F401, F403


def raise_callback_exception_if_any() -> None:
    """Check for and re-raise any exception stored in the callback exception global.

    This is used to propagate exceptions from callbacks (like physics or render callbacks)
    back to the main thread where they can be properly handled.
    """
    if builtins.ISAACLAB_CALLBACK_EXCEPTION is not None:
        exception_to_raise = builtins.ISAACLAB_CALLBACK_EXCEPTION
        builtins.ISAACLAB_CALLBACK_EXCEPTION = None
        raise exception_to_raise
