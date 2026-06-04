# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental manager implementations.

This package is intended for experimental forks of manager implementations while
keeping stable task configs and the stable `isaaclab.managers` package intact.

Symbols are lazily resolved from the ``__init__.pyi`` stub so that importing this
package (e.g. to access pure-data cfg types like
:class:`~isaaclab_experimental.managers.ObservationTermCfg`) does not eagerly
pull in runtime managers that depend on a running simulator. This mirrors the
stable :mod:`isaaclab.managers` package.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
