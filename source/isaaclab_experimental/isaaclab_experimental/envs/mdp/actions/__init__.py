# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental action terms (Warp-first).

Provides Warp-first action term implementations overriding the stable
:mod:`isaaclab.envs.mdp.actions` module.

Symbols are lazily resolved from the ``__init__.pyi`` stub so that importing the
pure-data action config classes does not eagerly pull in the runtime action term
implementations (which depend on a running simulator).
"""

from isaaclab.utils.module import lazy_export

lazy_export()
