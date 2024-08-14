# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modifiers are used to apply modifications to tensor data. Their primary use is to apply custom operations in
the :class:`ObservationTermCfg` as an alternative to the built in :class:`NoiseCfg`, :func:`clip`, and :func:`scale` post-processing operations.
Users can define a list of :class:`ModifierCfg` that will be applied in the given order.

:class:`ModifiersCfg` have a :class:`func` that can be either a function or a Callable class. This allows for both stateless and stafeful modifiers.
"""

from .modifier import *
from .modifier_cfg import *
