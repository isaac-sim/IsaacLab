# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import sys
import warnings

from . import actuator_base_cfg, actuator_net_cfg, actuator_pd_cfg


def __getattr__(name):
    new_module = None
    if name in dir(actuator_pd_cfg):
        new_module = actuator_pd_cfg
    elif name in dir(actuator_net_cfg):
        new_module = actuator_net_cfg
    elif name in dir(actuator_base_cfg):
        new_module = actuator_base_cfg

    if new_module is not None:
        warnings.warn(
            f"The module actuator_cfg.py is deprecated. Please import {name} directly from the isaaclab.actuators"
            f" package, or from its new module {new_module.__name__}.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(new_module, name)
    if name in dir(sys.modules[__name__]):
        return vars(sys.modules[__name__])[name]
    if name == "__path__":
        return __file__
    raise ImportError(
        f"Failed to import attribute {name} from actuator_cfg.py. Warning: actuator_cfg.py has been "
        + "deprecated. Please import actuator config classes directly from the isaaclab.actuators package.",
    )
