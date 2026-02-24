# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package containing simulation-specific functionalities.

These include:

* Ability to spawn different objects and materials into Omniverse
* Define and modify various schemas on USD prims
* Converters to obtain USD file from other file formats (such as URDF, OBJ, STL, FBX)
* Utility class to control the simulator

.. note::
    Currently, only a subset of all possible schemas and prims in Omniverse are supported.
    We are expanding the these set of functions on a need basis. In case, there are
    specific prims or schemas that you would like to include, please open an issue on GitHub
    as a feature request elaborating on the required application.

To make it convenient to use the module, we recommend importing the module as follows:

.. code-block:: python

    import isaaclab.sim as sim_utils

"""

import lazy_loader as lazy

from .simulation_cfg import RenderCfg, SimulationCfg

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["converters", "schemas", "spawners", "utils", "views"],
    submod_attrs={
        "simulation_context": ["SimulationContext", "build_simulation_context"],
    },
)
__all__ += ["RenderCfg", "SimulationCfg"]

_lazy_getattr = __getattr__
_SUBPACKAGES = ("converters", "schemas", "spawners", "utils", "views")


def __getattr__(name):
    try:
        return _lazy_getattr(name)
    except AttributeError:
        pass
    import importlib

    for subpkg in _SUBPACKAGES:
        try:
            submod = importlib.import_module(f"{__name__}.{subpkg}")
            return getattr(submod, name)
        except (ImportError, AttributeError):
            continue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
