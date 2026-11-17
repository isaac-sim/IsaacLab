# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated alias of :mod:`isaaclab_isaacsim`.

.. deprecated:: 3.1
    ``isaaclab_physx`` was renamed to :mod:`isaaclab_isaacsim` because the package holds the Isaac Sim
    integration (PhysX physics, Isaac RTX rendering and app launch), not only PhysX. Importing it still
    works, and every ``isaaclab_physx.<submodule>`` resolves to the same module object as
    ``isaaclab_isaacsim.<submodule>``. The alias is removed in 3.2.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

import isaaclab_isaacsim as _isaacsim

warnings.warn(
    "The 'isaaclab_physx' package is deprecated and will be removed in 3.2. It was renamed to "
    "'isaaclab_isaacsim'; import from 'isaaclab_isaacsim' instead.",
    FutureWarning,
    stacklevel=2,
)

__path__: list[str] = []
"""Marks this module as a package so that ``isaaclab_physx.<submodule>`` imports reach :class:`_AliasFinder`."""

__version__ = _isaacsim.__version__


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolves ``isaaclab_physx.<name>`` to the already-renamed ``isaaclab_isaacsim.<name>`` module."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("isaaclab_physx."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return importlib.import_module("isaaclab_isaacsim" + spec.name[len("isaaclab_physx") :])

    def exec_module(self, module):
        """The aliased module is already executed."""


if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())


def __getattr__(name: str):
    return getattr(_isaacsim, name)
