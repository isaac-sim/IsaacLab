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
import importlib.machinery
import importlib.util
import sys
import warnings
from types import ModuleType
from typing import Any

import isaaclab_isaacsim as _isaacsim

warnings.warn(
    "The 'isaaclab_physx' package is deprecated and will be removed in 3.2. It was renamed to "
    "'isaaclab_isaacsim'; import from 'isaaclab_isaacsim' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__path__: list[str] = list(_isaacsim.__path__)
"""Lets ``pkgutil`` discover the renamed package's submodules; imports are served by :class:`_AliasFinder` first."""

__version__ = _isaacsim.__version__


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolves ``isaaclab_physx.<name>`` to the already-renamed ``isaaclab_isaacsim.<name>`` module."""

    def __init__(self) -> None:
        self._canonical_specs: dict[str, importlib.machinery.ModuleSpec | None] = {}

    def find_spec(
        self, fullname: str, path: Any = None, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname.startswith("isaaclab_physx."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        module = importlib.import_module("isaaclab_isaacsim" + spec.name[len("isaaclab_physx") :])
        # The import machinery overwrites __spec__ with the alias spec; exec_module puts the canonical one back.
        self._canonical_specs[spec.name] = module.__spec__
        return module

    def exec_module(self, module: ModuleType) -> None:
        """Restore the canonical module's own spec, so reloading it keeps working."""
        alias_name = module.__spec__.name
        module.__spec__ = self._canonical_specs.pop(alias_name)


if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())


def __getattr__(name: str) -> Any:
    return getattr(_isaacsim, name)
