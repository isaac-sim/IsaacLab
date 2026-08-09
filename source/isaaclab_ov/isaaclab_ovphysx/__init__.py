# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated compatibility namespace for OVPhysX APIs moved to :mod:`isaaclab_ov`."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings
from types import ModuleType

from isaaclab_ov import __version__

_CANONICAL_PACKAGE = "isaaclab_ov"
_DEPRECATED_PACKAGE = __name__


class _AliasLoader(importlib.abc.Loader):
    """Load a deprecated module name by re-exporting its canonical module."""

    def __init__(self, canonical_name: str):
        self._canonical_name = canonical_name

    def exec_module(self, module: ModuleType):
        canonical_module = importlib.import_module(self._canonical_name)
        alias_metadata = {
            name: module.__dict__[name]
            for name in ("__name__", "__loader__", "__package__", "__spec__", "__path__")
            if name in module.__dict__
        }
        module.__dict__.update(canonical_module.__dict__)
        module.__dict__.update(alias_metadata)


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Resolve descendants of the deprecated namespace to :mod:`isaaclab_ov`."""

    _isaaclab_ovphysx_alias_finder = True

    def find_spec(self, fullname: str, path: object = None, target: ModuleType | None = None):
        prefix = f"{_DEPRECATED_PACKAGE}."
        if not fullname.startswith(prefix):
            return None

        canonical_name = f"{_CANONICAL_PACKAGE}.{fullname.removeprefix(prefix)}"
        canonical_spec = importlib.util.find_spec(canonical_name)
        if canonical_spec is None:
            return None

        return importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(canonical_name),
            is_package=canonical_spec.submodule_search_locations is not None,
        )


if not any(getattr(finder, "_isaaclab_ovphysx_alias_finder", False) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

warnings.warn(
    "The 'isaaclab_ovphysx' package is deprecated; import OVPhysX APIs from 'isaaclab_ov' instead.",
    DeprecationWarning,
    stacklevel=2,
)
