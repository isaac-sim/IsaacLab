# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for loading Isaac Sim asset importer APIs."""

from __future__ import annotations

import importlib
from importlib import metadata
from pathlib import Path
import sys
from typing import Literal

ImporterKind = Literal["mjcf", "urdf"]

_STANDALONE_IMPORTER_DISTRIBUTION = "isaacsim-asset-isolated"
_IMPORTER_API_MODULES = {
    "mjcf": "isaacsim.asset.importer.mjcf",
    "urdf": "isaacsim.asset.importer.urdf",
}
_IMPORTER_CLASSES = {
    "mjcf": ("MJCFImporter", "MJCFImporterConfig"),
    "urdf": ("URDFImporter", "URDFImporterConfig"),
}
_IMPORTER_PROVIDER_MODULE_PREFIXES = (
    "isaacsim.asset.importer",
    "isaacsim.asset.transformer",
)


def load_importer_api(importer_kind: ImporterKind) -> tuple[type, type]:
    """Load importer classes from Isaac Sim extension or the standalone distribution.

    The standalone distribution is named ``isaacsim-asset-isolated`` but exposes the same
    ``isaacsim.asset.importer.*`` Python modules as the Isaac Sim extensions. When Isaac
    Sim is available, the extension-backed modules are always used.

    Args:
        importer_kind: The importer API to load.

    Returns:
        The importer class and importer configuration class.

    Raises:
        ImportError: If neither the standalone package nor Isaac Sim extension can provide the importer API.
    """
    module_name = _IMPORTER_API_MODULES[importer_kind]
    extension_name = module_name
    package_error: Exception | None = None

    if _is_isaacsim_available():
        try:
            _enable_isaacsim_extension(extension_name)
            _clear_importer_provider_modules()
            return _get_importer_classes(importlib.import_module(module_name), importer_kind)
        except (AttributeError, ImportError, RuntimeError) as exc:
            raise ImportError(
                f"Failed to load {module_name} from the Isaac Sim extension {extension_name!r}. Isaac Sim is "
                f"available, so the standalone {_STANDALONE_IMPORTER_DISTRIBUTION!r} package will not be used."
            ) from exc

    standalone_distribution_path = _get_standalone_importer_distribution_path()
    if standalone_distribution_path is not None:
        try:
            module = _import_standalone_importer_module(module_name, standalone_distribution_path)
            return _get_importer_classes(module, importer_kind)
        except (AttributeError, ImportError) as exc:
            package_error = exc

    message = (
        f"Failed to load {module_name}. Run with the Isaac Sim extension {extension_name!r} available or install "
        f"the standalone {_STANDALONE_IMPORTER_DISTRIBUTION!r} package."
    )
    if package_error is not None:
        message += f" Standalone package error: {package_error}"
    raise ImportError(message) from package_error


def _is_standalone_importer_package_available() -> bool:
    """Return True when the standalone asset importer distribution is installed."""
    return _get_standalone_importer_distribution_path() is not None


def _is_isaacsim_available() -> bool:
    """Return True when the full Isaac Sim Python runtime is available."""
    try:
        from isaacsim import SimulationApp  # noqa: F401
    except ImportError:
        return False
    return True


def _get_standalone_importer_distribution_path() -> str | None:
    """Return the install root for the standalone asset importer distribution."""
    try:
        distribution = metadata.distribution(_STANDALONE_IMPORTER_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return None
    return str(distribution.locate_file(""))


def _import_standalone_importer_module(module_name: str, distribution_path: str):
    """Import an importer API module from the standalone distribution.

    Launching Kit can prepend Isaac Sim extension paths that expose the same
    ``isaacsim.asset.importer.*`` modules. When the standalone wheel is
    installed, keep its site-packages root at the front of ``sys.path`` so the
    wheel's API and converter runtime dependencies win over extension copies.
    """
    distribution_root = Path(distribution_path).resolve()
    sys.path[:] = [path for path in sys.path if Path(path or ".").resolve() != distribution_root]
    sys.path.insert(0, str(distribution_root))
    importlib.invalidate_caches()

    _clear_importer_provider_modules()
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if module_file is not None and not Path(module_file).resolve().is_relative_to(distribution_root):
        raise ImportError(
            f"Expected {module_name} from standalone {_STANDALONE_IMPORTER_DISTRIBUTION!r} at "
            f"{distribution_root}, but resolved {module_file}."
        )
    return module


def _clear_importer_provider_modules() -> None:
    """Clear loaded importer provider modules before switching providers."""
    for loaded_module_name in list(sys.modules):
        for module_prefix in _IMPORTER_PROVIDER_MODULE_PREFIXES:
            if loaded_module_name == module_prefix or loaded_module_name.startswith(f"{module_prefix}."):
                del sys.modules[loaded_module_name]
                break


def _enable_isaacsim_extension(extension_name: str) -> None:
    """Enable an Isaac Sim extension if it is not already enabled."""
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        if not manager.is_extension_enabled(extension_name):
            manager.set_extension_enabled_immediate(extension_name, True)
    except (AttributeError, ImportError, RuntimeError) as exc:
        raise ImportError(
            f"The Isaac Sim extension {extension_name!r} could not be enabled. Launch Isaac Sim before using the "
            "extension-backed importer."
        ) from exc


def _get_importer_classes(module, importer_kind: ImporterKind) -> tuple[type, type]:
    """Return importer classes from a loaded importer module."""
    importer_name, config_name = _IMPORTER_CLASSES[importer_kind]
    return getattr(module, importer_name), getattr(module, config_name)
