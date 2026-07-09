# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Provider selection for URDF and MJCF importer APIs."""

from __future__ import annotations

import importlib
from importlib import metadata
import subprocess
import sys
from typing import Literal

from isaaclab.utils.version import has_kit

ImporterKind = Literal["mjcf", "urdf"]


class ImporterProvider:
    """Load asset importer APIs from Isaac Sim or its standalone wheel.

    Both providers expose the same ``isaacsim.asset.importer.*`` modules. When Kit is
    running, its extension manager is used to enable the requested importer before the
    normal Python import. In kitless processes, the module resolves directly from the
    ``isaacsim-asset-isolated`` distribution.
    """

    standalone_distribution = "isaacsim-asset-isolated"
    """Distribution that provides the importer APIs without Isaac Sim."""

    _api_modules = {
        "mjcf": "isaacsim.asset.importer.mjcf",
        "urdf": "isaacsim.asset.importer.urdf",
    }
    _api_classes = {
        "mjcf": ("MJCFImporter", "MJCFImporterConfig"),
        "urdf": ("URDFImporter", "URDFImporterConfig"),
    }

    @classmethod
    def load_api(cls, importer_kind: ImporterKind) -> tuple[type, type]:
        """Load an importer and its configuration class.

        Args:
            importer_kind: Importer API to load.

        Returns:
            The importer class and importer configuration class.

        Raises:
            ImportError: If the requested API cannot be loaded from either provider.
        """
        module_name = cls._api_modules[importer_kind]
        try:
            cls._enable_kit_extension(module_name)
            module = importlib.import_module(module_name)
            importer_name, config_name = cls._api_classes[importer_kind]
            return getattr(module, importer_name), getattr(module, config_name)
        except (AttributeError, ImportError, RuntimeError) as exc:
            raise ImportError(
                f"Failed to load {module_name}. Launch Isaac Sim before using its importer extension, or install "
                f"the standalone {cls.standalone_distribution!r} distribution for kitless conversion."
            ) from exc

    @classmethod
    def is_standalone_available(cls) -> bool:
        """Return whether the standalone importer distribution is installed."""
        try:
            metadata.distribution(cls.standalone_distribution)
        except metadata.PackageNotFoundError:
            return False
        return True

    @classmethod
    def validate_standalone_runtime(cls, importer_kind: ImporterKind) -> None:
        """Verify in a subprocess that the standalone importer API can load.

        A broken environment (e.g. ``usd-core`` co-installed with ``usd-exchange``) aborts
        the interpreter via ``std::terminate`` rather than raising, so the check must run
        in a separate process to be observable.

        Args:
            importer_kind: Importer runtime to check.

        Raises:
            ImportError: If the standalone importer API cannot load in a fresh process.
        """
        module_name = cls._api_modules[importer_kind]
        importer_name, config_name = cls._api_classes[importer_kind]
        import_statement = f"from {module_name} import {importer_name}, {config_name}"
        try:
            result = subprocess.run(
                [sys.executable, "-c", import_statement],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            raise ImportError(
                f"The standalone {importer_kind.upper()} importer did not load within {exc.timeout} seconds."
            ) from exc
        if result.returncode == 0:
            return

        details = (result.stderr or result.stdout).strip()
        raise ImportError(
            f"The standalone {importer_kind.upper()} importer is installed but cannot load. Reinstall "
            f"{cls.standalone_distribution!r} and its OpenUSD dependencies from the same release."
            f"\nRuntime check exited with code {result.returncode}:\n{details}"
        )

    @staticmethod
    def _enable_kit_extension(extension_name: str) -> None:
        """Enable an importer extension when called from a running Kit process."""
        if not has_kit():
            return
        # has_kit() guarantees "omni.kit.app" is loaded with a non-None app
        manager = sys.modules["omni.kit.app"].get_app().get_extension_manager()
        if not manager.is_extension_enabled(extension_name):
            manager.set_extension_enabled_immediate(extension_name, True)
