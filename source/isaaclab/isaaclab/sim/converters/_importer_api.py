# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Provider selection for URDF and MJCF importer APIs."""

from __future__ import annotations

import importlib
import subprocess
import sys
from importlib import metadata
from typing import Literal, NamedTuple

from isaaclab.utils.version import has_kit

ImporterKind = Literal["mjcf", "urdf"]


class ImporterProvider:
    """Load asset importer APIs from Isaac Sim or its standalone wheel.

    Both providers expose the same ``isaacsim.asset.importer.*`` modules. When Kit is
    running, its extension manager is used to enable the requested importer before the
    normal Python import. In kitless processes, the module resolves directly from the
    ``isaacsim-asset-isolated`` distribution.
    """

    #: pip distribution that provides the importer APIs without Isaac Sim.
    _STANDALONE_DIST = "isaacsim-asset-isolated"

    class _Api(NamedTuple):
        """Location of one importer API within the ``isaacsim.asset.importer.*`` namespace."""

        module: str
        importer_class: str
        config_class: str

        @property
        def import_statement(self) -> str:
            """Python statement importing the API, for the standalone runtime preflight."""
            return f"from {self.module} import {self.importer_class}, {self.config_class}"

    _apis: dict[ImporterKind, _Api] = {
        "mjcf": _Api("isaacsim.asset.importer.mjcf", "MJCFImporter", "MJCFImporterConfig"),
        "urdf": _Api("isaacsim.asset.importer.urdf", "URDFImporter", "URDFImporterConfig"),
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
        api = cls._apis[importer_kind]
        try:
            # Under a running Kit app, enable the owning extension; kitless, resolve from the wheel.
            if has_kit():
                # Imported here rather than at module scope: the converter scripts import this
                # module before deciding whether to launch Kit, and ``isaaclab.sim.utils`` binds
                # its Kit dependencies at import time based on ``has_kit()``. Importing it too
                # early would leave those bindings unset for the rest of the process.
                from isaaclab.sim.utils import enable_extension  # noqa: PLC0415

                enable_extension(api.module)
            module = importlib.import_module(api.module)
            return getattr(module, api.importer_class), getattr(module, api.config_class)
        except (AttributeError, ImportError, RuntimeError) as exc:
            raise ImportError(
                f"Failed to load {api.module}. Launch Isaac Sim before using its importer extension, or install "
                f"the standalone {cls._STANDALONE_DIST!r} distribution for kitless conversion."
            ) from exc

    @classmethod
    def is_standalone_available(cls) -> bool:
        """Return whether the standalone importer distribution is installed."""
        try:
            metadata.distribution(cls._STANDALONE_DIST)
        except metadata.PackageNotFoundError:
            return False
        return True

    @classmethod
    def validate_standalone_runtime(cls, importer_kind: ImporterKind, timeout: float = 60.0) -> None:
        """Verify in a subprocess that the standalone importer API can load.

        The check runs in a separate process because a broken environment (e.g. a conflicting
        OpenUSD build) can abort the interpreter via ``std::terminate`` rather than raising an
        exception, which an in-process import cannot observe.

        Args:
            importer_kind: Importer runtime to check.
            timeout: Maximum time to wait for the subprocess [s].

        Raises:
            ImportError: If the standalone importer API cannot load in a fresh process.
        """
        statement = cls._apis[importer_kind].import_statement
        try:
            result = subprocess.run([sys.executable, "-c", statement], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise ImportError(
                f"Modules of the {cls._STANDALONE_DIST!r} distribution did not load within {exc.timeout} seconds."
            ) from exc
        if result.returncode == 0:
            return

        details = (result.stderr or result.stdout).strip()
        raise ImportError(
            f"The {cls._STANDALONE_DIST!r} distribution is installed but its modules cannot load. Reinstall "
            f"{cls._STANDALONE_DIST!r} and its dependencies from the same release."
            f"\nRuntime check exited with code {result.returncode}:\n{details}"
        )
