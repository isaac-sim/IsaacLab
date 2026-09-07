# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for loading optional OvPhysX runtime modules."""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from pathlib import Path
from types import ModuleType

_OVPHYSX_INSTALL_MESSAGE = (
    "The OvPhysX backend requires the optional 'ovphysx' runtime wheel, which is not installed. "
    "Run your command with: uv run --extra ovphysx <command> "
    "(or, manually: python -m pip install --extra-index-url https://pypi.nvidia.com ovphysx)."
)
_OVPHYSX_OMNICLIENT_VERSION_PATTERN = re.compile(r"^(\d+\.\d+\.\d+)(?:-|\+|$)")


def _required_omniverseclient_version(package_root: Path) -> str | None:
    """Read the OmniClient release required by an installed OvPhysX package.

    Args:
        package_root: Root directory containing the ``ovphysx`` package.

    Returns:
        Required ``omniverseclient`` distribution version, or ``None`` when the package does not
        provide a compatibility marker.

    Raises:
        RuntimeError: If the compatibility marker exists but has an unsupported format.
    """
    marker_path = package_root / "plugins" / "ovstage-omniclient.version"
    if not marker_path.is_file():
        return None

    marker = marker_path.read_text(encoding="utf-8").strip()
    match = _OVPHYSX_OMNICLIENT_VERSION_PATTERN.match(marker)
    if match is None:
        raise RuntimeError(f"Invalid OvPhysX OmniClient compatibility marker '{marker}' in {marker_path}.")
    return match.group(1)


def _installed_ovphysx_omniverseclient_version() -> str | None:
    """Return the OmniClient version required by the installed ``ovphysx`` distribution."""
    try:
        distribution = importlib.metadata.distribution("ovphysx")
    except importlib.metadata.PackageNotFoundError:
        return None
    package_root = Path(distribution.locate_file("ovphysx"))
    return _required_omniverseclient_version(package_root)


def _validate_ovphysx_omniverseclient() -> None:
    """Reject an installed OmniClient release that is incompatible with OvPhysX's OVStage plugin."""
    required = _installed_ovphysx_omniverseclient_version()
    if required is None:
        return
    try:
        installed = importlib.metadata.version("omniverseclient")
    except importlib.metadata.PackageNotFoundError:
        installed = "not installed"
    if installed != required:
        raise RuntimeError(
            f"The installed OvPhysX runtime requires omniverseclient=={required}, but {installed} is installed. "
            f"Install the matched runtime with: python -m pip install omniverseclient=={required}."
        )


def import_ovphysx(module_name: str = "ovphysx") -> ModuleType:
    """Import an optional ``ovphysx`` runtime module with an actionable install error.

    Args:
        module_name: Name of the ``ovphysx`` module to import.

    Returns:
        The imported runtime module.

    Raises:
        ModuleNotFoundError: If the optional ``ovphysx`` runtime wheel is not installed.
    """
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != "ovphysx":
            raise
        raise ModuleNotFoundError(_OVPHYSX_INSTALL_MESSAGE, name="ovphysx") from exc
    _validate_ovphysx_omniverseclient()
    return module
