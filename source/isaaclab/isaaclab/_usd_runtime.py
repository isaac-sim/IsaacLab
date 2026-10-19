# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Select Kit's OpenUSD provider before importing any native USD bindings."""

import argparse
import ctypes
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path


class _KitUsdFinder(importlib.abc.MetaPathFinder):
    """Preload only native libraries needed by bindings from the selected Kit provider.

    Kit's Python bindings do not search the extension's ``bin`` directory themselves. Each
    library's dependencies do, so loading its matching library by absolute path is sufficient.
    Imports outside this provider continue through the normal Python machinery unchanged.
    """

    def __init__(self, provider: Path):
        self.provider = provider
        self.dll_directory = os.add_dll_directory(str(provider / "bin")) if sys.platform == "win32" else None

    def find_spec(self, fullname, path=None, target=None):
        parts = fullname.split(".")
        if (
            sys.platform.startswith("linux")
            and len(parts) == 3
            and parts[0] == "pxr"
            and parts[2].startswith("_")
            and path
            and any(Path(entry).resolve().is_relative_to(self.provider / "pxr") for entry in path)
        ):
            library = self.provider / "bin" / f"libusd{parts[2]}.so"
            if library.is_file():
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        return None


def _kit_requested(argv: list[str]) -> bool:
    """Recognize explicit Kit viewers without consuming the application's arguments or help."""
    if "-h" in argv or "--help" in argv:
        return False
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    parser.add_argument("--visualizer", "--viz")
    try:
        args, _ = parser.parse_known_args(argv)
    except argparse.ArgumentError:
        return False  # The application's parser owns malformed-argument diagnostics.
    return args.visualizer is not None and "kit" in args.visualizer.split(",")


def _prepare_kit_usd() -> None:
    """Use the selected Isaac Sim build's Python and native USD provider in this process.

    This runs before prelaunch Newton imports for explicit Kit CLI selections and at the launch
    boundary for config-selected Kit. Native libraries cannot safely be unloaded after another
    OpenUSD build has been imported, so that case is rejected before Kit can crash the process.
    Source builds keep their configured runtime when USD is not in a bundled extension layout.
    """
    configured_root = os.environ.get("ISAAC_PATH")
    if configured_root:
        root = Path(configured_root)
    else:
        spec = importlib.util.find_spec("isaacsim")
        if spec is None or spec.origin is None:
            return  # Let the runtime raise its original missing-dependency error when it starts.
        root = Path(spec.origin).parent
        if root.parent.name == "python_packages":
            root = root.parent.parent  # Downloaded and source-build Isaac Sim Python packages.
    providers = [
        directory.resolve()
        for extensions in (root / "exts", root / "extscache", root / "kit" / "exts")
        for pattern in ("omni.usd.libs*", "omni.usd_libs*")
        for directory in extensions.glob(pattern)
        if (directory / "pxr").is_dir() and (directory / "bin").is_dir()
    ]
    if not providers:
        return
    providers = list(dict.fromkeys(providers))
    if len(providers) != 1:
        raise RuntimeError(f"Cannot select one Kit OpenUSD provider in {root}: {providers}")
    provider = providers[0]
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("pxr."):
            continue
        origin = getattr(module, "__file__", None)
        if (
            origin is not None
            and any(origin.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)
            and not Path(origin).resolve().is_relative_to(provider)
        ):
            raise RuntimeError(
                f"OpenUSD was already loaded from {origin}, but Kit requires {provider}. "
                "Start a fresh process with --visualizer kit before importing Newton or pxr, "
                "or call launch_simulation with the Kit configuration before those imports."
            )
    if any(isinstance(finder, _KitUsdFinder) and finder.provider == provider for finder in sys.meta_path):
        return
    sys.path.insert(0, str(provider))
    sys.meta_path.insert(0, _KitUsdFinder(provider))
