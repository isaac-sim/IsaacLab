# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the unified benchmark entry scripts."""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType


def get_backend_type(cli_backend: str) -> str:
    """Map CLI backend names to canonical backend type strings.

    Args:
        cli_backend: The backend name from CLI arguments (legacy long-form or
            short canonical form).

    Returns:
        Canonical backend type string; defaults to ``"omniperf"`` for unknown values.
    """
    mapping = {
        "OmniPerfKPIFile": "omniperf",
        "JSONFileMetrics": "json",
        "OsmoKPIFile": "osmo",
        "LocalLogMetrics": "json",
        "omniperf": "omniperf",
        "json": "json",
        "osmo": "osmo",
        "summary": "summary",
        "schema": "schema",
    }
    return mapping.get(cli_backend, "omniperf")


def get_backend_types(cli_backend: str) -> list[str]:
    """Split a comma-separated ``--benchmark_backend`` value into canonical backend types.

    Each token is normalized with :func:`get_backend_type` (so legacy long-form aliases and
    unknown-token fallback to ``"omniperf"`` still apply). Order is preserved and duplicates
    are removed. An empty input yields ``["omniperf"]``.

    Args:
        cli_backend: Raw ``--benchmark_backend`` value, e.g. ``"schema"`` or ``"schema,omniperf"``.

    Returns:
        Ordered, de-duplicated list of canonical backend type strings.
    """
    out: list[str] = []
    for tok in cli_backend.split(","):
        tok = tok.strip()
        if not tok:
            continue
        canon = get_backend_type(tok)
        if canon not in out:
            out.append(canon)
    return out or ["omniperf"]


def preset_tokens(remaining: list[str]) -> list[str]:
    """Extract active preset tokens from the raw Hydra remainder.

    Collects the comma-split values of every ``physics=``, ``renderer=``, and ``presets=``
    token in *remaining* (the verbatim remainder returned by
    :func:`~isaaclab_tasks.utils.setup_preset_cli`), preserving order and dropping duplicates.
    Returns an empty list when none are present.

    Args:
        remaining: Hydra argument remainder from :func:`~isaaclab_tasks.utils.setup_preset_cli`.

    Returns:
        List of active preset token strings.
    """
    out: list[str] = []
    for arg in remaining:
        for key in ("physics=", "renderer=", "presets="):
            if arg.startswith(key):
                for tok in arg.split("=", 1)[1].split(","):
                    if tok and tok not in out:
                        out.append(tok)
    return out


def import_module_from_path(module_name: str, module_path: str | os.PathLike[str]) -> ModuleType:
    """Import a module from an explicit file path without relying on package resolution.

    Loads the module by absolute path via ``importlib.util``, avoiding the need for the
    target file's directory to be an importable package (e.g. a script directory with no
    ``__init__.py``).  The loaded module is cached in ``sys.modules`` under *module_name*
    so repeated calls are free.

    Args:
        module_name: Unique module name to register in ``sys.modules``.
        module_path: Path to the Python file to import (``str`` or
            :class:`~pathlib.Path`).

    Returns:
        The imported module.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name!r} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
