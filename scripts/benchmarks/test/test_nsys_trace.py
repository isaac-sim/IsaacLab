# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sync tests for ``scripts/benchmarks/nsys_trace.json``.

The JSON lists Python functions that nsys annotates during profiling via the
``--python-functions-trace`` flag. Entries reference IsaacLab (and adjacent)
source by dotted paths. These tests catch two kinds of drift:

* Referenced functions that no longer resolve (hard failure)
* Covered classes that have public methods not listed in the JSON (warning —
  signals that new coverage may need to be added)
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import importlib
import inspect
import json
import warnings
from collections import defaultdict
from pathlib import Path

import pytest

TRACE_JSON_PATH = Path(__file__).resolve().parents[1] / "nsys_trace.json"


def _load_trace_entries() -> list[dict]:
    """Return the parsed JSON entries from the trace file."""
    if not TRACE_JSON_PATH.exists():
        raise RuntimeError(f"nsys trace JSON not found at {TRACE_JSON_PATH}")
    try:
        with TRACE_JSON_PATH.open() as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"nsys trace JSON at {TRACE_JSON_PATH} is malformed: {exc}") from exc


def _function_name_and_module(entry_module: str, func_spec) -> tuple[str, str]:
    """Normalize a function spec to ``(module, dotted_name)``.

    ``func_spec`` may be a bare string or a dict that optionally overrides
    ``module`` (per the nsys --python-functions-trace schema).
    """
    if isinstance(func_spec, str):
        return entry_module, func_spec
    return func_spec.get("module", entry_module), func_spec["function"]


def _iter_function_pairs(entries: list[dict]) -> list[tuple[str, str]]:
    """Yield ``(module, dotted_function_path)`` for every function in the JSON."""
    pairs: list[tuple[str, str]] = []
    for entry in entries:
        entry_module = entry["module"]
        for func_spec in entry["functions"]:
            pairs.append(_function_name_and_module(entry_module, func_spec))
    return pairs


def _resolve(module_name: str, dotted_path: str):
    """Import ``module_name`` and walk ``dotted_path`` via getattr."""
    obj = importlib.import_module(module_name)
    for attr in dotted_path.split("."):
        obj = getattr(obj, attr)
    return obj


def _group_methods_by_class(pairs: list[tuple[str, str]]) -> dict[tuple[str, str], set[str]]:
    """Group referenced method names by ``(module, class_name)``.

    Top-level functions (paths without a dot) are skipped — no class context.
    """
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for module_name, dotted_path in pairs:
        parts = dotted_path.split(".")
        if len(parts) >= 2:
            class_name, method_name = parts[0], parts[-1]
            grouped[(module_name, class_name)].add(method_name)
    return grouped


def _is_own_public_method(cls: type, name: str, member: object) -> bool:
    """True if ``member`` is a public method defined directly on ``cls``."""
    if not inspect.isfunction(member):
        return False
    if name not in cls.__dict__:
        return False
    # Skip dunders and private helpers from the unreferenced-method check; the JSON curates inclusions explicitly.
    if name.startswith("_"):
        return False
    return True


_FUNCTION_PAIRS = _iter_function_pairs(_load_trace_entries())


@pytest.mark.parametrize(
    "module_name, dotted_path",
    _FUNCTION_PAIRS,
    ids=[f"{m}:{p}" for m, p in _FUNCTION_PAIRS],
)
def test_function_resolves(module_name: str, dotted_path: str):
    """Every function referenced in the trace JSON must resolve to a callable.

    A missing reference silently loses profiling coverage, so this fails loudly.
    Modules that aren't importable in the current environment (e.g. optional
    RL frameworks) are skipped rather than failed.
    """
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        pytest.skip(f"Module '{module_name}' not importable here: {exc}")

    try:
        resolved = _resolve(module_name, dotted_path)
    except AttributeError as exc:
        pytest.fail(f"'{module_name}.{dotted_path}' not found: {exc}")

    assert callable(resolved), f"'{module_name}.{dotted_path}' resolved but is not callable"


def test_warn_unreferenced_methods_on_covered_classes():
    """Emit a warning for each public method that isn't listed in the JSON.

    Scope: classes that already have at least one method referenced. If the
    class gained a new public method since the JSON was last updated, it shows
    up here as a nudge to add (or intentionally omit) it. Inherited, dunder,
    and private methods are excluded to keep the signal actionable.
    """
    grouped = _group_methods_by_class(_FUNCTION_PAIRS)

    for (module_name, class_name), referenced in sorted(grouped.items()):
        try:
            cls = _resolve(module_name, class_name)
        except (ImportError, AttributeError):
            continue
        if not inspect.isclass(cls):
            continue

        own_public = {name for name, member in inspect.getmembers(cls) if _is_own_public_method(cls, name, member)}
        unreferenced = own_public - referenced
        if unreferenced:
            warnings.warn(
                f"{module_name}.{class_name} has public methods not listed in nsys_trace.json: {sorted(unreferenced)}",
                stacklevel=2,
            )
