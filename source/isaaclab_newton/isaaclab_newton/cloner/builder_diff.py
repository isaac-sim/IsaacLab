# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Structured comparison of Newton builders and finalized models.

Used by the batched-builder equivalence tests and the builder timing script to
verify that the legacy and batched replication paths produce the same Newton
model. Comparisons are exact by default; a tolerance can be supplied for
float fields where bitwise equality is not guaranteed (transform compositions
are performed in a different order by the two paths).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp
from newton import Model, ModelBuilder

# Float builder attributes affected by per-world transform composition.
_TRANSFORM_ATTRS = frozenset({"body_q", "shape_transform", "joint_X_p", "joint_q", "particle_q"})

# Builder attributes that hold configuration objects rather than model data.
_SKIPPED_BUILDER_ATTRS = frozenset({"default_bvh_cfg", "default_shape_cfg", "default_joint_cfg"})

_MAX_REPORTED_MISMATCHES = 5


def _as_array(value: Any) -> Any:
    """Convert warp/vector-like values to numpy arrays for comparison."""
    if hasattr(value, "__len__") and not isinstance(value, (str, list, tuple, dict, np.ndarray)):
        try:
            return np.array(value)
        except (TypeError, ValueError):
            return value
    return value


def _geometry_equal(a: Any, b: Any) -> bool:
    """Compare geometry sources (e.g. :class:`newton.Mesh`) structurally.

    Two independently parsed sources reference distinct but identical geometry
    objects, so identity comparison would report false mismatches.
    """
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    return np.array_equal(np.asarray(a.vertices), np.asarray(b.vertices)) and np.array_equal(
        np.asarray(a.indices), np.asarray(b.indices)
    )


def _compare_numeric(name: str, a: np.ndarray, b: np.ndarray, atol: float, errors: list[str]) -> None:
    bad = ~np.isclose(a, b, rtol=0.0, atol=atol, equal_nan=True)
    if not bad.any():
        return
    indices = np.argwhere(bad)[:_MAX_REPORTED_MISMATCHES]
    for idx in indices:
        key = tuple(idx.tolist())
        errors.append(f"{name}[{key}]: {a[key]!r} vs {b[key]!r} (diff {abs(a[key] - b[key])!r})")
    if bad.sum() > len(indices):
        errors.append(f"{name}: ... {int(bad.sum())} mismatches total")


def _compare_sequences(name: str, seq_a: Any, seq_b: Any, atol: float, errors: list[str]) -> None:
    if len(seq_a) != len(seq_b):
        errors.append(f"{name}: length {len(seq_a)} vs {len(seq_b)}")
        return
    # Fast path: numeric arrays.
    try:
        arr_a = np.asarray(seq_a, dtype=np.float64)
        arr_b = np.asarray(seq_b, dtype=np.float64)
        if arr_a.shape == arr_b.shape:
            _compare_numeric(name, arr_a, arr_b, atol, errors)
            return
    except (ValueError, TypeError):
        pass
    # Object path: element-wise comparison.
    reported = 0
    for i, (x, y) in enumerate(zip(seq_a, seq_b)):
        if hasattr(x, "vertices") and hasattr(x, "indices"):
            if not _geometry_equal(x, y):
                errors.append(f"{name}[{i}]: geometry differs ({x!r} vs {y!r})")
                reported += 1
                if reported >= _MAX_REPORTED_MISMATCHES:
                    errors.append(f"{name}: ... more mismatches follow")
                    return
            continue
        cx, cy = _as_array(x), _as_array(y)
        if isinstance(cx, np.ndarray) or isinstance(cy, np.ndarray):
            try:
                equal = np.allclose(
                    np.asarray(cx, dtype=np.float64), np.asarray(cy, dtype=np.float64), rtol=0.0, atol=atol
                )
            except (ValueError, TypeError):
                equal = np.array_equal(np.asarray(cx), np.asarray(cy))
        else:
            equal = cx == cy
        if not equal:
            errors.append(f"{name}[{i}]: {cx!r} vs {cy!r}")
            reported += 1
            if reported >= _MAX_REPORTED_MISMATCHES:
                errors.append(f"{name}: ... more mismatches follow")
                return


def _compare_custom_attributes(attrs_a: dict, attrs_b: dict, errors: list[str]) -> None:
    for key in sorted(set(attrs_a) | set(attrs_b)):
        a, b = attrs_a.get(key), attrs_b.get(key)
        if a is None or b is None:
            errors.append(f"custom_attributes[{key}]: present in one builder only (a={a is not None}, b={b is not None})")
            continue
        if isinstance(a.values, list) or isinstance(b.values, list):
            _compare_sequences(f"custom_attributes[{key}].values", a.values or [], b.values or [], 0.0, errors)
            continue
        values_a, values_b = a.values or {}, b.values or {}
        if set(values_a) != set(values_b):
            errors.append(f"custom_attributes[{key}].values keys differ: {sorted(set(values_a) ^ set(values_b))[:5]}")
            continue
        for k in values_a:
            cx, cy = _as_array(values_a[k]), _as_array(values_b[k])
            equal = np.array_equal(np.asarray(cx), np.asarray(cy)) if isinstance(cx, np.ndarray) else cx == cy
            if not equal:
                errors.append(f"custom_attributes[{key}].values[{k}]: {cx!r} vs {cy!r}")
                break


def compare_builder_states(a: ModelBuilder, b: ModelBuilder, *, transform_atol: float = 0.0) -> list[str]:
    """Compare the full replication-relevant state of two builders.

    Args:
        a: Reference builder (e.g. from the legacy path).
        b: Builder to compare (e.g. from the batched path).
        transform_atol: Absolute tolerance applied only to transform-composed float
            attributes (``body_q``, ``shape_transform``, ``joint_X_p``, ``joint_q``,
            ``particle_q``). All other fields are compared exactly.

    Returns:
        Human-readable mismatch descriptions; empty when the builders are equivalent.
    """
    errors: list[str] = []
    for key in sorted(set(vars(a)) | set(vars(b))):
        if key in _SKIPPED_BUILDER_ATTRS:
            continue
        va, vb = getattr(a, key, None), getattr(b, key, None)
        if isinstance(va, list) or isinstance(vb, list):
            atol = transform_atol if key in _TRANSFORM_ATTRS else 0.0
            _compare_sequences(key, va, vb, atol, errors)
        elif key == "custom_attributes":
            _compare_custom_attributes(va, vb, errors)
        elif key == "actuator_entries":
            if set(va) != set(vb):
                errors.append(f"{key}: entry keys differ: {set(va).symmetric_difference(set(vb))}")
                continue
            for entry_key in va:
                for field in ("indices", "pos_indices", "controller_args", "delay_args", "clamping_args"):
                    fa, fb = getattr(va[entry_key], field), getattr(vb[entry_key], field)
                    if fa != fb:
                        errors.append(f"{key}[{entry_key}].{field}: {str(fa)[:80]} vs {str(fb)[:80]}")
        elif isinstance(va, dict) or isinstance(vb, dict):
            if key == "custom_frequencies":
                if set(va) != set(vb):
                    errors.append(f"{key}: keys differ: {set(va).symmetric_difference(set(vb))}")
            elif va != vb:
                errors.append(f"{key}: {str(va)[:120]} vs {str(vb)[:120]}")
        elif isinstance(va, (int, float, str, bool, tuple, set, frozenset)) or va is None:
            if va != vb:
                errors.append(f"{key}: {va!r} vs {vb!r}")
    return errors


def compare_finalized_models(a: Model, b: Model, *, float_atol: float = 0.0) -> list[str]:
    """Compare two finalized Newton models field by field.

    Iterates every public attribute of the models: Warp arrays are compared by
    value (exactly for integer/boolean dtypes, with ``float_atol`` for floats),
    Python lists/dicts/scalars are compared exactly. Runtime handles (device
    pointers, BVHs, hash grids) are skipped.

    Args:
        a: Reference model.
        b: Model to compare.
        float_atol: Absolute tolerance for float arrays.

    Returns:
        Human-readable mismatch descriptions; empty when the models are equivalent.
    """
    errors: list[str] = []
    _compare_model_objects("model", a, b, float_atol, errors)
    return errors


def _compare_model_objects(prefix: str, a: Any, b: Any, float_atol: float, errors: list[str]) -> None:
    keys = sorted(set(vars(a)) | set(vars(b)))
    for key in keys:
        # bvh_* arrays are runtime acceleration structures whose buffers contain
        # uninitialized memory until first use (nondeterministic between identical builds).
        if key.startswith(("_", "bvh_")) or key.endswith("_ptr") or key in ("device", "particle_grid"):
            continue
        name = f"{prefix}.{key}"
        va, vb = getattr(a, key, None), getattr(b, key, None)
        if isinstance(va, wp.array) or isinstance(vb, wp.array):
            if va is None or vb is None:
                errors.append(f"{name}: present in one model only (a={va is not None}, b={vb is not None})")
                continue
            arr_a, arr_b = va.numpy(), vb.numpy()
            if arr_a.shape != arr_b.shape:
                errors.append(f"{name}: shape {arr_a.shape} vs {arr_b.shape}")
            elif arr_a.dtype.kind == "f":
                _compare_numeric(name, arr_a.astype(np.float64), arr_b.astype(np.float64), float_atol, errors)
            elif not np.array_equal(arr_a, arr_b):
                bad = np.argwhere(arr_a != arr_b)[:_MAX_REPORTED_MISMATCHES]
                for idx in bad:
                    k = tuple(idx.tolist())
                    errors.append(f"{name}[{k}]: {arr_a[k]!r} vs {arr_b[k]!r}")
        elif isinstance(va, (list, tuple)) or isinstance(vb, (list, tuple)):
            _compare_sequences(name, va or [], vb or [], float_atol, errors)
        elif isinstance(va, dict) or isinstance(vb, dict):
            if va != vb:
                errors.append(f"{name}: {str(va)[:120]} vs {str(vb)[:120]}")
        elif isinstance(va, (int, float, str, bool)) or va is None:
            if va != vb and not (va is None and vb is None):
                errors.append(f"{name}: {va!r} vs {vb!r}")
        elif type(va).__name__ == "AttributeNamespace" or type(vb).__name__ == "AttributeNamespace":
            _compare_model_objects(name, va, vb, float_atol, errors)
