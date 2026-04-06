# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Log file-backed asset paths from an environment configuration for training scripts."""

from __future__ import annotations

from typing import Any

BANNER_WIDTH = 78
BANNER = "=" * BANNER_WIDTH


def _format_variants(obj: Any) -> str:
    variants = getattr(obj, "variants", None)
    if variants is None:
        return ""
    if isinstance(variants, dict):
        d = variants
    elif hasattr(variants, "to_dict"):
        d = variants.to_dict()
    else:
        return ""
    if not d:
        return ""
    return f"variants={d!r}"


def _record_node_paths(role: str, obj: Any, sink: list[tuple[str, str, str, str]], seen: set[tuple[str, str, str, str]]) -> None:
    """Append (role, kind, path, notes) for any file path attributes on *obj*."""
    cls_name = type(obj).__name__
    usd_path = getattr(obj, "usd_path", None)
    if isinstance(usd_path, str) and usd_path.strip():
        notes = _format_variants(obj)
        if "TerrainImporter" in cls_name:
            notes = f"{notes}  [terrain USD]".strip()
        rec = (role, "USD", usd_path.strip(), notes)
        if rec not in seen:
            seen.add(rec)
            sink.append(rec)

    asset_path = getattr(obj, "asset_path", None)
    if isinstance(asset_path, str) and asset_path.strip():
        low = asset_path.lower()
        if low.endswith(".urdf"):
            kind = "URDF"
        elif low.endswith((".xml", ".mjcf")):
            kind = "MJCF"
        else:
            kind = "ASSET"
        rec = (role, kind, asset_path.strip(), "")
        if rec not in seen:
            seen.add(rec)
            sink.append(rec)


def _walk_env_cfg_for_paths(
    obj: Any,
    role: str,
    *,
    visited: set[int],
    sink: list[tuple[str, str, str, str]],
    seen_records: set[tuple[str, str, str, str]],
    depth: int,
) -> None:
    if depth > 28 or obj is None:
        return
    if isinstance(obj, (str, int, float, bool, bytes)):
        return

    oid = id(obj)
    is_container = isinstance(obj, (dict, list, tuple)) or hasattr(obj, "__dataclass_fields__")
    if is_container:
        if oid in visited:
            return
        visited.add(oid)

    if hasattr(obj, "__dataclass_fields__"):
        _record_node_paths(role, obj, sink, seen_records)

    spawn = getattr(obj, "spawn", None)
    if spawn is not None and spawn is not obj:
        _walk_env_cfg_for_paths(
            spawn, f"{role}.spawn", visited=visited, sink=sink, seen_records=seen_records, depth=depth + 1
        )

    if isinstance(obj, dict):
        for key, val in obj.items():
            _walk_env_cfg_for_paths(
                val, f"{role}.{key}", visited=visited, sink=sink, seen_records=seen_records, depth=depth + 1
            )
        return
    if isinstance(obj, (list, tuple)):
        for i, val in enumerate(obj):
            _walk_env_cfg_for_paths(
                val, f"{role}[{i}]", visited=visited, sink=sink, seen_records=seen_records, depth=depth + 1
            )
        return
    if not hasattr(obj, "__dataclass_fields__"):
        return

    for fname in obj.__dataclass_fields__:
        if fname.startswith("_") or fname == "spawn":
            continue
        try:
            val = getattr(obj, fname)
        except Exception:
            continue
        _walk_env_cfg_for_paths(
            val, f"{role}.{fname}", visited=visited, sink=sink, seen_records=seen_records, depth=depth + 1
        )


def collect_training_asset_path_records(env_cfg: Any) -> list[tuple[str, str, str, str]]:
    """Walk *env_cfg* and return deduplicated (role, kind, path, notes) entries for file-backed assets."""
    sink: list[tuple[str, str, str, str]] = []
    seen_records: set[tuple[str, str, str, str]] = set()
    visited: set[int] = set()
    _walk_env_cfg_for_paths(env_cfg, "env_cfg", visited=visited, sink=sink, seen_records=seen_records, depth=0)
    return sink


def log_training_asset_paths(task_id: str, env_cfg: Any, phase: str) -> None:
    """Print a clearly delimited summary of resolved asset file paths (for training logs / benchmarks)."""
    records = collect_training_asset_path_records(env_cfg)
    lines = [
        "",
        "",
        BANNER,
        f"ISAAC LAB TRAINING — resolved scene asset file paths  ({phase})",
        f"Task: {task_id}",
        BANNER,
        "",
    ]
    if not records:
        lines.append("  (no file-backed USD / URDF / MJCF paths found in env configuration tree)")
    else:
        for role, kind, path, notes in sorted(records, key=lambda r: (r[0], r[1], r[2])):
            line = f"  [{kind:5}]  {path}"
            if notes:
                line += f"    {notes}"
            line += f"\n           role: {role}"
            lines.append(line)
    lines.extend(
        [
            "",
            BANNER,
            "",
            "",
        ]
    )
    print("\n".join(lines))
