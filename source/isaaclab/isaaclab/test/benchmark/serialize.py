# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Serialisation helpers for benchmark bundles (stdlib only, Isaac-free)."""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any


def _to_plain(obj: Any) -> Any:
    """Recursively convert dataclass instances to plain dicts/lists."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, list):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def write_bundle_file(bundle: Any, path: str) -> None:
    """Write a bundle dataclass to disk as schema-v1 JSON, atomically.

    Serialises to a sibling ``<path>.tmp`` then ``os.replace``\\ s it into place,
    so an interrupted write never leaves a partially-written ``path``. Uses
    ``indent=2`` for readability; payloads are small (~10 KB training.json).

    Args:
        bundle: A frozen dataclass tree of primitives, lists, and dicts —
            typically :class:`~isaaclab.test.benchmark.schema.TrainingBundle`,
            :class:`~isaaclab.test.benchmark.schema.RuntimeBundle`, or
            :class:`~isaaclab.test.benchmark.schema.StartupBundle`.
        path: Output file path.
    """
    # os.path.abspath always yields a non-empty dirname, so no fallback needed.
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(_to_plain(bundle), f, indent=2, sort_keys=False)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
