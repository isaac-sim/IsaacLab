# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Apply dotted-path overrides onto config objects (typically FrankaCableEnvCfg).

The driver passes overrides as a flat ``{dotted_path: value}`` dict so it can be
JSON-serialized alongside the rest of the trial metadata. This module walks the
path with :func:`getattr` and sets the final attribute with :func:`setattr`.
"""

from __future__ import annotations

from typing import Any


def apply_overrides(root: Any, overrides: dict[str, Any]) -> None:
    """Apply each ``"a.b.c": value`` entry to ``root`` in-place.

    Raises ``AttributeError`` if any intermediate attribute is missing.
    """
    for path, value in overrides.items():
        parts = path.split(".")
        node = root
        for part in parts[:-1]:
            if not hasattr(node, part):
                raise AttributeError(f"Override path '{path}' missing at '{part}'")
            node = getattr(node, part)
        leaf = parts[-1]
        if not hasattr(node, leaf):
            raise AttributeError(f"Override path '{path}' missing at '{leaf}'")
        setattr(node, leaf, value)


def parse_override_value(text: str) -> Any:
    """Parse a CLI override value string into int/float/bool/str.

    Order: bool ("true"/"false") -> int -> float -> raw string.
    """
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text
