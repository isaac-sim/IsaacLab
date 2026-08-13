# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for detecting and driving the ``device_split`` pytest marker.

Test files that declare ``pytestmark = pytest.mark.device_split`` at module
scope must be re-invoked once per device (CPU and GPU) in separate processes
to work around process-global device locks such as ``ovphysx<=0.3.7`` gap G5.
The :func:`is_device_split_file` predicate lets the per-file CI runner in
``tools/conftest.py`` detect this without importing the test module.
"""

from __future__ import annotations

import re
from pathlib import Path

# Per-pass pytest ``-k`` selectors used by ``tools/conftest.py`` when a file
# declares the ``device_split`` marker. Each entry is ``(suffix, k_expr)``:
#   - ``suffix`` is appended to the JUnit report filename to keep both passes' XML.
#   - ``k_expr`` is the ``-k`` keyword expression. ``"cpu or not cuda"`` keeps
#     non-parametrized tests in the CPU pass; ``"cuda"`` catches GPU-parametrized
#     tests only.
DEVICE_SPLIT_PASSES: list[tuple[str, str]] = [
    ("-cpu", "cpu or not cuda"),
    ("-cuda", "cuda"),
]


def has_pytestmark(path: Path | str, marker: str, source: str | None = None) -> bool:
    """Return whether a test file's module-level ``pytestmark`` names ``marker``.

    Single markers and single-line marker lists are supported. A missing or
    unreadable file returns ``False`` so callers retain their default behavior.

    Args:
        path: Filesystem path to a candidate test file.
        source: Optional preloaded source text to inspect.

    Returns:
        ``True`` when the file's module-level ``pytestmark`` mentions the marker.
    """
    if source is None:
        try:
            source = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
    return bool(re.search(rf"^\s*pytestmark\b.*\b{re.escape(marker)}\b", source, re.MULTILINE))


def is_device_split_file(path: Path | str, source: str | None = None) -> bool:
    """Return whether the test file declares the ``device_split`` marker."""
    return has_pytestmark(path, "device_split", source)
