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

_DEVICE_SPLIT_MARK_RE = re.compile(r"^\s*pytestmark\b.*\bdevice_split\b", re.MULTILINE)
"""Match a module-level ``pytestmark`` assignment that mentions ``device_split``.

Recognises both single-mark and single-line list forms:

* ``pytestmark = pytest.mark.device_split``
* ``pytestmark = [pytest.mark.device_split, pytest.mark.foo]``

Multi-line list forms are not supported (currently no test file uses one); if
a future test needs that, expand the parsing rule.
"""

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


def is_device_split_file(path: Path | str, source: str | None = None) -> bool:
    """Return whether the test file at ``path`` declares the ``device_split`` marker.

    Matches :data:`_DEVICE_SPLIT_MARK_RE` against ``source`` when supplied.
    Otherwise, reads the file source from ``path``. A missing or unreadable
    file returns ``False`` so the caller falls back to the default single-pass
    invocation.

    Args:
        path: Filesystem path to a candidate test file.
        source: Optional preloaded source text to inspect.

    Returns:
        ``True`` when the file's module-level ``pytestmark`` mentions
        ``device_split``; ``False`` otherwise.
    """
    if source is None:
        try:
            source = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
    return bool(_DEVICE_SPLIT_MARK_RE.search(source))
