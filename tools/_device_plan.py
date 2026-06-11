# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure planner for the per-file CI test runner.

Turns a runner's claimed files plus its runtime device mask into a list of
``(file, mask)`` work units, where ``mask`` is the ``ISAACLAB_TEST_DEVICES``
value the executor will set for that subprocess. A file is one unit unless it
declares the ``device_isolated`` marker and the runtime spans more than one
device, in which case it splits into one single-device unit per device (to work
around backends whose device mode is process-global, e.g. ovphysx).

This module is pure: no I/O beyond reading a file's source for marker detection,
no subprocess, no pytest collection. It replaces the ``DEVICE_SPLIT_PASSES`` /
``is_device_split_file`` pieces of the former ``_device_split`` module.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

_ISOLATED_MARK_RE = re.compile(r"^\s*pytestmark\b.*\bdevice_isolated\b", re.MULTILINE)
"""Match a module-level ``pytestmark`` assignment that mentions ``device_isolated``.

Recognises both the single-mark and single-line list forms:

* ``pytestmark = pytest.mark.device_isolated``
* ``pytestmark = [pytest.mark.device_isolated, pytest.mark.slow]``
"""


def is_isolated(path: Path | str, source: str | None = None) -> bool:
    """Return whether a test file declares the ``device_isolated`` marker.

    Detection is by source regex, not import, so the planner stays collection-free.

    Args:
        path: Filesystem path to the candidate test file.
        source: Optional preloaded source text to inspect instead of reading
            ``path``.

    Returns:
        ``True`` when the file's module-level ``pytestmark`` mentions
        ``device_isolated``; ``False`` otherwise (including a missing or
        unreadable file).
    """
    if source is None:
        try:
            source = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
    return bool(_ISOLATED_MARK_RE.search(source))


def _single_bit_mask(index: int, width: int) -> str:
    """Return a width-``width`` device mask with only ``index`` set.

    Args:
        index: The single device position to enable.
        width: Total mask width.

    Returns:
        A mask string such as ``"010"`` (``index=1``, ``width=3``).
    """
    return "".join("1" if pos == index else "0" for pos in range(width))


def plan_units(
    files: list[str],
    runtime_mask: str,
    is_isolated: Callable[[Path | str], bool] = is_isolated,
) -> list[tuple[str, str]]:
    """Plan the ``(file, mask)`` work units for one runner.

    Args:
        files: Test files this runner is responsible for, in run order.
        runtime_mask: The runner's concrete device mask (e.g. ``"110"`` on the
            single-GPU lane, ``"0001"`` for an mgpu shard). Must not contain the
            open-ended ``"X"`` form, which is a scope-only construct.
        is_isolated: Predicate deciding whether a file needs one process per
            device. Defaults to :func:`is_isolated` (marker detection by source).

    Returns:
        Work units in run order. A non-isolated file, or any file on a
        single-device runtime, yields one unit at ``runtime_mask``. An isolated
        file on a multi-device runtime yields one single-device unit per set bit.

    Raises:
        ValueError: When ``runtime_mask`` contains ``"X"``.
    """
    if "X" in runtime_mask:
        raise ValueError(f"runtime mask {runtime_mask!r} must be concrete (no 'X'); 'X' is a scope-only construct")
    set_bits = [pos for pos, char in enumerate(runtime_mask) if char == "1"]
    units: list[tuple[str, str]] = []
    for test_file in files:
        if is_isolated(test_file) and len(set_bits) > 1:
            units.extend((test_file, _single_bit_mask(pos, len(runtime_mask))) for pos in set_bits)
        else:
            units.append((test_file, runtime_mask))
    return units
