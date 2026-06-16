# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static pre-flight validation for ``tasks.json``.

Catches the class of bug where a ``task_id`` in ``tasks.json`` no longer matches
a registered Gymnasium environment (e.g. a renamed/version-bumped task), which
otherwise only surfaces after a multi-minute GPU benchmark job fails at task
lookup. Runs without Isaac Sim: it loads the task matrix via the normal loader
(schema check) and statically scans the repository's ``gym.register(id=...)``
calls (registry check), so it is safe on a plain CPU runner.

Usage::

    python3 tools/perf_regression_gate/validate_tasks.py
"""

import re
import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
_TOOLS_DIR = _MODULE_DIR.parent
_REPO_ROOT = _TOOLS_DIR.parent

if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from task_config import load_tasks  # noqa: E402

_ID_RE = re.compile(r"""id\s*=\s*["']([^"']+)["']""")


def registered_task_ids(source_root: Path) -> set[str]:
    """Collect Gymnasium environment ids registered under ``source_root``.

    Scans ``__init__.py`` files that call ``gym.register(...)`` and extracts the
    ``id="..."`` arguments. This is a static text scan (no imports), so it works
    without Isaac Sim installed.
    """
    ids: set[str] = set()
    for init_file in source_root.rglob("__init__.py"):
        try:
            text = init_file.read_text(errors="replace")
        except OSError:
            continue
        if "register(" not in text:
            continue
        for block in text.split("register(")[1:]:
            match = _ID_RE.search(block[:200])
            if match:
                ids.add(match.group(1))
    return ids


def validate(source_root: Path | None = None) -> list[str]:
    """Return a list of human-readable problems; empty list means valid."""
    source_root = source_root or (_REPO_ROOT / "source")
    problems: list[str] = []

    try:
        tasks = load_tasks()
    except Exception as exc:  # noqa: BLE001 - surface any schema/parse error verbatim
        return [f"tasks.json failed to load: {exc}"]

    registered = registered_task_ids(source_root)
    if not registered:
        # Don't hard-fail when the source tree isn't present (e.g. partial checkout);
        # the schema check above still ran.
        problems.append(f"warning: no gym.register ids found under {source_root}; skipping registry check")
        return problems

    seen_task_ids = {task.task_id for task in tasks}
    for task_id in sorted(seen_task_ids):
        if task_id not in registered:
            problems.append(f"task_id {task_id!r} is not a registered Gymnasium environment")
    return problems


def main() -> int:
    problems = validate()
    hard_errors = [p for p in problems if not p.startswith("warning:")]
    for problem in problems:
        prefix = "WARN " if problem.startswith("warning:") else "ERROR"
        print(f"[validate_tasks] {prefix}: {problem}")
    if hard_errors:
        print(f"[validate_tasks] FAILED with {len(hard_errors)} problem(s).")
        return 1
    print("[validate_tasks] OK: all task_ids resolve to registered environments.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
