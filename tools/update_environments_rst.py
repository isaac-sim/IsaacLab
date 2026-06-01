# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Update the comprehensive environment list in ``environments.rst``.

Maintainer tooling: reads the Gym registry (RL libraries, presets, workflows,
and inference task names) and rewrites the auto-generated section in
``docs/source/overview/environments.rst``.

Example usage::

    python tools/update_environments_rst.py
    python tools/update_environments_rst.py --check
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path


def _bootstrap_paths() -> None:
    """Prepend editable ``source/*`` packages and ``tools/`` for dev-tree runs."""
    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "source"
    tools_dir = repo_root / "tools"

    prepend: list[str] = [str(tools_dir)]
    if source_dir.is_dir():
        for package_dir in sorted(source_dir.iterdir()):
            if not package_dir.is_dir():
                continue
            module_root = package_dir / package_dir.name
            if module_root.is_dir():
                prepend.append(str(package_dir))

    for path in reversed(prepend):
        if path not in sys.path:
            sys.path.insert(0, path)


_bootstrap_paths()

from environ_docs import (  # noqa: E402
    collect_environment_doc_rows,
    patch_environments_rst,
    render_comprehensive_list_table,
)

import isaaclab_tasks  # noqa: E402, F401

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: E402, F401


def _default_rst_path() -> Path:
    return Path(__file__).resolve().parents[1] / "docs" / "source" / "overview" / "environments.rst"


def main() -> int:
    """Generate and optionally write the comprehensive environment list."""
    parser = argparse.ArgumentParser(description="Update environments.rst from the Gym registry.")
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_rst_path(),
        help="Path to environments.rst (default: docs/source/overview/environments.rst).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with code 1 when environments.rst is out of date instead of rewriting it.",
    )
    args = parser.parse_args()

    rows = collect_environment_doc_rows()
    generated_table = render_comprehensive_list_table(rows)

    output_path = args.output.resolve()
    original = output_path.read_text(encoding="utf-8")
    updated = patch_environments_rst(original, generated_table)

    if args.check:
        if updated != original:
            print(
                f"[ERROR] {output_path} is out of date. Run python tools/update_environments_rst.py to refresh it.",
                file=sys.stderr,
            )
            return 1
        print(f"[INFO] {output_path} is up to date ({len(rows)} training environments).")
        return 0

    if updated == original:
        print(f"[INFO] {output_path} already up to date ({len(rows)} training environments).")
        return 0

    output_path.write_text(updated, encoding="utf-8")
    print(f"[INFO] Updated {output_path} with {len(rows)} training environments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
