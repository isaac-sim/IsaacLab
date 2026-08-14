# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Update environment tables in ``environments.rst``.

Maintainer tooling: reads the Gym registry (RL libraries, presets, and
workflows), synchronizes the curated tables, and rewrites the comprehensive
auto-generated section in ``docs/source/overview/environments.rst`` and the
task rows in ``docs/source/_static/css/environment-browser.js``.

Example usage::

    uv run python tools/update_environments_rst.py
    uv run python tools/update_environments_rst.py --check
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
    patch_curated_environment_tables,
    patch_environment_browser_javascript,
    patch_environments_rst,
    render_comprehensive_list_table,
    render_environment_browser_task_rows,
)

import isaaclab_tasks  # noqa: E402, F401

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: E402, F401


def _default_rst_path() -> Path:
    return Path(__file__).resolve().parents[1] / "docs" / "source" / "overview" / "environments.rst"


def _default_browser_path() -> Path:
    return Path(__file__).resolve().parents[1] / "docs" / "source" / "_static" / "css" / "environment-browser.js"


def main() -> int:
    """Generate and optionally write the environment tables."""
    parser = argparse.ArgumentParser(description="Update environment documentation from the Gym registry.")
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_rst_path(),
        help="Path to environments.rst (default: docs/source/overview/environments.rst).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with code 1 when generated environment documentation is out of date instead of rewriting it.",
    )
    parser.add_argument(
        "--browser_output",
        type=Path,
        default=_default_browser_path(),
        help="Path to environment-browser.js (default: docs/source/_static/css/environment-browser.js).",
    )
    args = parser.parse_args()

    rows = collect_environment_doc_rows()
    generated_table = render_comprehensive_list_table(rows)

    output_path = args.output.resolve()
    original = output_path.read_text(encoding="utf-8")
    updated = patch_environments_rst(original, generated_table)
    updated = patch_curated_environment_tables(updated, rows)

    browser_output_path = args.browser_output.resolve()
    browser_original = browser_output_path.read_text(encoding="utf-8")
    browser_rows = render_environment_browser_task_rows(rows)
    browser_updated = patch_environment_browser_javascript(browser_original, browser_rows)

    if args.check:
        outdated_paths = []
        if updated != original:
            outdated_paths.append(output_path)
        if browser_updated != browser_original:
            outdated_paths.append(browser_output_path)
        if outdated_paths:
            paths = ", ".join(str(path) for path in outdated_paths)
            print(
                f"[ERROR] Generated environment documentation is out of date: {paths}. "
                "Run uv run python tools/update_environments_rst.py to refresh it.",
                file=sys.stderr,
            )
            return 1
        print(f"[INFO] Environment documentation is up to date ({len(rows)} training environments).")
        return 0

    updated_paths = []
    if updated != original:
        output_path.write_text(updated, encoding="utf-8")
        updated_paths.append(output_path)
    if browser_updated != browser_original:
        browser_output_path.write_text(browser_updated, encoding="utf-8")
        updated_paths.append(browser_output_path)

    if updated_paths:
        paths = ", ".join(str(path) for path in updated_paths)
        print(f"[INFO] Updated {paths} with {len(rows)} training environments.")
    else:
        print(f"[INFO] Environment documentation already up to date ({len(rows)} training environments).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
