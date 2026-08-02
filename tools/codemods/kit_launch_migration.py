# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rewrite test modules from a module-scope ``AppLauncher`` to the shared ``launch_kit()``.

A test module that constructs :class:`~isaaclab.app.AppLauncher` at module scope boots its
own Kit app during pytest collection, so a process covering several such files pays Kit
startup once per file. :func:`~isaaclab.test.launch.launch_kit` is idempotent, so migrated
files share one app per process.

The rewrite is deliberately in-place and line-based rather than an ``ast.unparse`` round
trip, which would discard comments, ``# isort:skip`` directives, and docstring formatting.
Each edit replaces a statement's own line range, so import ordering -- which matters here,
because Kit must boot before the Kit-dependent imports below it -- is preserved exactly.

Usage::

    uv run python tools/codemods/kit_launch_migration.py source/isaaclab/test/sim
    uv run python tools/codemods/kit_launch_migration.py --check source/isaaclab/test/sim

Files the transform cannot handle safely are reported and left untouched.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

_LAUNCH_IMPORT = "from isaaclab.test.launch import launch_kit"
_APP_IMPORT_MODULE = "isaaclab.app"

# Docstrings used purely as section separators around the old launch block. They document a
# launch step that no longer exists in the file once it is migrated.
_BOILERPLATE_DOCSTRINGS = ("Launch Isaac Sim Simulator first.", "Rest everything follows.")

_BOILERPLATE_COMMENTS = ("# launch omniverse app", "# launch the simulator")


class Unsupported(Exception):
    """Raised when a file needs manual attention rather than a mechanical rewrite."""


def _module_scope_nodes(tree: ast.Module):
    """Yield nodes that execute at import, without descending into callables."""
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _name_usage_count(tree: ast.Module, name: str) -> int:
    """Count how many times ``name`` is loaded anywhere in the module."""
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == name)


def _find_launcher(tree: ast.Module) -> tuple[ast.stmt, ast.Call]:
    """Return the module-scope statement that builds the app, and the ``AppLauncher`` call."""
    found = []
    for statement in tree.body:
        for node in ast.walk(statement):
            if _call_name(node) == "SimulationApp":
                raise Unsupported("constructs SimulationApp directly")
            if _call_name(node) == "AppLauncher":
                found.append((statement, node))

    if not found:
        raise Unsupported("no module-scope AppLauncher call")
    if len(found) > 1:
        raise Unsupported(f"{len(found)} module-scope AppLauncher calls")

    statement, call = found[0]

    # The whole statement is replaced by a bare launch_kit() call, so the launch must be
    # unconditional. A file that boots Kit only on some branch -- e.g.
    # `AppLauncher(...).app if _USE_KIT else None`, used where a standalone wheel lets the
    # tests run kitlessly -- would silently become an unconditional boot. Accept only
    # `<target> = AppLauncher(...)`, `<target> = AppLauncher(...).app`, or a bare call.
    value = statement.value if isinstance(statement, ast.Assign | ast.Expr) else None
    if isinstance(value, ast.Attribute):
        value = value.value
    if value is not call:
        raise Unsupported(f"AppLauncher launch is conditional or nested: `{ast.unparse(statement).splitlines()[0]}`")

    # `AppLauncher` must not be referenced for anything else, since its import is removed.
    if _name_usage_count(tree, "AppLauncher") > 1:
        raise Unsupported("`AppLauncher` is referenced beyond the launch call")

    return statement, call


def _resolve_cameras(call: ast.Call) -> bool:
    """Map the AppLauncher keywords onto the ``cameras`` argument of ``launch_kit``."""
    if call.args:
        raise Unsupported("AppLauncher called with positional arguments")

    cameras = False
    for keyword in call.keywords:
        if keyword.arg is None:
            raise Unsupported("AppLauncher called with **kwargs")
        value = keyword.value
        literal = value.value if isinstance(value, ast.Constant) else None

        if keyword.arg == "headless":
            # `headless=True`, or `headless=HEADLESS` where HEADLESS is a True constant.
            if literal is not True and not isinstance(value, ast.Name):
                raise Unsupported(f"headless={ast.unparse(value)} is not a literal True")
        elif keyword.arg == "enable_cameras":
            if not isinstance(literal, bool):
                raise Unsupported(f"enable_cameras={ast.unparse(value)} is not a literal bool")
            cameras = literal
        elif keyword.arg == "device":
            # launch_kit always applies resolve_test_sim_device(); anything else is a real
            # difference in behaviour and must be looked at by hand.
            if ast.unparse(value) != "resolve_test_sim_device()":
                raise Unsupported(f"device={ast.unparse(value)} is not resolve_test_sim_device()")
        else:
            raise Unsupported(f"unsupported AppLauncher keyword {keyword.arg}=")

    return cameras


def _pytestmark_statement(tree: ast.Module) -> ast.Assign | None:
    marks = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets)
    ]
    if len(marks) > 1:
        raise Unsupported("multiple module-scope pytestmark assignments; merge them first")
    return marks[0] if marks else None


def _render_pytestmark(existing: ast.Assign | None, marker: str) -> str:
    """Build the new ``pytestmark`` line with the Kit marker in front."""
    new = f"pytest.mark.{marker}"
    if existing is None:
        return f"pytestmark = {new}"
    value = existing.value
    if isinstance(value, ast.List | ast.Tuple):
        parts = [new] + [ast.unparse(element) for element in value.elts]
    else:
        parts = [new, ast.unparse(value)]
    return f"pytestmark = [{', '.join(parts)}]"


def _is_boilerplate_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        and node.value.value.strip() in _BOILERPLATE_DOCSTRINGS
    )


def migrate_source(source: str) -> tuple[str, str]:
    """Return the rewritten source and the marker it should carry.

    Raises:
        Unsupported: If the file needs manual attention.
    """
    tree = ast.parse(source)
    statement, call = _find_launcher(tree)
    cameras = _resolve_cameras(call)
    marker = "kit_cameras" if cameras else "kit"
    existing_mark = _pytestmark_statement(tree)

    lines = source.splitlines()
    # 1-indexed line numbers to drop entirely.
    drop: set[int] = set()
    # 1-indexed line number -> replacement text.
    replace: dict[int, str] = {}
    # 1-indexed line number -> text appended after that line.
    insert_after: dict[int, list[str]] = {}

    # The launch statement becomes the launch_kit() call, in place, so that the Kit-dependent
    # imports below it still run after Kit has started.
    replace[statement.lineno] = "launch_kit(cameras=True)" if cameras else "launch_kit()"
    drop.update(range(statement.lineno + 1, (statement.end_lineno or statement.lineno) + 1))

    # `from isaaclab.app import AppLauncher` becomes the launch_kit import, keeping its slot.
    app_import_replaced = False
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == _APP_IMPORT_MODULE:
            names = [alias.name for alias in node.names]
            if names == ["AppLauncher"]:
                replace[node.lineno] = _LAUNCH_IMPORT
                drop.update(range(node.lineno + 1, (node.end_lineno or node.lineno) + 1))
                app_import_replaced = True
            else:
                raise Unsupported(f"`from isaaclab.app import {', '.join(names)}` imports more than AppLauncher")
    if not app_import_replaced:
        raise Unsupported("no `from isaaclab.app import AppLauncher` to replace")

    # Drop `resolve_test_sim_device` imports that only existed to feed AppLauncher, and
    # `HEADLESS = True` constants that nothing else reads. launch_kit covers both.
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "isaaclab.test.utils":
            names = [alias.name for alias in node.names]
            if "resolve_test_sim_device" not in names or _name_usage_count(tree, "resolve_test_sim_device") != 1:
                continue
            remaining = [name for name in names if name != "resolve_test_sim_device"]
            span = range(node.lineno, (node.end_lineno or node.lineno) + 1)
            if remaining:
                # Keep the other names; re-emit as a single line, which is how these imports
                # are already written and how the formatter would leave them.
                replace[node.lineno] = f"from {node.module} import {', '.join(remaining)}"
                drop.update(list(span)[1:])
            else:
                drop.update(span)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Name)
                and target.id in ("HEADLESS", "headless")
                and _name_usage_count(tree, target.id) == 1
            ):
                drop.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))

    # Drop the separator docstrings and comments that described the removed launch block.
    for node in tree.body:
        if _is_boilerplate_docstring(node):
            drop.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    for index, line in enumerate(lines, start=1):
        if line.strip().lower() in _BOILERPLATE_COMMENTS:
            drop.add(index)

    # Attach the marker, either by extending the existing pytestmark or by adding one after
    # the last module-scope import (where such a declaration conventionally sits).
    marked = _render_pytestmark(existing_mark, marker)
    if existing_mark is not None:
        replace[existing_mark.lineno] = marked
        drop.update(range(existing_mark.lineno + 1, (existing_mark.end_lineno or existing_mark.lineno) + 1))
    else:
        import_ends = [
            node.end_lineno or node.lineno for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)
        ]
        if not import_ends:
            raise Unsupported("no imports to anchor a new pytestmark to")
        if _name_usage_count(tree, "pytest") == 0 and not any(
            isinstance(node, ast.Import) and any(a.name == "pytest" for a in node.names) for node in tree.body
        ):
            raise Unsupported("pytest is not imported, so a pytestmark cannot be added")
        insert_after.setdefault(max(import_ends), []).append(marked)

    # Only the header is rewritten, so blank-line cleanup is confined to it. Collapsing
    # runs across the whole file would also eat the blank lines PEP 8 requires between
    # top-level definitions and produce a diff far larger than the change being made.
    header_end = max([*drop, *replace, *insert_after, 1])

    out: list[str] = []
    for index, line in enumerate(lines, start=1):
        if index in replace:
            emitted = replace[index]
        elif index not in drop:
            emitted = line
        else:
            emitted = None

        if emitted is not None:
            in_header = index <= header_end
            if not (in_header and not emitted.strip() and out and not out[-1].strip()):
                out.append(emitted)

        for extra in insert_after.get(index, []):
            out.extend(["", extra])

    result = "\n".join(out).rstrip("\n") + "\n"
    ast.parse(result)  # refuse to emit anything that does not parse
    return result, marker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="files or directories to migrate")
    parser.add_argument("--check", action="store_true", help="report what would change without writing")
    args = parser.parse_args(argv)

    targets: list[Path] = []
    for path in args.paths:
        targets.extend(sorted(path.rglob("test_*.py")) if path.is_dir() else [path])

    changed, skipped = [], []
    for path in targets:
        source = path.read_text(encoding="utf-8")
        try:
            new_source, marker = migrate_source(source)
        except Unsupported as exc:
            skipped.append((path, str(exc)))
            continue
        except SyntaxError as exc:
            skipped.append((path, f"produced invalid syntax: {exc}"))
            continue
        if new_source != source and not args.check:
            path.write_text(new_source, encoding="utf-8", newline="\n")
        changed.append((path, marker))

    for path, marker in changed:
        print(f"{'would migrate' if args.check else 'migrated'}: {path.as_posix()} -> {marker}")
    for path, reason in skipped:
        print(f"skipped: {path.as_posix()}: {reason}", file=sys.stderr)
    print(f"\n{len(changed)} migrated, {len(skipped)} skipped, {len(targets)} scanned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
