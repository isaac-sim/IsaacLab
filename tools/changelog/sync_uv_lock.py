# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Re-point ``uv.lock``'s workspace-member versions at ``pyproject.toml``.

``cli.py compile`` bumps ``version`` in each package's ``pyproject.toml``,
but ``uv.lock`` pins those same workspace members by version. Left behind,
every ``uv sync --locked`` / ``--frozen`` consumer (notably the install-ci
suite) fails with "the lockfile needs to be updated".

Running ``uv lock`` in the nightly job would fix that, but it is a full
resolve: it can rewrite third-party pins, hashes, and markers in ways that
warrant human review and must not land unreviewed in an auto-commit. This
script does the narrow thing instead — rewrite the ``version`` line of the
workspace members' own ``[[package]]`` blocks and nothing else. No network,
no resolution, no third-party churn.

That narrowness is also its limit. The script deliberately fails when the
lock is stale in a way a version rewrite cannot fix (a package added to or
removed from ``source/``, or a dependency edit), because those genuinely
need a real ``uv lock`` and a reviewed diff.

Usage:

    # Rewrite uv.lock in place (run after ``cli.py compile --all``):
    python3 tools/changelog/sync_uv_lock.py

    # Report what would change, write nothing. Exits non-zero if the lock
    # is out of sync, so it doubles as a CI gate:
    python3 tools/changelog/sync_uv_lock.py --check
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

from cli import REPO_ROOT, Package, _display_path

LOCK_PATH = REPO_ROOT / "uv.lock"
ROOT_TOML_PATH = REPO_ROOT / "pyproject.toml"

# A ``[[package]]`` block's own ``name`` / ``version`` fields sit at column
# zero. The same keys appear indented inside ``dependencies`` and
# ``requires-dist`` tables (``    { name = "isaaclab" },``), so anchoring at
# the line start is what keeps this from rewriting a dependency reference.
PACKAGE_HEADER = "[[package]]"
NAME_RE = re.compile(r'^name = "([^"]+)"')
VERSION_RE = re.compile(r'^version = "([^"]+)"')


def read_lock_members(lock_path: Path = LOCK_PATH) -> dict[str, tuple[str, Path]]:
    """Return ``{package name: (locked version, package root)}`` for workspace members.

    Members are the entries uv records as ``source = { editable = "<path>" }``.
    The virtual root (``source = { virtual = "." }``) carries no version of its
    own and is skipped.
    """
    data = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    members: dict[str, tuple[str, Path]] = {}
    for pkg in data.get("package", []):
        editable = pkg.get("source", {}).get("editable")
        if editable is not None:
            members[pkg["name"]] = (pkg["version"], REPO_ROOT / editable)
    return members


def resolve_target_versions(lock_path: Path = LOCK_PATH) -> dict[str, str]:
    """Return ``{package name: version}`` that ``uv.lock`` should record.

    The version is read out of each member's ``pyproject.toml`` with the same
    parser :meth:`cli.Package.current_version` uses, so this script and the
    compiler cannot drift on what "the version" means.
    """
    return {name: Package(root).current_version().text for name, (_, root) in read_lock_members(lock_path).items()}


def read_declared_members(root_toml_path: Path = ROOT_TOML_PATH) -> set[Path]:
    """Return the package roots the root ``pyproject.toml`` declares as editable.

    ``[tool.uv.sources]`` is the authoritative member list — not ``source/*/``,
    which also holds directories uv does not track, and not
    :meth:`cli.Package.discover`, which filters to packages the changelog
    compiler manages (``isaaclab_tasks_experimental`` is a member with no
    ``CHANGELOG.rst``).
    """
    sources = tomllib.loads(root_toml_path.read_text(encoding="utf-8"))["tool"]["uv"]["sources"]
    return {
        REPO_ROOT / spec["path"] for spec in sources.values() if isinstance(spec, dict) and spec.get("editable") is True
    }


def assert_lock_is_repairable(lock_path: Path = LOCK_PATH, root_toml_path: Path = ROOT_TOML_PATH) -> None:
    """Fail when ``uv.lock``'s member set no longer matches the declared members.

    A version rewrite can only fix a lock whose membership is already correct.
    A package added to or removed from ``[tool.uv.sources]`` needs a full
    ``uv lock`` (new dependency edges, new transitive pins), so refuse loudly
    rather than write a lock that is still stale in a way the caller cannot see.
    """
    declared = {p.resolve() for p in read_declared_members(root_toml_path)}
    in_lock = {root.resolve() for _, root in read_lock_members(lock_path).values()}
    missing = sorted(_display_path(p) for p in declared - in_lock)
    extra = sorted(_display_path(p) for p in in_lock - declared)
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"declared but absent from the lock: {', '.join(missing)}")
        if extra:
            detail.append(f"locked but no longer declared: {', '.join(extra)}")
        raise SystemExit(
            f"{_display_path(lock_path)} membership does not match {_display_path(root_toml_path)} "
            f"({'; '.join(detail)}). This needs a full `uv lock` and a reviewed diff, not a version rewrite."
        )


def rewrite_versions(text: str, targets: dict[str, str]) -> tuple[str, list[tuple[str, str, str]]]:
    """Return ``(new lock text, [(package, old version, new version), ...])``.

    Only the ``version`` line of a named package's own ``[[package]]`` block is
    touched; every other byte of the file is passed through verbatim.
    """
    out: list[str] = []
    changes: list[tuple[str, str, str]] = []
    # Name of the block currently open, or None when we are outside a
    # ``[[package]]`` block or have already rewritten its version.
    current: str | None = None
    for line in text.splitlines(keepends=True):
        if line.startswith("["):
            # Any table header — ``[[package]]``, ``[package.metadata]``, ... —
            # closes the previous block's name/version pair. Belt-and-braces:
            # uv always emits ``name`` before ``version`` in a block, so the
            # next ``name`` would rescope anyway. Keeping the key bound to its
            # own table is cheap next to the cost of corrupting a lockfile.
            current = None
        elif (match := NAME_RE.match(line)) is not None:
            current = match.group(1)
        elif current is not None and (match := VERSION_RE.match(line)) is not None:
            old, new = match.group(1), targets.get(current)
            if new is not None and new != old:
                line = f'version = "{new}"\n'
                changes.append((current, old, new))
            current = None
        out.append(line)
    return "".join(out), changes


def sync(lock_path: Path = LOCK_PATH, *, check: bool) -> int:
    """Sync (or, under ``check``, report on) the lock. Returns a process exit code."""
    assert_lock_is_repairable(lock_path)
    text = lock_path.read_text(encoding="utf-8")
    updated, changes = rewrite_versions(text, resolve_target_versions(lock_path))

    if not changes:
        print(f"{_display_path(lock_path)} is already in sync with the workspace pyproject.toml versions.")
        return 0

    for name, old, new in changes:
        print(f"  {name}: {old} -> {new}")

    if check:
        print(f"\n{_display_path(lock_path)} is out of sync. Run `python3 tools/changelog/sync_uv_lock.py` to fix.")
        return 1

    lock_path.write_text(updated, encoding="utf-8")
    print(f"\nUpdated {len(changes)} version(s) in {_display_path(lock_path)}.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report drift and exit non-zero instead of writing uv.lock.",
    )
    sys.exit(sync(check=parser.parse_args().check))


if __name__ == "__main__":
    main()
