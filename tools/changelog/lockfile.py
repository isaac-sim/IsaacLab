# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The workspace ``uv.lock``, viewed as a set of member version pins.

``cli.py compile`` bumps ``version`` in each package's version metadata
file, but ``uv.lock`` pins those same workspace members by version. Left
behind, every ``uv sync --locked`` / ``--frozen`` consumer (notably the
install-ci suite) fails with "the lockfile needs to be updated".

Running ``uv lock`` in the nightly job would fix that, but it is a full
resolve: it can rewrite third-party pins, hashes, and markers in ways that
warrant human review and must not land unreviewed in an auto-commit.
:class:`LockFile` does the narrow thing instead — rewrite the ``version``
line of the workspace members' own ``[[package]]`` blocks and nothing else.
No network, no resolution, no third-party churn.

That narrowness is also its limit, and the limit is enforced rather than
just documented: :meth:`LockFile.assert_repairable` refuses to touch a lock
whose *membership* no longer matches the workspace declaration, because a
package added to or removed from ``[tool.uv.sources]`` brings new dependency
edges and transitive pins that a version rewrite cannot invent.

This module deliberately does not import ``cli``: ``cli`` imports it, and
the version reader it needs arrives by injection instead. That keeps the
class testable on its own and lets the caller decide where "the version"
comes from — :class:`cli.Package` passes its own parser, so the compiler
and the lock cannot drift on the answer.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import tomllib


class LockFile:
    """A ``uv.lock`` whose workspace-member versions can be re-pointed in place.

    Args:
        repo_root: Workspace root — the directory holding ``uv.lock`` and the
            root ``pyproject.toml``, and the base that member paths recorded
            in the lock resolve against.
        version_of: Maps a member's package root to the version its own
            manifest declares, or ``None`` when that member carries no
            readable version. Injected so this class does not have to know
            how a package stores its version.
    """

    class Error(RuntimeError):
        """Raised when the lock exists but cannot be synced.

        Callers catch this one type; the subclass distinguishes the case
        worth a specific message.
        """

    class MembershipMismatch(Error):
        """Raised when the lock's member set no longer matches the workspace."""

    LOCK_NAME = "uv.lock"
    ROOT_TOML_NAME = "pyproject.toml"

    # A ``[[package]]`` block's own ``name`` / ``version`` fields sit at column
    # zero. The same keys appear indented inside ``dependencies`` and
    # ``requires-dist`` tables (``    { name = "isaaclab" },``), so anchoring
    # at the line start is what keeps this from rewriting a dependency
    # reference into a package pin.
    _NAME_RE = re.compile(r'^name = "([^"]+)"')
    _VERSION_RE = re.compile(r'^version = "([^"]+)"')

    def __init__(self, repo_root: Path, version_of: Callable[[Path], str | None]):
        self.repo_root = repo_root
        self.version_of = version_of

    @property
    def path(self) -> Path:
        """Absolute path to the lockfile (which may not exist)."""
        return self.repo_root / self.LOCK_NAME

    @property
    def root_toml_path(self) -> Path:
        """Absolute path to the workspace root ``pyproject.toml``."""
        return self.repo_root / self.ROOT_TOML_NAME

    @property
    def exists(self) -> bool:
        """Whether this branch carries a lockfile at all.

        Release branches cut before the uv workspace landed have no
        ``uv.lock``; on those, every operation here is a no-op rather than
        an error.
        """
        return self.path.is_file()

    def locked_members(self) -> dict[str, Path]:
        """Return ``{package name: package root}`` for the lock's workspace members.

        Members are the entries uv records as ``source = { editable = "<path>" }``.
        The virtual workspace root (``source = { virtual = "." }``) carries no
        version of its own and is skipped.
        """
        data = tomllib.loads(self.path.read_text(encoding="utf-8"))
        return {
            pkg["name"]: self.repo_root / editable
            for pkg in data.get("package", [])
            if (editable := pkg.get("source", {}).get("editable")) is not None
        }

    def declared_members(self) -> set[Path]:
        """Return the package roots the root ``pyproject.toml`` declares as editable.

        ``[tool.uv.sources]`` is the authoritative member list — not
        ``source/*/``, which also holds directories uv does not track, and not
        the changelog compiler's own package discovery, which filters to
        packages with a ``CHANGELOG.rst`` (``isaaclab_tasks_experimental`` is
        a workspace member with no changelog).
        """
        data = tomllib.loads(self.root_toml_path.read_text(encoding="utf-8"))
        sources = data.get("tool", {}).get("uv", {}).get("sources", {})
        return {
            self.repo_root / spec["path"]
            for spec in sources.values()
            if isinstance(spec, dict) and spec.get("editable") is True
        }

    def assert_repairable(self) -> None:
        """Raise :class:`MembershipMismatch` when a version rewrite cannot fix this lock.

        A rewrite can only repair a lock whose membership is already correct.
        A package added to or removed from ``[tool.uv.sources]`` needs a full
        ``uv lock`` — new dependency edges, new transitive pins — so refuse
        loudly rather than write a lock that is still stale in a way the
        caller cannot see.
        """
        declared = {p.resolve() for p in self.declared_members()}
        in_lock = {root.resolve() for root in self.locked_members().values()}
        missing = sorted(str(p) for p in declared - in_lock)
        extra = sorted(str(p) for p in in_lock - declared)
        if not missing and not extra:
            return
        detail = []
        if missing:
            detail.append(f"declared but absent from the lock: {', '.join(missing)}")
        if extra:
            detail.append(f"locked but no longer declared: {', '.join(extra)}")
        raise self.MembershipMismatch(
            f"{self.LOCK_NAME} membership does not match {self.ROOT_TOML_NAME} ({'; '.join(detail)}). "
            f"This needs a full `uv lock` and a reviewed diff, not a version rewrite."
        )

    def target_versions(self) -> dict[str, str]:
        """Return ``{package name: version}`` the lock should record.

        Members whose :paramref:`version_of` lookup comes back ``None`` are
        omitted, which leaves their locked version untouched.
        """
        targets = {}
        for name, root in self.locked_members().items():
            version = self.version_of(root)
            if version is not None:
                targets[name] = version
        return targets

    @classmethod
    def rewrite(cls, text: str, targets: dict[str, str]) -> tuple[str, list[tuple[str, str, str]]]:
        """Return ``(new lock text, [(package, old version, new version), ...])``.

        Only the ``version`` line of a named package's own ``[[package]]``
        block is touched; every other byte of the file is passed through
        verbatim.

        Args:
            text: Current lockfile contents.
            targets: Versions to write, keyed by package name. Names absent
                from this mapping are left alone.
        """
        out: list[str] = []
        changes: list[tuple[str, str, str]] = []
        # Name of the block currently open, or None when we are outside a
        # ``[[package]]`` block or have already rewritten its version.
        current: str | None = None
        for line in text.splitlines(keepends=True):
            if line.startswith("["):
                # Any table header — ``[[package]]``, ``[package.metadata]``,
                # ... — closes the previous block's name/version pair.
                # Belt-and-braces: uv always emits ``name`` before ``version``
                # in a block, so the next ``name`` would rescope anyway.
                # Keeping the key bound to its own table is cheap next to the
                # cost of corrupting a lockfile.
                current = None
            elif (match := cls._NAME_RE.match(line)) is not None:
                current = match.group(1)
            elif current is not None and (match := cls._VERSION_RE.match(line)) is not None:
                old, new = match.group(1), targets.get(current)
                if new is not None and new != old:
                    line = f'version = "{new}"\n'
                    changes.append((current, old, new))
                current = None
            out.append(line)
        return "".join(out), changes

    def drift(self) -> tuple[str, list[tuple[str, str, str]]]:
        """Return the rewritten text and the drift it would resolve, writing nothing."""
        text = self.path.read_text(encoding="utf-8")
        return self.rewrite(text, self.target_versions())

    def sync(self, *, dry_run: bool = False) -> list[Path]:
        """Re-point the lock's member versions at their manifests.

        Returns the list of paths written — ``[self.path]`` when the lock
        changed, empty otherwise. Returning the touched paths (rather than a
        bool) is what lets :class:`cli.AutoBumpRun` stage the lock through
        the same manifest as every other compiler output, with no separate
        ``git add`` to keep in sync.

        Args:
            dry_run: Report the drift and write nothing.

        Raises:
            MembershipMismatch: The lock needs a full ``uv lock``, not a
                version rewrite.
            Error: The lock or the root manifest could not be read.
        """
        if not self.exists:
            print(f"No {self.LOCK_NAME} on this branch — nothing to sync.")
            return []
        try:
            self.assert_repairable()
            updated, changes = self.drift()
        except (OSError, tomllib.TOMLDecodeError) as e:
            # Unreadable or malformed TOML on either side. Surfaced as this
            # module's own error type so callers need not import tomllib to
            # write an ``except`` clause.
            raise self.Error(f"could not read {self.LOCK_NAME} or {self.ROOT_TOML_NAME}: {e}") from e
        if not changes:
            print(f"{self.LOCK_NAME} is already in sync with the workspace versions.")
            return []
        for name, old, new in changes:
            print(f"  {name}: {old} -> {new}")
        if dry_run:
            print(f"DRY RUN — would update {len(changes)} version(s) in {self.LOCK_NAME}.")
            return []
        self.path.write_text(updated, encoding="utf-8")
        print(f"Updated {len(changes)} version(s) in {self.LOCK_NAME}.")
        return [self.path]
