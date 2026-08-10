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
No network, no resolution, no third-party churn. Measured on this repo, that
is an 8-line diff where a full ``uv lock`` rewrote 3136 lines; both satisfy
``uv lock --check``.

That narrowness is also its limit, and the limit is enforced rather than
just documented: :meth:`LockFile.assert_repairable` refuses to touch a lock
whose *membership* no longer matches the workspace declaration, because a
package added to or removed from ``[tool.uv.sources]`` brings new dependency
edges and transitive pins that a version rewrite cannot invent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import tomllib
from packages import Package, RootPackage


class LockFile:
    """A ``uv.lock`` whose workspace-member versions can be re-pointed in place.

    Args:
        root_package: The workspace declaration this lock is validated
            against — membership lives there, versions live here.
    """

    # ---- Nested types ---------------------------------------------------

    class Error(RuntimeError):
        """Raised when the lock exists but cannot be synced.

        Callers catch this one type; the subclass distinguishes the case
        worth a specific message.
        """

    class MembershipMismatch(Error):
        """Raised when the lock's member set no longer matches the workspace."""

    @dataclass(frozen=True)
    class Drift:
        """One member's pin moving: ``(package, old version, new version)``."""

        package: str
        old: str
        new: str

    # ---- Class constants ------------------------------------------------

    LOCK_NAME = "uv.lock"
    PACKAGE_HEADER = "[[package]]"

    # A ``[[package]]`` block's own ``name`` / ``version`` fields sit at column
    # zero. The same keys appear indented inside ``dependencies`` and
    # ``requires-dist`` tables (``    { name = "isaaclab" },``), so anchoring
    # at the line start is what keeps this from rewriting a dependency
    # reference into a package pin.
    _NAME_RE = re.compile(r'^name = "([^"]+)"')
    _VERSION_RE = re.compile(r'^version = "([^"]+)"')

    # ---- Construction ---------------------------------------------------

    def __init__(self, root_package: RootPackage):
        self.root_package = root_package

    # ---- Properties -----------------------------------------------------

    @property
    def path(self) -> Path:
        """Absolute path to the lockfile (which may not exist)."""
        return self.root_package.root / self.LOCK_NAME

    @property
    def exists(self) -> bool:
        """Whether this branch carries a lockfile at all.

        Release branches cut before the uv workspace landed have no
        ``uv.lock``; on those, every operation here is a no-op rather than
        an error.
        """
        return self.path.is_file()

    # ---- Public API: inspect the lock, then repair it --------------------

    def check(self) -> list[Drift]:
        """Return the drift a :meth:`sync` would resolve, writing nothing.

        Raises:
            MembershipMismatch: The lock needs a full ``uv lock``.
            Error: The lock or the root manifest could not be read.
        """
        if not self.exists:
            return []
        return self._guarded_drift()[1]

    def sync(self, *, dry_run: bool = False) -> list[Path]:
        """Re-point the lock's member versions at their manifests.

        Returns the list of paths written — ``[self.path]`` when the lock
        changed, empty otherwise. Returning the touched paths (rather than a
        bool) is what lets :class:`autobump.AutoBumpRun` stage the lock
        through the same manifest as every other compiler output, with no
        separate ``git add`` to keep in sync.

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
        updated, drifts = self._guarded_drift()
        if not drifts:
            print(f"{self.LOCK_NAME} is already in sync with the workspace versions.")
            return []
        for d in drifts:
            print(f"  {d.package}: {d.old} -> {d.new}")
        if dry_run:
            print(f"DRY RUN — would update {len(drifts)} version(s) in {self.LOCK_NAME}.")
            return []
        self._assert_rewrite_is_sound(self._document[0], updated, drifts)
        self._write(updated)
        print(f"Updated {len(drifts)} version(s) in {self.LOCK_NAME}.")
        return [self.path]

    @classmethod
    def _assert_rewrite_is_sound(cls, before: str, after: str, drifts: list[Drift]) -> None:
        """Verify the rewritten lock before it is allowed to reach disk.

        The rewrite is careful — column-zero anchors, block-index scoping —
        but that scoping is precisely what could be wrong, so it cannot be
        its own witness. This re-derives the outcome from the produced text.

        The lock is never bad in an *interesting* way: it is machine-written
        and the nightly commits it unreviewed, so a malformed or mis-pinned
        result would be discovered by whoever next runs ``uv sync`` rather
        than here. Checking costs one parse of an 8k-line file.

        Args:
            before: Lock text as read.
            after: Lock text the rewrite produced.
            drifts: The moves the rewrite reported making.

        Raises:
            Error: The result is not a sound realisation of ``drifts``.
        """

        def pins(text: str) -> list[tuple[str, str]]:
            """``(name, version)`` per ``[[package]]`` block, in file order."""
            data = tomllib.loads(text)
            return [(p["name"], p["version"]) for p in data.get("package", []) if "version" in p]

        # V1 — still TOML. Anything else here is unsafe to reason about.
        try:
            after_pins = pins(after)
        except tomllib.TOMLDecodeError as e:
            raise cls.Error(f"the {cls.LOCK_NAME} rewrite produced invalid TOML: {e}") from e

        # V2 — a version rewrite replaces lines; it never adds or drops any.
        if (a := len(after.splitlines())) != (b := len(before.splitlines())):
            raise cls.Error(
                f"the {cls.LOCK_NAME} rewrite changed the line count ({b} -> {a}); expected in-place edits only."
            )

        before_pins = pins(before)
        if len(before_pins) != len(after_pins):
            raise cls.Error(
                f"the {cls.LOCK_NAME} rewrite changed the package count ({len(before_pins)} -> {len(after_pins)})."
            )

        # V3/V4 — exactly the reported moves happened, and nothing else did.
        # Compared position-wise: two blocks may share a name (an editable
        # member and a registry release of the same project), so a name-keyed
        # comparison would let a rewrite of the wrong one pass unnoticed.
        moved = {i for i, (b_pin, a_pin) in enumerate(zip(before_pins, after_pins)) if b_pin != a_pin}
        expected = {(d.package, d.old, d.new) for d in drifts}
        actual = {(after_pins[i][0], before_pins[i][1], after_pins[i][1]) for i in moved}
        if actual != expected:
            raise cls.Error(
                f"the {cls.LOCK_NAME} rewrite did not match the drift it reported "
                f"(reported {sorted(expected)}, applied {sorted(actual)})."
            )

    def assert_repairable(self) -> None:
        """Raise :class:`MembershipMismatch` when a version rewrite cannot fix this lock.

        A rewrite can only repair a lock whose membership is already correct.
        A package added to or removed from ``[tool.uv.sources]`` needs a full
        ``uv lock`` — new dependency edges, new transitive pins — so refuse
        loudly rather than write a lock that is still stale in a way the
        caller cannot see.
        """
        declared = {p.resolve() for p in self.root_package.declared_members()}
        in_lock = {root.resolve() for root in self._locked_members().values()}
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
            f"{self.LOCK_NAME} membership does not match {self.root_package.path.name} "
            f"({'; '.join(detail)}). This needs a full `uv lock` and a reviewed diff, "
            f"not a version rewrite."
        )

    # ---- Internals ------------------------------------------------------

    @cached_property
    def _document(self) -> tuple[str, dict]:
        """One read of the lock: raw text for rewriting, parsed form for queries.

        Cached because a single operation asks four separate questions of the
        same file — membership, member paths, target versions, block
        positions — and re-reading for each is pure waste.

        The cache is dropped by :meth:`_write`. That invalidation belongs to
        this class, not to a convention about how often callers may call:
        the object models a file that it also mutates, so keeping the two in
        agreement is part of what it means to model that file at all.
        """
        text = self.path.read_text(encoding="utf-8")
        return text, tomllib.loads(text)

    def _write(self, text: str) -> None:
        """Write the lock and drop the now-stale cached read."""
        self.path.write_text(text, encoding="utf-8")
        self.__dict__.pop("_document", None)

    def _guarded_drift(self) -> tuple[str, list[Drift]]:
        """Compute the rewrite behind the membership guard and the TOML guard.

        Every entry point goes through here, so none of them can report a
        clean lock while :meth:`assert_repairable` would have raised, or
        surface a parser error as a bare traceback.
        """
        try:
            self.assert_repairable()
            text, _ = self._document
            return self._rewrite(text, self._target_versions(), self._editable_blocks())
        except (OSError, tomllib.TOMLDecodeError) as e:
            # Unreadable or malformed TOML on either side. Surfaced as this
            # module's own error type so callers need not import tomllib to
            # write an ``except`` clause.
            raise self.Error(f"could not read {self.LOCK_NAME} or {self.root_package.path.name}: {e}") from e

    def _locked_members(self) -> dict[str, Path]:
        """Return ``{package name: package root}`` for the lock's workspace members.

        Members are the entries uv records as ``source = { editable = "<path>" }``.
        The virtual workspace root (``source = { virtual = "." }``) carries no
        version of its own and is skipped.
        """
        _, data = self._document
        return {
            pkg["name"]: self.root_package.root / editable
            for pkg in data.get("package", [])
            if (editable := pkg.get("source", {}).get("editable")) is not None
        }

    def _target_versions(self) -> dict[str, str]:
        """Return ``{package name: version}`` the lock should record.

        Members whose manifest carries no readable version are omitted, which
        leaves their locked version untouched.
        """
        targets = {}
        for name, root in self._locked_members().items():
            version = Package.declared_version(root)
            if version is not None:
                targets[name] = version
        return targets

    def _editable_blocks(self) -> set[int]:
        """Return the positions of the editable members' ``[[package]]`` blocks.

        Package *names* are not a safe key for the rewrite. A lock may carry
        two blocks under one name — an editable workspace member and a
        registry release of the same project — and a name-keyed rewrite would
        move the registry block's pin too, leaving its hashes and source
        metadata describing a different version. Blocks are counted in file
        order instead, which is unambiguous.
        """
        _, data = self._document
        return {i for i, pkg in enumerate(data.get("package", [])) if pkg.get("source", {}).get("editable") is not None}

    # ---- Pure helpers ---------------------------------------------------

    @classmethod
    def _rewrite(
        cls,
        text: str,
        targets: dict[str, str],
        editable_blocks: set[int] | None = None,
    ) -> tuple[str, list[Drift]]:
        """Return ``(new lock text, drifts)``.

        Only the ``version`` line of a named package's own ``[[package]]``
        block is touched; every other byte is passed through verbatim.

        Args:
            text: Current lockfile contents.
            targets: Versions to write, keyed by package name. Names absent
                from this mapping are left alone.
            editable_blocks: Positions, in file order, of the ``[[package]]``
                blocks that may be rewritten. ``None`` applies no positional
                restriction and matches on name alone — for exercising the
                line mechanics in isolation.
        """
        out: list[str] = []
        drifts: list[LockFile.Drift] = []
        # Name of the block currently open, or None when we are outside a
        # ``[[package]]`` block or have already rewritten its version.
        current: str | None = None
        # Index of the ``[[package]]`` block being read; -1 until the first.
        block = -1
        for line in text.splitlines(keepends=True):
            if line.startswith("["):
                # Any table header — ``[[package]]``, ``[package.metadata]``,
                # ... — closes the previous block's name/version pair.
                current = None
                if line.startswith(cls.PACKAGE_HEADER):
                    block += 1
            elif (match := cls._NAME_RE.match(line)) is not None:
                current = match.group(1)
            elif current is not None and (match := cls._VERSION_RE.match(line)) is not None:
                allowed = editable_blocks is None or block in editable_blocks
                old, new = match.group(1), targets.get(current)
                if allowed and new is not None and new != old:
                    line = f'version = "{new}"\n'
                    drifts.append(cls.Drift(current, old, new))
                current = None
            out.append(line)
        return "".join(out), drifts
