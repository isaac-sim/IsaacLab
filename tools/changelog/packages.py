# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Package manifests, changelog output files, and per-package compilation.

Fragment parsing lives in ``fragments``, version arithmetic in ``versions``,
and PR-gate policy in ``checks``. Original class imports from this module
remain available for compatibility; new callers should import their owners.

The branch-specific version layout is owned by Package.toml_path,
Package.current_version, and Package.write_version. Port all three when
backporting to a branch that stores versions in config/extension.toml."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import tomllib
from fragments import Fragment as Fragment
from fragments import FragmentBatch as FragmentBatch
from fragments import FragmentFilename as FragmentFilename
from paths import PACKAGES_ROOT as PACKAGES_ROOT
from paths import REPO_ROOT as REPO_ROOT
from paths import RepositoryPaths
from versions import Version as Version

if TYPE_CHECKING:
    from checks import PRDiff

__all__ = [
    "REPO_ROOT",
    "PACKAGES_ROOT",
    "Version",
    "FragmentFilename",
    "Fragment",
    "FragmentBatch",
    "Changelog",
    "Package",
    "RootPackage",
    "PRDiff",
]


@dataclass(frozen=True)
class Changelog:
    """A package's ``CHANGELOG.rst`` — the file the compiler prepends to.

    Owns both halves of the header invariant. :data:`HEADER_RE` is the strict
    anchor the compiler needs a match for; :data:`_SELF_HEAL_RE` is the one
    narrow repair applied before checking it. They were separate before, so
    the compiler and the PR gate each carried their own copy of the repair
    and could disagree about which files are acceptable — a fragment passing
    the gate and then wedging the nightly, or the reverse.
    """

    # ---- Class constants ------------------------------------------------

    # A managed package's file must contain at minimum ``Changelog\n---+\n\n``
    # — header, underline, then a blank line — so there is a place to prepend
    # the next version block.
    HEADER_RE: ClassVar[re.Pattern[str]] = re.compile(r"^Changelog\n-+\s*\n\s*\n", re.MULTILINE)

    # A contributor who ships ``Changelog\n---+\n`` (the isaaclab_ppisp shape
    # #5748 introduced) is missing only the trailing blank line. Repair that
    # in memory rather than failing, and write it back so the file ends up
    # canonical on first compile. No-op when the blank line is already there.
    _SELF_HEAL_RE: ClassVar[re.Pattern[str]] = re.compile(r"^(Changelog\n-+)\n(?!\n)", re.MULTILINE)

    # ---- Fields ---------------------------------------------------------

    path: Path

    # ---- Properties -----------------------------------------------------

    @property
    def exists(self) -> bool:
        """Whether the file is present."""
        return self.path.is_file()

    # ---- Public API -----------------------------------------------------

    def normalized(self) -> str:
        """File contents with the missing-blank-line header repaired."""
        return self._SELF_HEAL_RE.sub(r"\1\n\n", self.path.read_text(encoding="utf-8"), count=1)

    def has_valid_header(self) -> bool:
        """Whether a compile could find its insertion point in this file.

        Answers the same question for the PR gate and for the compiler, so
        the two cannot disagree about what "valid" means.
        """
        return self.HEADER_RE.search(self.normalized()) is not None

    def prepend(self, entry: str, *, dry_run: bool) -> list[Path]:
        """Insert ``entry`` directly below the header. Returns paths written.

        Empty in dry-run, so callers get one source of truth for what
        actually changed on disk.
        """
        text = self.normalized()
        m = self.HEADER_RE.search(text)
        if not m:
            raise ValueError(f"Could not locate changelog header in {self.path}")
        updated = text[: m.end()] + entry + "\n" + text[m.end() :]
        if dry_run:
            print(f"\n{'=' * 60}")
            print(f"DRY RUN — would write to {RepositoryPaths.display(self.path)}")
            print(f"{'=' * 60}")
            print(entry)
            return []
        self.path.write_text(updated, encoding="utf-8")
        return [self.path]


@dataclass(frozen=True)
class Package:
    """A source/<pkg>/ directory the changelog tool can manage.

    A package is "managed" if it has both a version metadata file (see
    :attr:`toml_path`, the file the compiler bumps) and a
    ``docs/CHANGELOG.rst`` (the file the compiler updates).
    :meth:`discover` returns only managed packages; instances created
    directly may not be managed (use :attr:`is_managed`).
    """

    class CompileFailed(Exception):
        """A compile that raised *after* it had already written to disk.

        Carries the paths it had written when it failed, so the caller can
        put them back. ``original_contents`` preserves pre-compile bytes,
        including local edits that Git cannot restore. A compile is only meaningful
        whole, and half of one is a changelog entry announcing a version the
        manifest never received, over a fragment that was never consumed —
        which the next run would compile into a second, identical entry.

        Discarding them is also what keeps the working tree clean, which the
        nightly's ``git rebase`` requires. Committing them would satisfy that
        too, which is why the distinction is worth stating: the tree must be
        clean *and* the branch must not carry a half-applied compile.
        """

        def __init__(
            self, cause: Exception, written: list[Path], *, original_contents: dict[Path, bytes] | None = None
        ):
            super().__init__(str(cause))
            self.cause = cause
            self.written = written
            self.original_contents = original_contents

    root: Path

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def changelog(self) -> Changelog:
        """This package's ``CHANGELOG.rst``, as a model rather than a path."""
        return Changelog(self.root / "docs" / "CHANGELOG.rst")

    @property
    def toml_path(self) -> Path:
        return self.root / "pyproject.toml"

    # Name of the per-package fragment directory. Stated once so the model,
    # the PR gate's path matching, and the contributor-facing help all agree.
    FRAGMENT_DIR_NAME: ClassVar[str] = "changelog.d"

    @property
    def default_fragment_dir(self) -> Path:
        return self.root / self.FRAGMENT_DIR_NAME

    @classmethod
    def package_prefix(cls, name: str) -> str:
        """Repo-relative POSIX prefix of a package directory.

        The string form exists because :class:`PRDiff` matches git-diff
        output, which is repo-relative text, while the model holds absolute
        paths. Same location, two representations — derived here rather than
        spelled out at each use.
        """
        return RepositoryPaths.package_prefix(name)

    @classmethod
    def fragment_dir_prefix(cls, name: str) -> str:
        """Repo-relative POSIX prefix of a package's fragment directory."""
        return f"{cls.package_prefix(name)}{cls.FRAGMENT_DIR_NAME}/"

    @property
    def is_managed(self) -> bool:
        return self.toml_path.is_file() and self.changelog.exists

    def current_version(self) -> Version:
        in_project = False
        for line in self.toml_path.read_text(encoding="utf-8").splitlines():
            if re.match(r"^\[project\]", line):
                in_project = True
            elif re.match(r"^\[", line):
                in_project = False
            if in_project:
                m = re.match(r'^version\s*=\s*"([^"]+)"', line)
                if m:
                    return Version(m.group(1))
        raise ValueError(f"{self.name}: no version field found under [project] in {self.toml_path}")

    @classmethod
    def declared_version(cls, root: Path) -> str | None:
        """Return the version the package at ``root`` declares, or ``None`` if unreadable.

        The tolerant counterpart to :meth:`current_version`, for callers
        enumerating directories they do not control — :class:`LockFile`
        walks every uv workspace member, including ones the changelog
        compiler does not manage. A missing or malformed version metadata
        file yields ``None`` instead of raising, so one unmanaged member
        cannot fail an operation that spans the whole workspace.

        Args:
            root: The package directory (``source/<pkg>``).
        """
        pkg = cls(root)
        if not pkg.toml_path.is_file():
            return None
        try:
            return str(pkg.current_version())
        except (OSError, ValueError):
            return None

    def write_changelog_entry(self, entry: str, *, dry_run: bool) -> list[Path]:
        """Prepend ``entry`` to this package's CHANGELOG.rst. Returns paths written."""
        return self.changelog.prepend(entry, dry_run=dry_run)

    def write_version(self, new_version: Version, *, dry_run: bool) -> list[Path]:
        """Set ``version = "<new_version>"`` in this package's version metadata file.

        One of the three branch-layout members (with :attr:`toml_path` and
        :meth:`current_version`); a cherry-pick to another layout must port
        all three. Returns the list of paths written (empty in dry-run) so
        :class:`AutoBumpRun` has a single source of truth for what changed
        on disk.
        """
        text = self.toml_path.read_text(encoding="utf-8")
        in_project = False
        new_lines = []
        for line in text.splitlines(keepends=True):
            if re.match(r"^\[project\]", line):
                in_project = True
            elif re.match(r"^\[", line):
                in_project = False
            if in_project and re.match(r'^version\s*=\s*"[^"]+"', line):
                line = re.sub(r'^(version\s*=\s*)"[^"]+"', f'\\1"{new_version}"', line)
            new_lines.append(line)
        if dry_run:
            print(f'DRY RUN — would set version = "{new_version}" in {RepositoryPaths.display(self.toml_path)}')
            return []
        self.toml_path.write_text("".join(new_lines), encoding="utf-8")
        return [self.toml_path]

    @classmethod
    def from_name(cls, name: str, packages_root: Path = PACKAGES_ROOT) -> Package:
        return cls(packages_root / name)

    @classmethod
    def discover(cls, packages_root: Path = PACKAGES_ROOT) -> list[Package]:
        """Return all managed packages under ``packages_root``, sorted by name."""
        if not packages_root.is_dir():
            return []
        return sorted(
            (cls(child) for child in packages_root.iterdir() if child.is_dir() and cls(child).is_managed),
            key=lambda p: p.name,
        )

    def compile(
        self,
        *,
        fragments_dir: Path | None = None,
        explicit_version: Version | None = None,
        dry_run: bool = False,
    ) -> tuple[bool, list[Path]]:
        """Compile fragments for this package.

        There are exactly two modes: ``dry_run=True`` previews and writes
        nothing; ``dry_run=False`` writes the new entry, bumps the version,
        **and** deletes the consumed fragments. There is deliberately no
        third "write but keep fragments" mode — leaving fragments in place
        after a real compile is a footgun (the next compile would re-emit
        them as a duplicate version block).

        Args:
            fragments_dir: Read fragments from here instead of
                :attr:`default_fragment_dir`. Useful for previewing against
                example fixtures.
            explicit_version: Pin the new version to this string (skips the
                per-fragment bump inference).
            dry_run: Preview only — no files are written or deleted.

        Returns:
            ``(compiled, touched)`` where ``compiled`` is ``True`` if at
            least one fragment was found and processed, and ``touched`` is
            the list of paths actually written to disk (empty in dry-run
            mode). The ``touched`` list is the in-process manifest
            :class:`AutoBumpRun` stages — no out-of-band file or glob
            needed.
        """
        batch = FragmentBatch.from_dir(self._resolve_fragments_dir(fragments_dir))

        # Snapshot the compiler's inputs before mutation, including uncommitted edits.
        original_contents = None
        if not dry_run and (batch.valid or batch.skip_paths):
            paths = [self.changelog.path, self.toml_path, *[f.path for f in batch.valid], *batch.skip_paths]
            original_contents = {path: path.read_bytes() for path in paths if path.is_file()}

        for p in batch.invalid:
            print(
                f"  WARNING: {RepositoryPaths.display(p)} does not match any recognised fragment "
                f"pattern ({FragmentFilename.pattern_summary()}) — skipping.",
                file=sys.stderr,
            )

        if not batch.valid:
            if batch.skip_paths:
                n = len(batch.skip_paths)
                if dry_run:
                    print(f"  {self.name}: would clean {n} stale skip file(s).")
                else:
                    # No entry and no bump, but the skip files are still gone
                    # from the working tree — report them so the deletion is
                    # staged rather than silently reverted on next checkout.
                    print(f"  {self.name}: cleaned {n} stale skip file(s).")
                    try:
                        return False, batch.delete_skips()
                    except FragmentBatch.PartialDeletion as e:
                        # Same contract as the main path: whatever went is
                        # reported, wrapped so the caller can undo it.
                        raise self.CompileFailed(e, e.deleted, original_contents=original_contents) from e
            else:
                print(f"  {self.name}: no fragments, skipping.")
            return False, []

        # Apply the same content-validation rules the PR gate uses, so a
        # malformed fragment that somehow reached this package (e.g. a
        # stale fragment that predates a content-rule tightening, or a
        # locally-edited file) doesn't silently produce a half-empty
        # version block. Runs every fragment that survived filename
        # validation in ``from_dir``.
        validation_errors = [(f, err) for f in batch.valid if (err := f.validate()) is not None]
        if validation_errors:
            for f, err in validation_errors:
                print(f"  ERROR: {RepositoryPaths.display(f.path)}: {err}", file=sys.stderr)
            raise ValueError(
                f"{self.name}: {len(validation_errors)} fragment(s) failed content validation; "
                "fix or remove them before compiling."
            )

        parsed_pairs = batch.parsed
        if not parsed_pairs:
            print(f"  {self.name}: all fragments empty after parsing, skipping.")
            return False, []

        new_version, bump_label, entry = batch.compile_to_entry(
            self.current_version(), explicit_version=explicit_version
        )
        print(f"  {self.name}: {len(parsed_pairs)} fragment(s) → version {new_version}{bump_label}")

        if not self.changelog.exists:
            # Should never happen with managed packages discovered via
            # ``Package.discover()`` — defensive check for callers that
            # construct a ``Package`` directly with an unmanaged root.
            raise ValueError(
                f"{RepositoryPaths.display(self.changelog.path)} does not exist; "
                f"package {self.name!r} is not managed (missing CHANGELOG.rst)."
            )
        # Everything below mutates the working tree, so from the first write
        # on the compile is no longer all-or-nothing. A later failure reports
        # what it managed to write so the caller can undo it -- see
        # :class:`CompileFailed` for why undoing beats keeping.
        touched: list[Path] = []
        try:
            touched.extend(self.write_changelog_entry(entry, dry_run=dry_run))
            touched.extend(self.write_version(new_version, dry_run=dry_run))

            if not dry_run:
                deleted_frags, deleted_skips = batch.delete_all()
                # Deletions are part of the change set: they must be staged
                # with the entry that consumed them, or the fragments come
                # back on the next checkout and recompile into a duplicate
                # version block.
                touched.extend(deleted_frags)
                touched.extend(deleted_skips)
                msg = f"  {self.name}: deleted {len(deleted_frags)} fragment(s)"
                if deleted_skips:
                    msg += f" and {len(deleted_skips)} skip file(s)"
                print(msg + ".")
        except FragmentBatch.PartialDeletion as e:
            # Deletions that did land are changes like any other, so they
            # join the set the caller has to undo.
            touched.extend(e.deleted)
            raise self.CompileFailed(e, touched, original_contents=original_contents) from e
        except (OSError, ValueError) as e:
            if not touched:
                # Nothing reached disk, so there is nothing to undo and the
                # original exception type is the more useful one. Only a
                # genuinely half-applied compile needs the wrapper.
                raise
            raise self.CompileFailed(e, touched, original_contents=original_contents) from e

        return True, touched

    def _resolve_fragments_dir(self, override: Path | None) -> Path:
        """Pick the directory ``compile`` should read fragments from.

        ``None`` means "use this package's own ``changelog.d/``"; an
        absolute path is used as-is; a relative path is resolved against
        ``REPO_ROOT`` so callers can pass things like
        ``tools/changelog/test/integration/01_patch_bump/fragments`` without
        worrying about the cwd.
        """
        if override is None:
            return self.default_fragment_dir
        return override if override.is_absolute() else (REPO_ROOT / override).resolve()


@dataclass(frozen=True)
class RootPackage:
    """The repo-root ``pyproject.toml`` — the uv workspace declaration.

    :class:`Package` models a *member's* manifest; this models the root one.
    Membership is declared here, so this is what a lockfile is validated
    against: without it the root manifest would have no owner and every
    caller needing the member list would parse it inline.
    """

    # ---- Fields ---------------------------------------------------------

    root: Path

    # ---- Properties -----------------------------------------------------

    @property
    def path(self) -> Path:
        """Absolute path to the root manifest (which may not exist)."""
        return self.root / "pyproject.toml"

    @property
    def exists(self) -> bool:
        """Whether this branch carries a root manifest at all."""
        return self.path.is_file()

    # ---- Public API -----------------------------------------------------

    def declared_members(self) -> set[Path]:
        """Return the package roots declared as editable workspace members.

        ``[tool.uv.sources]`` is the authoritative member list — not
        ``source/*/``, which also holds directories uv does not track, and
        not :meth:`Package.discover`, which filters to packages the changelog
        compiler manages (``isaaclab_tasks_experimental`` is a workspace
        member with no ``CHANGELOG.rst``).
        """
        data = tomllib.loads(self.path.read_text(encoding="utf-8"))
        sources = data.get("tool", {}).get("uv", {}).get("sources", {})
        return {
            self.root / spec["path"] for spec in sources.values() if isinstance(spec, dict) and spec.get("editable")
        }


def __getattr__(name: str) -> type[PRDiff]:
    """Preserve the original PRDiff import without a packages/checks import cycle."""
    if name == "PRDiff":
        return import_module("checks").PRDiff
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
