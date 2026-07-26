# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The changelog domain model: versions, fragments, packages, PR diffs.

The bottom layer of the changelog tool. Everything here answers questions
about what is on disk -- what version a package declares, which fragments
are pending, what a compiled entry looks like -- and knows nothing about
git, the nightly job, or the command line.

Which file holds a package's version is a property of the branch, not of
this module's callers: :attr:`Package.toml_path` resolves it, so the same
code is correct on branches that keep versions in ``pyproject.toml`` and on
release branches that still keep them in ``config/extension.toml``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property
from pathlib import Path
from typing import ClassVar

# Walk three levels up: tools/changelog/packages.py -> tools/changelog/ -> tools/ -> repo root.
REPO_ROOT = Path(__file__).parent.parent.parent
PACKAGES_ROOT = REPO_ROOT / "source"

# Recognised fragment filename patterns. ``<slug>`` is any short identifier
# the contributor chose — typically their branch name with ``/`` replaced by
# ``-``. The slug must not contain ``.`` (reserved for the tier suffix) or
# ``/`` (path separator), but otherwise mirrors what git allows in a ref name.
# These regexes live at module level because Fragment, FragmentBatch, and
# PRDiff all match against them — they are the wire-format contract between
# contributors and the gate.
FRAGMENT_RE = re.compile(r"^(?P<slug>[^./][^./]*)(?:\.(?P<bump>minor|major))?\.rst$")
SKIP_RE = re.compile(r"^(?P<slug>[^./][^./]*)\.skip$")

# Anchor the compile-time insertion point in ``CHANGELOG.rst``. A managed
# package's file must contain at minimum ``Changelog\n---+\n\n`` — header,
# underline, then a blank line — so the bot has a place to prepend the next
# version block. Imported by both ``Package.write_changelog_entry`` (the
# producer) and ``test_validate`` (the regression gate) so the two cannot
# drift on what "valid header" means.
CHANGELOG_HEADER_RE = re.compile(r"^Changelog\n-+\s*\n\s*\n", re.MULTILINE)


def _display_path(p: Path) -> str:
    """Pretty-print a Path. Strips ``REPO_ROOT`` if ``p`` is inside the repo,
    falls back to the absolute path otherwise (``--fragments-dir`` may
    legitimately point at an external directory like ``/tmp/...``).

    Lives at module level because both :class:`Package` (writing on-disk
    paths) and :class:`FragmentBatch` (warning about external fragment
    paths) use it.
    """
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Version:
    """A semver-style version string ``X.Y.Z`` (optionally suffixed with ``.devN``).

    Models a version as a value object: immutable, comparable by its text,
    knows how to produce a bumped successor. PEP 440 ``.devN`` suffixes
    are tolerated on the way *in* (stripped before bumping) but never
    written back out — :meth:`bumped` always returns a clean ``X.Y.Z``.

    Construction validates the format up front so that an invalid
    ``--version`` flag from the CLI fails fast instead of silently writing
    a malformed entry to ``CHANGELOG.rst``.
    """

    # ``X.Y.Z`` with an optional PEP 440 ``.devN`` suffix. The suffix is
    # tolerated on the way *in* (e.g. when reading a stale dev version out
    # of an existing version metadata file) but :meth:`bumped` always strips
    # it before producing a successor.
    _SEMVER_RE: ClassVar[re.Pattern[str]] = re.compile(r"^\d+\.\d+\.\d+(\.dev\d+)?$")

    text: str

    def __post_init__(self) -> None:
        if not self._SEMVER_RE.match(self.text):
            raise ValueError(f"Invalid version {self.text!r}; expected X.Y.Z (optionally suffixed with .devN)")

    def bumped(self, tier: str) -> Version:
        """Return a new Version one tier ahead of this one.

        ``tier`` is ``'major'``, ``'minor'``, or ``'patch'``. Major zeros
        the minor and patch components; minor zeros patch. Any ``.devN``
        suffix on the current version is stripped before bumping.
        """
        # __post_init__ guarantees the format, so this split is safe.
        parts = self.text.split(".dev")[0].split(".")
        if tier == "major":
            return Version(f"{int(parts[0]) + 1}.0.0")
        if tier == "minor":
            return Version(f"{parts[0]}.{int(parts[1]) + 1}.0")
        return Version(f"{parts[0]}.{parts[1]}.{int(parts[2]) + 1}")

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class Fragment:
    """One fragment file in a package's ``changelog.d/`` (or an examples dir).

    A :class:`Fragment` instance is just a path plus methods that interpret
    it as a changelog fragment. ``.gitkeep`` and ``*.skip`` files should
    not be wrapped — only files matching :data:`FRAGMENT_RE`.
    """

    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    @cached_property
    def _match(self) -> re.Match[str] | None:
        return FRAGMENT_RE.match(self.name)

    @property
    def is_valid_filename(self) -> bool:
        return self._match is not None

    @property
    def bump(self) -> str:
        """Bump tier declared by the filename suffix (defaults to ``'patch'``)."""
        if self._match and self._match.group("bump"):
            return self._match.group("bump")
        return "patch"

    def parse(self) -> dict[str, list[str]]:
        """Return ``{section: [lines]}`` from this fragment's content.

        Lines are kept as-is (including trailing newlines) so the compiled
        output is byte-for-byte identical to what the contributor wrote. A
        section heading is a non-empty line followed by ``^`` underline of
        equal-or-greater length.
        """
        text = self.path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        sections: dict[str, list[str]] = {}
        current: str | None = None
        buf: list[str] = []

        i = 0
        while i < len(lines):
            raw = lines[i]
            stripped = raw.rstrip("\n")
            if (
                i + 1 < len(lines)
                and stripped
                and re.fullmatch(r"\^+", lines[i + 1].rstrip("\n"))
                and len(lines[i + 1].rstrip("\n")) >= len(stripped)
            ):
                if current is not None:
                    sections[current] = self._strip_trailing_blank(buf)
                current = stripped
                buf = []
                i += 2  # skip heading + underline
                if i < len(lines) and not lines[i].strip():
                    i += 1
                continue
            if current is not None:
                buf.append(raw)
            i += 1

        if current is not None:
            sections[current] = self._strip_trailing_blank(buf)

        return sections

    @staticmethod
    def _strip_trailing_blank(lines: list[str]) -> list[str]:
        """Drop trailing blank lines from a section's raw line buffer."""
        while lines and not lines[-1].strip():
            lines.pop()
        return lines

    @staticmethod
    def parse_slug(filename: str) -> str | None:
        """Return the slug declared by a fragment / skip filename, or ``None``.

        Used by :class:`PRDiff` to detect collisions between an added
        fragment's slug and an existing fragment in the same directory,
        without needing to materialise a :class:`Fragment` (the diff entry
        may not exist on disk yet during a gate run).
        """
        m = FRAGMENT_RE.match(filename) or SKIP_RE.match(filename)
        return m.group("slug") if m else None

    def merge_time(self) -> int:
        """Unix timestamp of the merge commit that introduced this fragment.

        Uses ``git log --diff-filter=A --first-parent`` to follow develop's
        first-parent history, so the timestamp reflects when the PR's merge
        commit landed (not the feature-branch commit that originally added
        the file). Falls back to the file's most recent commit time when
        not yet in first-parent history (e.g. local dry-runs on a feature
        branch), and ultimately to ``0`` if git is unavailable.
        """
        for cmd in (
            ["git", "log", "--diff-filter=A", "--first-parent", "-1", "--format=%ct", "--", str(self.path)],
            ["git", "log", "-1", "--format=%ct", "--", str(self.path)],
        ):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=REPO_ROOT)
                ts = result.stdout.strip()
                if ts:
                    return int(ts)
            except (subprocess.CalledProcessError, ValueError):
                continue
        return 0

    def validate(self) -> str | None:
        """Return a human-readable error string if malformed, else ``None``.

        Filename rules: must match :data:`FRAGMENT_RE` (``.gitkeep`` and
        ``*.skip`` files are filtered out at :meth:`FragmentBatch.from_dir`
        level and never reach this method). Content rules (for ``*.rst``
        fragments only): non-empty file with at least one valid section
        heading and at least one bullet point.
        """
        if not self.is_valid_filename:
            return (
                "invalid filename — must be <slug>.rst, <slug>.minor.rst, "
                "<slug>.major.rst, or <slug>.skip (slug = your branch name "
                "with `/` replaced by `-`, no dots)"
            )
        if not self.path.exists():
            # Deleted fragments don't need validating (consumed by a previous compile).
            return None
        text = self.path.read_text(encoding="utf-8")
        if not text.strip():
            return "fragment is empty"
        sections = self.parse()
        if not sections:
            return (
                "no recognised section headings (expected one or more of "
                "Added / Changed / Deprecated / Removed / Fixed, each underlined "
                "with carets ``^`` of equal-or-greater length)"
            )
        # Every declared section must carry at least one bullet — otherwise
        # the compiled output emits a heading with no body, which is both
        # ugly and almost certainly a contributor authoring mistake (typed
        # the heading, forgot the bullet).
        empty = [s for s, lines in sections.items() if not any(line.lstrip().startswith("*") for line in lines)]
        if empty:
            return (
                f"section(s) {', '.join(repr(s) for s in empty)} have no bullet entries — "
                "use ``* `` to start each entry, or remove the heading"
            )
        # Every line inside a section body must be a bullet (``* ``), a
        # continuation (leading whitespace), or blank. A column-0 non-blank
        # line that isn't a bullet terminates the list under RST rules and
        # then sits as a paragraph adjacent to the next ``* `` — which the
        # compile step splices into ``CHANGELOG.rst`` under the same
        # ``^^^`` subheading and Sphinx then rejects with
        # ``Unexpected indentation``. Catch it here before merge.
        for section, lines in sections.items():
            for offset, line in enumerate(lines):
                if not line.strip():
                    continue
                if line[0].isspace() or line.lstrip().startswith("*"):
                    continue
                snippet = line.strip()[:80]
                return (
                    f"section {section!r} contains an orphan paragraph "
                    f"(non-bullet line {offset + 1}: {snippet!r}). Every line under "
                    "a section heading must start with ``* `` (new bullet) or whitespace "
                    "(continuation of the previous bullet). A flush-left paragraph here "
                    "splits the bullet list and Sphinx fails the doc build with "
                    "``Unexpected indentation``."
                )
        return None


@dataclass(frozen=True)
class FragmentBatch:
    """A collection of fragments collected from a directory.

    ``valid`` are :class:`Fragment` instances sorted by merge time
    (oldest first). ``invalid`` are paths that don't match any recognised
    filename pattern — surfaced so the caller can warn or fail. ``.skip``
    and ``.gitkeep`` files are tolerated but excluded from both lists.

    Holds the pure-data class methods that turn a batch (or a synthetic
    list of bumps / sections) into a compiled changelog entry. The
    instance methods (:meth:`aggregate_bump`, :meth:`merged_sections`,
    :meth:`compile_to_entry`) read the batch's own state; the
    underscore-prefixed static methods (:meth:`_aggregate`, etc.) are
    the underlying pure transformations and are used directly by tests
    that exercise edge cases without a real fragments directory.
    """

    # Canonical ordering of section headings in compiled output. Anything
    # not listed here keeps insertion order *after* these.
    _SECTION_ORDER: ClassVar[list[str]] = ["Added", "Changed", "Deprecated", "Removed", "Fixed"]

    # Strict ordering of bump tiers (``major`` strictly outranks ``minor``
    # outranks ``patch``). Unrecognised tiers sort below ``patch``.
    _BUMP_RANK: ClassVar[dict[str, int]] = {"patch": 0, "minor": 1, "major": 2}

    valid: list[Fragment]
    invalid: list[Path]
    skip_paths: list[Path] = field(default_factory=list)

    # ---- Construction --------------------------------------------------

    @classmethod
    def from_dir(cls, fragment_dir: Path) -> FragmentBatch:
        if not fragment_dir.is_dir():
            return cls([], [])
        valid: list[Fragment] = []
        invalid: list[Path] = []
        skips: list[Path] = []
        for p in fragment_dir.iterdir():
            if p.is_dir() or p.name == ".gitkeep":
                continue
            if SKIP_RE.match(p.name):
                skips.append(p)
                continue
            f = Fragment(p)
            if f.is_valid_filename:
                valid.append(f)
            else:
                invalid.append(p)
        # Sort by merge time, breaking ties on filename so the compiled output
        # is deterministic when fragments share a merge commit (or when none
        # are in git history yet — e.g. a local dry-run against test fixtures).
        valid.sort(key=lambda f: (f.merge_time(), f.name))
        return cls(valid, invalid, skips)

    # ---- Queries against this batch's state ---------------------------

    def aggregate_bump(self) -> str:
        """Highest bump tier declared by fragments that parsed to content.

        Empty fragments (which the compiler warns about and skips) are
        excluded so they don't influence the version. Defaults to
        ``patch`` if nothing parsed.
        """
        return self._aggregate([f.bump for f, _ in self.parsed()])

    def parsed(self) -> list[tuple[Fragment, dict[str, list[str]]]]:
        """Return ``(fragment, sections)`` pairs, dropping fragments that parse empty."""
        return [(f, s) for f, s in ((f, f.parse()) for f in self.valid) if s]

    def merged_sections(self) -> dict[str, list[str]]:
        """Cross-fragment merged section map for this batch."""
        return self._merge_sections([s for _, s in self.parsed()])

    def compile_to_entry(
        self,
        current_version: Version,
        *,
        explicit_version: Version | None = None,
    ) -> tuple[Version, str, str]:
        """Return ``(new_version, bump_label, entry_text)`` for this batch.

        ``new_version`` is either ``explicit_version`` verbatim or the
        result of bumping ``current_version`` by the aggregated tier.
        ``bump_label`` is a human-readable suffix like ``" (bump: minor)"``
        for log lines (empty when ``explicit_version`` is used).
        ``entry_text`` is the rendered RST block ready to prepend to a
        ``CHANGELOG.rst``. Pure computation — no I/O.
        """
        if explicit_version is not None:
            new_version = explicit_version
            bump_label = ""
        else:
            chosen_bump = self.aggregate_bump()
            new_version = current_version.bumped(chosen_bump)
            bump_label = f" (bump: {chosen_bump})"
        entry = self._format_entry(new_version.text, self.merged_sections())
        return new_version, bump_label, entry

    # ---- Cleanup -------------------------------------------------------

    # These return the paths they removed, not just counts. A deletion is a
    # change to the working tree exactly like a write is, and the nightly
    # auto-commit stages what the compile reports it changed — so a consumed
    # fragment that vanishes from disk without appearing in that report would
    # never be staged, survive on the branch, and recompile the next night
    # into a duplicate entry and a second version bump.

    def delete_all(self) -> tuple[list[Path], list[Path]]:
        """Delete every consumed fragment + skip file. Returns ``(fragments, skips)`` deleted."""
        return self.delete_valid(), self.delete_skips()

    def delete_valid(self) -> list[Path]:
        """Delete the consumed fragments. Returns the paths removed."""
        deleted = [f.path for f in self.valid]
        for path in deleted:
            path.unlink()
        return deleted

    def delete_skips(self) -> list[Path]:
        """Delete the ``.skip`` files. Returns the paths removed."""
        deleted = list(self.skip_paths)
        for path in deleted:
            path.unlink()
        return deleted

    # ---- Pure transformations (the data class methods) ----------------
    # Static so callers and tests can exercise them with synthetic
    # primitives — no FragmentBatch instance needed when the question
    # is "given these tiers, which wins?" or "how do these dicts merge?"

    @classmethod
    def _aggregate(cls, bumps: list[str]) -> str:
        """Highest-ranked bump from ``bumps`` (``major > minor > patch``).

        An empty list defaults to ``'patch'``.
        """
        if not bumps:
            return "patch"
        return max(bumps, key=lambda b: cls._BUMP_RANK.get(b, -1))

    @staticmethod
    def _merge_sections(fragments: list[dict[str, list[str]]]) -> dict[str, list[str]]:
        """Merge multiple parsed fragments into a single section map.

        Bullets from different fragments that share a section heading are
        concatenated directly (no blank line between them) to match the
        dominant style in IsaacLab's existing ``CHANGELOG.rst`` files.
        """
        merged: dict[str, list[str]] = {}
        for frag in fragments:
            for section, lines in frag.items():
                if section not in merged:
                    merged[section] = list(lines)
                else:
                    merged[section].extend(lines)
        return merged

    @classmethod
    def _format_entry(cls, version: str, sections: dict[str, list[str]]) -> str:
        """Return a complete RST version entry, ready to prepend to ``CHANGELOG.rst``.

        Sections appear in :attr:`_SECTION_ORDER` (Added, Changed,
        Deprecated, Removed, Fixed). Anything else keeps insertion order
        *after* the canonical ones.
        """
        today = date.today().strftime("%Y-%m-%d")
        heading = f"{version} ({today})"
        out = [heading, "~" * len(heading), ""]

        ordered = [s for s in cls._SECTION_ORDER if s in sections]
        extras = [s for s in sections if s not in cls._SECTION_ORDER]

        for section in ordered + extras:
            out.append(section)
            out.append("^" * len(section))
            out.append("")
            for line in sections[section]:
                out.append(line.rstrip("\n"))
            out.append("")

        return "\n".join(out) + "\n"


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

        Carries the paths written before the failure. A caller that stages
        what the compile reports must still receive them: the writes really
        happened, and leaving them unstaged strands a half-applied compile in
        the working tree, which in turn makes the nightly's ``git rebase``
        refuse to run.
        """

        def __init__(self, cause: Exception, touched: list[Path]):
            super().__init__(str(cause))
            self.cause = cause
            self.touched = touched

    root: Path

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def changelog_path(self) -> Path:
        return self.root / "docs" / "CHANGELOG.rst"

    @property
    def toml_path(self) -> Path:
        return self.root / "pyproject.toml"

    @property
    def default_fragment_dir(self) -> Path:
        return self.root / "changelog.d"

    @property
    def is_managed(self) -> bool:
        return self.toml_path.is_file() and self.changelog_path.is_file()

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
        """Prepend ``entry`` to this package's CHANGELOG.rst. Returns the
        list of paths written (empty in dry-run, so callers like
        :class:`AutoBumpRun` get a single source of truth for "what just
        changed on disk")."""
        text = self.changelog_path.read_text(encoding="utf-8")
        # Self-heal a header that lacks the trailing blank line. The compile
        # regex needs ``Changelog\n---+\n\n`` as an anchor; a contributor who
        # ships ``Changelog\n---+\n`` (the isaaclab_ppisp shape PR #5748
        # introduced) would otherwise wedge the nightly. Insert the missing
        # ``\n`` in-memory and write it back so the on-disk file ends up
        # canonical on first compile. No-op when the blank line is already
        # there (negative lookahead).
        text = re.sub(r"^(Changelog\n-+)\n(?!\n)", r"\1\n\n", text, count=1, flags=re.MULTILINE)
        m = CHANGELOG_HEADER_RE.search(text)
        if not m:
            raise ValueError(f"Could not locate changelog header in {self.changelog_path}")
        updated = text[: m.end()] + entry + "\n" + text[m.end() :]
        if dry_run:
            print(f"\n{'=' * 60}")
            print(f"DRY RUN — would write to {_display_path(self.changelog_path)}")
            print(f"{'=' * 60}")
            print(entry)
            return []
        self.changelog_path.write_text(updated, encoding="utf-8")
        return [self.changelog_path]

    def write_version(self, new_version: Version, *, dry_run: bool) -> list[Path]:
        """Set ``version = "<new_version>"`` in this package's version metadata file.

        Which file that is depends on the branch layout — :attr:`toml_path`
        resolves it (``pyproject.toml`` here, ``config/extension.toml`` on
        release branches that predate #6505). Returns the list of paths
        written (empty in dry-run) so :class:`AutoBumpRun` has a single
        source of truth for what changed on disk.
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
            print(f'DRY RUN — would set version = "{new_version}" in {_display_path(self.toml_path)}')
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

        for p in batch.invalid:
            print(
                f"  WARNING: {_display_path(p)} does not match any recognised fragment "
                "pattern (<slug>.rst, <slug>.minor.rst, <slug>.major.rst, <slug>.skip) — skipping.",
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
                    return False, batch.delete_skips()
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
                print(f"  ERROR: {_display_path(f.path)}: {err}", file=sys.stderr)
            raise ValueError(
                f"{self.name}: {len(validation_errors)} fragment(s) failed content validation; "
                "fix or remove them before compiling."
            )

        parsed_pairs = batch.parsed()
        if not parsed_pairs:
            print(f"  {self.name}: all fragments empty after parsing, skipping.")
            return False, []

        new_version, bump_label, entry = batch.compile_to_entry(
            self.current_version(), explicit_version=explicit_version
        )
        print(f"  {self.name}: {len(parsed_pairs)} fragment(s) → version {new_version}{bump_label}")

        if not self.changelog_path.exists():
            # Should never happen with managed packages discovered via
            # ``Package.discover()`` — defensive check for callers that
            # construct a ``Package`` directly with an unmanaged root.
            raise ValueError(
                f"{_display_path(self.changelog_path)} does not exist; "
                f"package {self.name!r} is not managed (missing CHANGELOG.rst)."
            )
        # Everything below mutates the working tree. Once the first write
        # lands the run is no longer all-or-nothing, so a later failure has to
        # hand back what it already did rather than let those paths vanish
        # from the caller's staging set.
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
        except (OSError, ValueError) as e:
            if not touched:
                # Nothing reached disk, so the caller has nothing to stage and
                # the original exception type is the more useful one. Only a
                # genuinely half-applied compile needs the wrapper.
                raise
            raise self.CompileFailed(e, touched) from e

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
class PRDiff:
    """A snapshot of "what this PR changed against its base branch."

    Wraps two views from the same git diff: ``changed`` (any file modified
    or added) and ``added`` (the strict subset that's new on this branch).
    Tests construct ``PRDiff`` directly with synthetic sets;
    :meth:`from_git` runs the real ``git diff`` for production use.
    """

    changed: set[str]
    added: set[str]

    @classmethod
    def from_git(cls, base_ref: str) -> PRDiff:
        """Run ``git diff`` against ``origin/<base_ref>...HEAD`` to populate the diff."""

        def _diff(extra_args: list[str]) -> set[str]:
            result = subprocess.run(
                ["git", "diff", "--name-only", *extra_args, f"origin/{base_ref}...HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=REPO_ROOT,
            )
            return {f for f in result.stdout.splitlines() if f}

        return cls(changed=_diff([]), added=_diff(["--diff-filter=A"]))

    def evaluate(
        self,
        packages: list[Package],
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """Apply the PR-gate rules and return ``(missing_packages, invalid_fragments)``.

        Rules:

        1. **Immutability** — every fragment file in the diff must be in
           ``added`` (added on this branch). Modifying or renaming an existing
           fragment is rejected with a hint to add a new one instead.

        2. **Content validity** — every added ``*.rst`` fragment must parse
           (recognised section headings + at least one bullet). ``.skip`` and
           ``.gitkeep`` are exempt.

        3. **Slug uniqueness** — within a package's ``changelog.d/``, no two
           fragments may share the same slug. If an added fragment's slug
           collides with an existing or co-added fragment, fail with a hint
           to rename (e.g. append ``-2``).

        4. **Required fragment per touched package** — for each managed
           package the PR touches in ``source/`` (outside ``changelog.d/``),
           the PR must *add* at least one valid fragment to that package's
           ``changelog.d/``. Chained PRs (parent PR's fragment shows up in
           the child's diff) naturally satisfy this — slug uniqueness is
           the only constraint that matters.
        """
        missing: list[str] = []
        invalid_fragments: list[tuple[str, str]] = []

        for pkg in packages:
            pkg_prefix = f"source/{pkg.name}/"
            changelog_dir = f"source/{pkg.name}/changelog.d/"

            source_changed = [f for f in self.changed if f.startswith(pkg_prefix) and not f.startswith(changelog_dir)]
            fragment_changes = [f for f in self.changed if f.startswith(changelog_dir)]

            # Map *pre-existing* fragments in the package's changelog.d/ by slug,
            # for the uniqueness check below. The CI checkout contains both
            # base-branch fragments and the PR's additions side by side, so we
            # must explicitly exclude added files — otherwise an added file can
            # overwrite the entry for a colliding pre-existing fragment with
            # the same slug, hiding the very collision we're trying to detect.
            # Skip ``.gitkeep`` and unrecognised filenames — they can't collide.
            added_basenames = {Path(f).name for f in self.added if f.startswith(changelog_dir)}
            existing_slugs: dict[str, str] = {}
            existing_dir = pkg.default_fragment_dir
            if existing_dir.is_dir():
                for p in existing_dir.iterdir():
                    if p.is_dir() or p.name == ".gitkeep" or p.name in added_basenames:
                        continue
                    slug = Fragment.parse_slug(p.name)
                    if slug is not None:
                        existing_slugs[slug] = p.name

            added_slugs: dict[str, str] = {}
            for f in fragment_changes:
                path = Path(f)
                if path.name == ".gitkeep":
                    continue

                # Rule 1: immutability — modifying an existing fragment is forbidden.
                if f not in self.added:
                    invalid_fragments.append(
                        (
                            f,
                            "fragments are immutable — add a new fragment with a different slug "
                            "instead of editing an existing one",
                        )
                    )
                    continue

                # Rule 2: content validity (only for *.rst, not *.skip).
                if not SKIP_RE.match(path.name):
                    err = Fragment(REPO_ROOT / f).validate()
                    if err:
                        invalid_fragments.append((f, err))
                        continue

                # Rule 3: slug uniqueness within the package's changelog.d/.
                slug = Fragment.parse_slug(path.name)
                if slug is None:
                    # Filename validation already flagged this above for *.rst,
                    # but a malformed *.skip would slip through. Surface it.
                    invalid_fragments.append(
                        (f, "invalid filename — must be <slug>.rst, <slug>.minor.rst, <slug>.major.rst, or <slug>.skip")
                    )
                    continue
                if slug in existing_slugs and existing_slugs[slug] != path.name:
                    invalid_fragments.append(
                        (
                            f,
                            f"slug {slug!r} collides with existing fragment "
                            f"{existing_slugs[slug]!r} — rename to {slug}-2 (or any unused slug)",
                        )
                    )
                    continue
                if slug in added_slugs and added_slugs[slug] != path.name:
                    invalid_fragments.append(
                        (
                            f,
                            f"slug {slug!r} collides with another added fragment "
                            f"{added_slugs[slug]!r} — rename one to {slug}-2 (or any unused slug)",
                        )
                    )
                    continue
                added_slugs[slug] = path.name

            if not source_changed:
                continue

            # Rule 4: this PR must add at least one valid fragment for the package.
            owned = [
                f
                for f in fragment_changes
                if f in self.added and (FRAGMENT_RE.match(Path(f).name) or SKIP_RE.match(Path(f).name))
            ]
            if not owned:
                missing.append(pkg.name)

        return missing, invalid_fragments
