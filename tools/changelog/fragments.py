# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Fragment filenames, content validation, and batch compilation."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property
from pathlib import Path
from typing import ClassVar

from paths import REPO_ROOT, RepositoryPaths
from versions import Version


@dataclass(frozen=True)
class FragmentFilename:
    """A fragment's filename parsed into ``(slug, tier)``.

    Wire-format contract between contributors and the gate. Three classes
    interpret a filename — :class:`Fragment` (instance, has the file on disk),
    :class:`FragmentBatch` (directory walk, filters out skips), and
    :class:`PRDiff` (gate, may see paths that don't exist on disk yet) — and
    they all need to agree on what counts as a fragment, what tier it
    declares, and what slug it owns. Centralising that logic on a value object
    keeps the three in lockstep without forcing every caller to materialise a
    :class:`Fragment`.

    Suffix matching is anchored from the right with the longest suffix winning
    (``foo.minor.rst`` is ``foo`` + minor, never ``foo.minor`` + patch).

    Slug rules are ``git check-ref-format --branch`` minus ``/`` — exactly the
    names git lets you give a branch, which is the point: the convention is
    "slug = your branch name". So dots are fine
    (``bump-newton-1.2.0rc2.minor.rst``), a leading ``-`` is not, and ``/``
    becomes ``-`` because a slug is a filename.
    ``test_slug_rules_track_git_branch_rules`` pins the equivalence.
    """

    # Recognised filename suffixes, longest first. Exposed as a class
    # attribute so tests and the contributor-facing error message can refer to
    # the canonical list without re-stating it.
    SUFFIXES: ClassVar[tuple[tuple[str, str], ...]] = (
        (".minor.rst", "minor"),
        (".major.rst", "major"),
        (".skip", "skip"),
        (".rst", "patch"),
    )

    # Housekeeping files that live in a fragment directory but are not
    # fragments. Stated once: the directory walker and the PR gate both ask,
    # and a second sentinel added to one and not the other would make them
    # disagree about what a directory contains.
    IGNORED_NAMES: ClassVar[frozenset[str]] = frozenset({".gitkeep"})

    # Chars forbidden inside a slug, mirroring ``git check-ref-format``.
    _FORBIDDEN_CHARS: ClassVar[frozenset[str]] = frozenset(" ~^:?*[\\\x7f")

    name: str

    @cached_property
    def _parsed(self) -> tuple[str, str] | None:
        for suffix, tier in self.SUFFIXES:
            if not self.name.endswith(suffix):
                continue
            slug = self.name[: -len(suffix)]
            if not self._slug_is_valid(slug):
                return None
            return slug, tier
        return None

    @classmethod
    def _slug_is_valid(cls, slug: str) -> bool:
        """``True`` if ``slug`` satisfies the git-refname-minus-``/`` rules."""
        if not slug:
            return False
        if slug[0] in "-." or slug[-1] == ".":
            return False
        if slug.endswith(".lock") or ".." in slug or "@{" in slug:
            return False
        return not any(c in cls._FORBIDDEN_CHARS or ord(c) < 32 or c == "/" for c in slug)

    @property
    def is_ignorable(self) -> bool:
        """``True`` for housekeeping files that are not fragments by design."""
        return self.name in self.IGNORED_NAMES

    @property
    def is_valid(self) -> bool:
        """``True`` if the filename parses as either a fragment or a skip marker."""
        return self._parsed is not None

    @property
    def is_fragment(self) -> bool:
        """``True`` if the filename declares an ``.rst`` fragment (not a skip)."""
        return self._parsed is not None and self._parsed[1] != "skip"

    @property
    def is_skip(self) -> bool:
        """``True`` if the filename is a ``.skip`` marker."""
        return self._parsed is not None and self._parsed[1] == "skip"

    @property
    def slug(self) -> str | None:
        """Slug component, or ``None`` if the filename does not parse."""
        return self._parsed[0] if self._parsed is not None else None

    @property
    def tier(self) -> str | None:
        """Bump tier (``patch`` / ``minor`` / ``major`` / ``skip``), or ``None``."""
        return self._parsed[1] if self._parsed is not None else None

    # ---- User-facing pattern descriptions (derived from SUFFIXES) ---------

    # Display order for help / error messages. The parser order in
    # :attr:`SUFFIXES` is "longest suffix first" (semantically required), but
    # readers prefer "tiers ascending" (patch → minor → major → skip).
    _DISPLAY_ORDER: ClassVar[tuple[str, ...]] = ("patch", "minor", "major", "skip")

    @classmethod
    def pattern_summary(cls) -> str:
        """Comma-separated list of accepted patterns: ``<slug>.rst, ..., or <slug>.skip``.

        Single source of truth for the user-facing pattern list. Derived from
        :attr:`SUFFIXES` so that adding a tier updates every error message
        and help block at once.
        """
        by_tier = {tier: suffix for suffix, tier in cls.SUFFIXES}
        parts = [f"<slug>{by_tier[t]}" for t in cls._DISPLAY_ORDER if t in by_tier]
        return ", ".join(parts[:-1]) + f", or {parts[-1]}"

    @classmethod
    def help_lines_for_package(cls, package_name: str) -> list[str]:
        """Per-tier help lines used when a package is missing a fragment.

        Returns one ``add ...`` / ``or ...`` line per tier, formatted with
        the path under the package's ``changelog.d/`` directory and an inline
        annotation describing the bump.
        """
        return cls.help_lines(RepositoryPaths.fragment_dir_prefix(package_name))

    @classmethod
    def help_lines(cls, fragment_dir: str) -> list[str]:
        """Format per-tier help for a fragment directory prefix ending in ``/``."""
        annotations = {
            "patch": "(patch bump)",
            "minor": "(minor bump)",
            "major": "(major bump)",
            "skip": "(no entry, no bump)",
        }
        by_tier = {tier: suffix for suffix, tier in cls.SUFFIXES}
        # Pad the suffix column so the annotations line up regardless of tier
        # length — purely cosmetic, but the existing CI output already aligns.
        suffix_width = max(len(s) for s in by_tier.values())
        lines: list[str] = []
        for i, t in enumerate(cls._DISPLAY_ORDER):
            if t not in by_tier:
                continue
            verb = "add " if i == 0 else "or  "
            path = f"{fragment_dir}<slug>{by_tier[t]}"
            padding = " " * (suffix_width - len(by_tier[t]))
            lines.append(f"{verb} {path}{padding}   {annotations[t]}")
        return lines


@dataclass(frozen=True)
class Fragment:
    """One fragment file in a package's ``changelog.d/`` (or an examples dir).

    A :class:`Fragment` instance is just a path plus methods that interpret
    it as a changelog fragment. ``.gitkeep`` and ``*.skip`` files should
    not be wrapped — only files whose :class:`FragmentFilename` is
    ``is_fragment`` (an ``.rst`` fragment, not a skip marker).
    """

    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    @cached_property
    def _filename(self) -> FragmentFilename:
        """Cached parsed view of this fragment's filename."""
        return FragmentFilename(self.name)

    @property
    def is_valid_filename(self) -> bool:
        # ``.skip`` markers parse as a FragmentFilename but never reach a
        # Fragment — :meth:`FragmentBatch.from_dir` peels them off first.
        # Only ``.rst`` fragments need content validation and a tier.
        return self._filename.is_fragment

    @property
    def bump(self) -> str:
        """Bump tier declared by the filename suffix (defaults to ``'patch'``)."""
        return self._filename.tier or "patch"

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
        return FragmentFilename(filename).slug

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

        Filename rules: must parse as a :class:`FragmentFilename` with
        ``is_fragment`` true (``.gitkeep`` and
        ``*.skip`` files are filtered out at :meth:`FragmentBatch.from_dir`
        level and never reach this method). Content rules (for ``*.rst``
        fragments only): non-empty file with at least one valid section
        heading and at least one bullet point.
        """
        # 1. Filename shape — the suffix declares the bump tier, so an
        #    unrecognised name has no tier and cannot be compiled.
        if not self.is_valid_filename:
            return (
                f"invalid filename — must be {FragmentFilename.pattern_summary()}. "
                "Slug rules mirror git refnames (excluding `/`): non-empty, no "
                "whitespace or any of `~ ^ : ? * [ \\`, no leading `.` or `-`, "
                "no trailing `.` or `.lock`, no `..` or `@{`. Dots inside the "
                "slug are fine (e.g. `bump-newton-1.2.0rc2.minor.rst`)."
            )
        # 2. Still on disk — a fragment consumed by an earlier compile is
        #    gone and has nothing left to validate.
        if not self.path.exists():
            return None
        # 3. Non-empty — an empty file contributes no entry, so it is a
        #    mistake rather than a no-op worth accepting.
        text = self.path.read_text(encoding="utf-8")
        if not text.strip():
            return "fragment is empty"
        # 4. At least one recognised section heading — without one there is
        #    nothing to merge into the changelog.
        sections = self.parse()
        if not sections:
            return (
                "no recognised section headings (expected one or more of "
                "Added / Changed / Deprecated / Removed / Fixed, each underlined "
                "with carets ``^`` of equal-or-greater length)"
            )
        # 5. Every declared section carries a bullet — otherwise the compiled
        #    output emits a heading with no body, which is both ugly and
        #    almost certainly an authoring slip (typed the heading, forgot
        #    the bullet).
        empty = [s for s, lines in sections.items() if not any(line.lstrip().startswith("*") for line in lines)]
        if empty:
            return (
                f"section(s) {', '.join(repr(s) for s in empty)} have no bullet entries — "
                "use ``* `` to start each entry, or remove the heading"
            )
        # 6. No orphan paragraphs — every line in a section body must be a
        #    bullet (``* ``), a continuation (leading whitespace), or blank.
        #    A column-0 non-blank line that isn't a bullet terminates the
        #    list under RST rules and then sits as a paragraph adjacent to
        #    the next ``* ``, which the compile step splices into
        #    ``CHANGELOG.rst`` under the same ``^^^`` subheading; Sphinx
        #    then fails the doc build with ``Unexpected indentation``.
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

    # ---- Nested types ---------------------------------------------------

    class PartialDeletion(OSError):
        """An unlink failed part-way through consuming a batch.

        ``deleted`` holds the paths already removed, so the caller can
        account for them instead of treating a partial deletion as none.
        """

        def __init__(self, cause: OSError, deleted: list[Path]):
            super().__init__(str(cause))
            self.cause = cause
            self.deleted = deleted

    # ---- Class constants ------------------------------------------------

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
            if p.is_dir() or FragmentFilename(p.name).is_ignorable:
                continue
            if FragmentFilename(p.name).is_skip:
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

    # ---- Public API: inspect the batch, compile it, then consume it -----

    @cached_property
    def parsed(self) -> list[tuple[Fragment, dict[str, list[str]]]]:
        """``(fragment, sections)`` pairs, dropping fragments that parse empty.

        Cached because parsing re-reads every fragment from disk and a single
        compile consults it repeatedly -- the bump tier, the merged sections
        and the compiler's own progress line all derive from it. The batch is
        immutable and short-lived, so one parse per run is both correct and
        the only sensible cost.
        """
        return [(f, s) for f, s in ((f, f.parse()) for f in self.valid) if s]

    def aggregate_bump(self) -> str:
        """Highest bump tier declared by fragments that parsed to content.

        Empty fragments (which the compiler warns about and skips) are
        excluded so they don't influence the version. Defaults to
        ``patch`` if nothing parsed.
        """
        return self._aggregate([f.bump for f, _ in self.parsed])

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
        entry = self._format_entry(new_version.text, self._merged_sections())
        return new_version, bump_label, entry

    # Deletions return the paths they removed, not just counts. A deletion is
    # a change to the working tree exactly like a write is, and the nightly
    # auto-commit stages what the compile reports it changed — so a consumed
    # fragment that vanishes from disk without appearing in that report would
    # never be staged, survive on the branch, and recompile the next night
    # into a duplicate entry and a second version bump.

    def delete_all(self) -> tuple[list[Path], list[Path]]:
        """Delete every consumed fragment + skip file. Returns ``(fragments, skips)`` deleted."""
        return self._delete_valid(), self.delete_skips()

    def delete_skips(self) -> list[Path]:
        """Delete the ``.skip`` files. Returns the paths removed.

        Separate from :meth:`delete_all` because a batch of nothing but skip
        files produces no entry and no bump, yet still has to consume them.
        A ``.skip`` is matched on filename alone and never parsed, so its
        contents are irrelevant to whether it is removed.
        """
        return self._unlink_all(self.skip_paths)

    # ---- Internals ------------------------------------------------------

    def _merged_sections(self) -> dict[str, list[str]]:
        """Cross-fragment merged section map for this batch."""
        return self._merge_sections([s for _, s in self.parsed])

    def _delete_valid(self) -> list[Path]:
        """Delete the consumed fragments. Returns the paths removed."""
        return self._unlink_all([f.path for f in self.valid])

    @classmethod
    def _unlink_all(cls, paths: list[Path]) -> list[Path]:
        """Delete ``paths``, returning those actually removed.

        A failure part-way carries the deletions already made out with it
        rather than dropping them. Returning "all five or none" when three
        are genuinely gone leaves real changes unaccounted for -- the same
        class of omission that lets a consumed fragment survive on a branch.

        Raises:
            PartialDeletion: An unlink failed; ``deleted`` holds what went.
        """
        deleted: list[Path] = []
        for path in paths:
            try:
                path.unlink()
            except OSError as e:
                raise cls.PartialDeletion(e, deleted) from e
            deleted.append(path)
        return deleted

    # ---- Pure helpers ---------------------------------------------------
    # Stateless, so callers and tests can exercise them with synthetic
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
