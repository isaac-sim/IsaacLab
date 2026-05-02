# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manage changelog fragments — single entry point with two subcommands.

Each PR drops a fragment under ``source/<package>/changelog.d/<slug>.rst``.
The slug is any short, unique name — the contributor's branch name (with
``/`` replaced by ``-``) is the recommended default. The file mirrors
the RST that will appear in the changelog — one or more section headings
(``Added``, ``Changed``, ``Deprecated``, ``Removed``, ``Fixed``) each
underlined with ``^``. The **filename suffix** declares the bump tier:

- ``<slug>.rst`` — patch bump.
- ``<slug>.minor.rst`` — minor bump.
- ``<slug>.major.rst`` — major bump.
- ``<slug>.skip`` — no entry, no bump.

When a batch compiles together, the highest declared bump wins for the
package (one ``.major.rst`` anywhere → major).

Subcommands:

  check    PR gate. Verifies every modified package has a valid fragment.
  compile  Roll accumulated fragments into ``CHANGELOG.rst`` and bump
           ``extension.toml``. Run by the nightly workflow
           (``.github/workflows/nightly-changelog.yml``) on a cron and
           by maintainers manually when cutting a release.

Usage:

    # ── check ─────────────────────────────────────────────────────
    # CI invocation on every pull_request:
    cli.py check <base-branch>

    # ── compile ───────────────────────────────────────────────────
    # Normal release-time invocation — bump every managed package
    # from accumulated fragments, write entries, delete fragments:
    cli.py compile --all

    # Preview only (no writes, no deletes):
    cli.py compile --all --dry-run

    # Pin one package to a specific version (single-package only —
    # each managed package has its own version trajectory):
    cli.py compile --package isaaclab --version 4.7.0

    # Preview against a worked example without touching real packages:
    cli.py compile --package isaaclab --dry-run \\
        --fragments-dir tools/changelog/test/integration/02_minor_bump/fragments

For big version jumps (e.g. ``2.1`` → ``4.7``) edit
``source/<pkg>/config/extension.toml`` and prepend a manual entry to
``source/<pkg>/docs/CHANGELOG.rst``. The compiler is for fragment-driven
incremental bumps, not for jumps.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property
from pathlib import Path
from typing import ClassVar

# Walk three levels up: tools/changelog/cli.py -> tools/changelog/ -> tools/ -> repo root.
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
    # of an existing ``extension.toml``) but :meth:`bumped` always strips
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

    def delete_all(self) -> tuple[int, int]:
        """Delete every consumed fragment + skip file. Returns ``(n_frag, n_skip)``."""
        n_frag = self.delete_valid()
        n_skip = self.delete_skips()
        return n_frag, n_skip

    def delete_valid(self) -> int:
        for f in self.valid:
            f.path.unlink()
        return len(self.valid)

    def delete_skips(self) -> int:
        for p in self.skip_paths:
            p.unlink()
        return len(self.skip_paths)

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

    A package is "managed" if it has both a ``config/extension.toml`` (the
    version file the compiler bumps) and a ``docs/CHANGELOG.rst`` (the
    file the compiler updates). :meth:`discover` returns only managed
    packages; instances created directly may not be managed (use
    :attr:`is_managed`).
    """

    root: Path

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def changelog_path(self) -> Path:
        return self.root / "docs" / "CHANGELOG.rst"

    @property
    def toml_path(self) -> Path:
        return self.root / "config" / "extension.toml"

    @property
    def default_fragment_dir(self) -> Path:
        return self.root / "changelog.d"

    @property
    def is_managed(self) -> bool:
        return self.toml_path.is_file() and self.changelog_path.is_file()

    def current_version(self) -> Version:
        for line in self.toml_path.read_text(encoding="utf-8").splitlines():
            m = re.match(r'^version\s*=\s*"([^"]+)"', line)
            if m:
                return Version(m.group(1))
        raise ValueError(f"No version field found in {self.toml_path}")

    def write_changelog_entry(self, entry: str, *, dry_run: bool) -> None:
        text = self.changelog_path.read_text(encoding="utf-8")
        m = re.search(r"^Changelog\n-+\s*\n\s*\n", text, re.MULTILINE)
        if not m:
            raise ValueError(f"Could not locate changelog header in {self.changelog_path}")
        updated = text[: m.end()] + entry + "\n" + text[m.end() :]
        if dry_run:
            print(f"\n{'=' * 60}")
            print(f"DRY RUN — would write to {_display_path(self.changelog_path)}")
            print(f"{'=' * 60}")
            print(entry)
        else:
            self.changelog_path.write_text(updated, encoding="utf-8")

    def write_version(self, new_version: Version, *, dry_run: bool) -> None:
        text = self.toml_path.read_text(encoding="utf-8")
        updated = re.sub(r'^version\s*=\s*"[^"]+"', f'version = "{new_version}"', text, flags=re.MULTILINE)
        if dry_run:
            print(f'DRY RUN — would set version = "{new_version}" in {_display_path(self.toml_path)}')
        else:
            self.toml_path.write_text(updated, encoding="utf-8")

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
    ) -> bool:
        """Compile fragments for this package. Returns True if any were compiled.

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
                    batch.delete_skips()
                    print(f"  {self.name}: cleaned {n} stale skip file(s).")
            else:
                print(f"  {self.name}: no fragments, skipping.")
            return False

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
            return False

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
        self.write_changelog_entry(entry, dry_run=dry_run)
        self.write_version(new_version, dry_run=dry_run)

        if not dry_run:
            n_frag, n_skip = batch.delete_all()
            msg = f"  {self.name}: deleted {n_frag} fragment(s)"
            if n_skip:
                msg += f" and {n_skip} skip file(s)"
            print(msg + ".")

        return True

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


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_compile(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.fragments_dir is not None and args.all:
        parser.error("--fragments-dir requires --package (it cannot apply to all packages at once)")
    if args.version is not None and args.all:
        parser.error(
            "--version requires --package (each managed package has its own version trajectory; "
            "pin one with --package <name>)"
        )
    # Validate ``--version`` shape up front so a typo like ``--version 4.7``
    # fails at argument parsing instead of silently writing ``4.7`` into
    # ``CHANGELOG.rst`` and ``extension.toml``.
    explicit_version: Version | None = None
    if args.version is not None:
        try:
            explicit_version = Version(args.version)
        except ValueError as e:
            parser.error(f"--version: {e}")

    if args.package:
        pkg = Package.from_name(args.package)
        if not pkg.root.is_dir():
            parser.error(f"--package {args.package!r}: directory not found at {pkg.root}")
        if not pkg.is_managed:
            parser.error(
                f"--package {args.package!r} is not managed: missing config/extension.toml or "
                f"docs/CHANGELOG.rst at {pkg.root}. Run with --all to see the discovered list."
            )
        packages = [pkg]
    else:
        packages = Package.discover()

    any_compiled = False
    for pkg in packages:
        try:
            compiled = pkg.compile(
                fragments_dir=args.fragments_dir,
                explicit_version=explicit_version,
                dry_run=args.dry_run,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            return 1
        any_compiled = any_compiled or compiled

    if not any_compiled:
        print("No fragments found in any package.")
    return 0


def cmd_check(args: argparse.Namespace, _parser: argparse.ArgumentParser) -> int:
    try:
        diff = PRDiff.from_git(args.base_ref)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: git diff failed: {e.stderr}", file=sys.stderr)
        return 1

    missing, invalid_fragments = diff.evaluate(Package.discover())

    if invalid_fragments:
        print("::error::Invalid changelog fragment(s) in this PR:")
        for path, reason in invalid_fragments:
            print(f"  • {path}")
            print(f"    → {reason}")
        print()

    if missing:
        print("::error::Missing changelog fragments for the following packages:")
        for pkg_name in missing:
            print(f"  • {pkg_name}")
            print(f"    → add  source/{pkg_name}/changelog.d/<slug>.rst         (patch bump)")
            print(f"    → or   source/{pkg_name}/changelog.d/<slug>.minor.rst   (minor bump)")
            print(f"    → or   source/{pkg_name}/changelog.d/<slug>.major.rst   (major bump)")
            print(f"    → or   source/{pkg_name}/changelog.d/<slug>.skip        (no entry, no bump)")
        print()
        print("Slug = your branch name with `/` replaced by `-` (or any short, unique name).")
        print()
        print("Fragment format (source/<pkg>/changelog.d/<slug>[.minor|.major].rst):")
        print()
        print("    Added")
        print("    ^^^^^")
        print()
        print("    * Added :class:`~pkg.Foo` for feature X.")
        print()
        print("    Fixed")
        print("    ^^^^^")
        print()
        print("    * Fixed edge case in :meth:`~pkg.Foo.bar`.")
        print()
        print("See AGENTS.md ## Changelog for full guidance.")

    if invalid_fragments or missing:
        return 1

    print("✓ All modified packages have valid changelog fragments.")
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        # The module docstring carries the full usage walkthrough — surfacing
        # it as the parser description means ``cli.py --help`` shows the same
        # guidance someone reading the source would see.
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="{compile,check}")

    p_compile = sub.add_parser(
        "compile",
        help="Compile fragments into CHANGELOG.rst (maintainer release-time tool).",
        description="Compile accumulated fragments into per-package CHANGELOG.rst entries and bump extension.toml.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_compile.set_defaults(func=cmd_compile)

    # ── Target: which packages to compile (required, mutually exclusive) ──
    target = p_compile.add_argument_group("target", "Which package(s) to compile (required, mutually exclusive)")
    target_group = target.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--package", metavar="NAME", help="Compile a single package.")
    target_group.add_argument("--all", action="store_true", help="Compile all managed packages.")

    # ── Version source: by default inferred from filename suffixes ────────
    version_group = p_compile.add_argument_group(
        "version (optional)",
        "By default the new version is inferred from the filename suffixes of the consumed fragments.",
    )
    version_group.add_argument(
        "--version",
        metavar="X.Y.Z",
        help=(
            "Pin the package to an explicit version, skipping the per-fragment bump inference. "
            "Requires --package — each managed package has its own version trajectory and "
            "applying a single version to all of them would corrupt their independent histories."
        ),
    )

    # ── Execution mode: preview vs apply, where to read fragments from ────
    exec_group = p_compile.add_argument_group("execution")
    exec_group.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Preview only — no files are written or deleted. Without this flag, "
            "the compile writes the new entry, bumps the version, and deletes "
            "the consumed fragments."
        ),
    )
    exec_group.add_argument(
        "--fragments-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Override the directory to read fragments from "
            "(default: source/<pkg>/changelog.d/). "
            "Useful for previewing against example fragments without touching real ones. "
            "Only valid with --package."
        ),
    )

    p_check = sub.add_parser(
        "check",
        help="Verify each modified package has a valid fragment (PR gate).",
        description="Verify each modified package has a valid changelog fragment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_check.set_defaults(func=cmd_check)
    p_check.add_argument(
        "base_ref",
        help=(
            "Base branch to diff against (e.g. 'main' or 'develop'). "
            "The diff is taken against ``origin/<base_ref>...HEAD``."
        ),
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args, parser))


if __name__ == "__main__":
    main()
