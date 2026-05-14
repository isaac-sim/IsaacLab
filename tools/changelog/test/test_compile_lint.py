# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compile-output lint — the docutils-backed PR-gate check.

These tests exercise :meth:`cli.FragmentBatch._CompiledLinter.lint` and
:meth:`cli.FragmentBatch.validate_compile_output` at two layers:

1. **Direct unit tests** that feed synthetic RST text into the helper and
   assert what docutils flags. These are fast and isolate the lint logic
   from filesystem / git state.
2. **Integration tests** that build a real :class:`cli.FragmentBatch`
   from a temp directory and verify the wired-up batch method. The
   batch tests pass parsed-section dicts directly through
   :meth:`cli.FragmentBatch._merge_sections` to bypass the git-merge-time
   sort (which is unavailable in a temp dir without git history).
"""

from __future__ import annotations

from pathlib import Path

import cli
import pytest

# ---------------------------------------------------------------------------
# Helper-level tests — direct docutils integration
# ---------------------------------------------------------------------------


def test_lint_accepts_clean_compiled_output():
    """A well-formed version block parses cleanly."""
    text = (
        "1.2.3 (2026-05-14)\n"
        "~~~~~~~~~~~~~~~~~~\n\n"
        "Added\n^^^^^\n\n"
        "* Added :class:`~pkg.Foo`.\n"
        "* Added :meth:`~pkg.Foo.bar`.\n\n"
        "Changed\n^^^^^^^\n\n"
        "* Renamed :attr:`old` to :attr:`new`.\n"
    )
    assert cli.FragmentBatch._CompiledLinter.lint(text, "test") == []


def test_lint_rejects_orphan_paragraph_seam():
    """The exact #5400 incident shape: bullet list, blank line, paragraph,
    then a multi-line bullet directly underneath the paragraph (no blank line).
    The post-paragraph bullet must have a 2-space-indented continuation —
    that continuation line is what docutils reads as ``Unexpected
    indentation`` because the parser is treating the paragraph as the active
    block and the indented continuation breaks list-vs-paragraph parsing."""
    text = (
        "1.2.3 (2026-05-14)\n"
        "~~~~~~~~~~~~~~~~~~\n\n"
        "Added\n^^^^^\n\n"
        "* First bullet.\n"
        "* Second bullet.\n\n"
        "Orphan paragraph between bullets and the next bullet group.\n"
        "* Third bullet with multi-line content where the\n"
        "  continuation line is indented two spaces.\n"
    )
    errors = cli.FragmentBatch._CompiledLinter.lint(text, "test")
    assert errors, "expected docutils to flag the orphan-then-bullet seam"
    assert any("indentation" in e.lower() for e in errors)


def test_lint_rejects_unclosed_inline_literal():
    """An unclosed ``...`` literal in a bullet — Layer 1's regex misses this
    because the bullet shape is fine; docutils does catch it."""
    text = (
        "1.2.3 (2026-05-14)\n"
        "~~~~~~~~~~~~~~~~~~\n\n"
        "Added\n^^^^^\n\n"
        "* Added support for ``BaseArticulation.body_link_jacobian_w property\n"
        "  for task-space controllers.\n"
        "* Added other thing.\n"
    )
    errors = cli.FragmentBatch._CompiledLinter.lint(text, "test")
    assert errors, "expected docutils to flag the unclosed literal"
    assert any("literal" in e.lower() and "end-string" in e.lower() for e in errors)


def test_lint_rejects_under_indented_continuation():
    """A continuation line with 1-space indent instead of 2 breaks list parsing."""
    text = (
        "1.2.3 (2026-05-14)\n"
        "~~~~~~~~~~~~~~~~~~\n\n"
        "Added\n^^^^^\n\n"
        "* Added a feature with a long description that wraps to a continuation\n"
        " line indented only one space instead of two.\n"
    )
    errors = cli.FragmentBatch._CompiledLinter.lint(text, "test")
    assert errors, "expected docutils to flag the under-indented continuation"
    assert any(("unindent" in e.lower() or "indentation" in e.lower()) for e in errors)


def test_lint_accepts_sphinx_roles_as_stubs():
    """Sphinx roles (:attr:, :class:, :meth:, :paramref:, etc.) are pre-registered
    as no-op stubs so docutils doesn't emit ``Unknown interpreted text role`` noise.
    Without the stubs, every fragment using these roles would false-positive."""
    text = (
        "1.2.3 (2026-05-14)\n"
        "~~~~~~~~~~~~~~~~~~\n\n"
        "Added\n^^^^^\n\n"
        "* Added :attr:`~pkg.Foo.bar`, :class:`~pkg.Foo`, :meth:`~pkg.Foo.qux`,\n"
        "  :func:`~pkg.helper`, :mod:`pkg.sub`, :data:`pkg.CONST`,\n"
        "  :exc:`pkg.Err`, :obj:`pkg.thing`, :paramref:`~pkg.Foo.bar`, :ref:`label`.\n"
    )
    assert cli.FragmentBatch._CompiledLinter.lint(text, "test") == []


def test_lint_accepts_prose_intro_then_bullets():
    """The ``isaaclab_rl 0.1.0`` initial-version shape: prose paragraph,
    blank line, then a bullet list. Valid RST; Layer 1's regex would have
    rejected this but Layer 2 correctly accepts it."""
    text = (
        "0.1.0 (2024-12-27)\n"
        "~~~~~~~~~~~~~~~~~~\n\n"
        "Added\n^^^^^\n\n"
        "Initial version of the extension.\n"
        "This extension is split off from ``other_pkg`` to include the wrapper\n"
        "scripts for the supported RL libraries.\n\n"
        "Supported RL libraries are:\n\n"
        "* RL Games\n"
        "* RSL RL\n"
        "* SKRL\n"
        "* Stable Baselines3\n"
    )
    assert cli.FragmentBatch._CompiledLinter.lint(text, "test") == []


def test_lint_returns_deduplicated_messages():
    """Multiple emissions of the same diagnostic collapse to a single entry."""
    text = (
        "1.2.3 (2026-05-14)\n"
        "~~~~~~~~~~~~~~~~~~\n\n"
        "Added\n^^^^^\n\n"
        "* First with ``unclosed and\n"
        "* Second with ``also unclosed and\n"
    )
    errors = cli.FragmentBatch._CompiledLinter.lint(text, "test")
    # We don't assert an exact count (docutils' emission count is internal)
    # but the same (line, text) pair must never appear twice.
    assert len(errors) == len(set(errors))


# ---------------------------------------------------------------------------
# Batch-level tests — FragmentBatch.validate_compile_output
# ---------------------------------------------------------------------------


def _write(d: Path, name: str, body: str) -> Path:
    p = d / name
    p.write_text(body, encoding="utf-8")
    return p


def test_validate_compile_output_empty_batch_is_clean(tmp_path):
    """A directory with only ``.gitkeep`` / no fragments produces no entry to lint."""
    (tmp_path / ".gitkeep").touch()
    batch = cli.FragmentBatch.from_dir(tmp_path)
    assert batch.validate_compile_output("pkg") is None


def test_validate_compile_output_clean_single_fragment(tmp_path):
    """A well-formed single fragment compiles to clean RST."""
    _write(
        tmp_path,
        "1234.rst",
        "Added\n^^^^^\n\n* Added :class:`~pkg.Foo`.\n",
    )
    batch = cli.FragmentBatch.from_dir(tmp_path)
    assert batch.validate_compile_output("pkg") is None


def test_validate_compile_output_isolated_orphan_paragraph_is_clean(tmp_path):
    """The #5400 fragment *alone* compiles to a valid block (paragraph followed
    by next section underline = legal RST). The bug only forms when a sibling
    fragment adds bullets to the same section."""
    _write(
        tmp_path,
        "1234.rst",
        (
            "Added\n^^^^^\n\n"
            "* First bullet.\n"
            "* Second bullet.\n\n"
            "Trailing paragraph at column 0.\n\n"
            "Changed\n^^^^^^^\n\n"
            "* Some change.\n"
        ),
    )
    batch = cli.FragmentBatch.from_dir(tmp_path)
    assert batch.validate_compile_output("pkg") is None


def test_validate_compile_output_catches_merge_seam():
    """Two fragments individually OK, merged output broken — the actual #5400
    failure mode. Tests :meth:`_merge_sections` + :meth:`_format_entry` +
    :meth:`_CompiledLinter.lint` end-to-end without depending on git merge-time
    ordering (we hand the section dicts directly to :meth:`_merge_sections`)."""
    frag_a_sections = {
        "Added": [
            "* First bullet from A.",
            "* Second bullet from A.",
            "",
            "Trailing paragraph from fragment A.",
            "Second line of that paragraph.",
        ],
    }
    frag_b_sections = {
        "Added": [
            # Must be multi-line — the continuation indent is what docutils
            # actually flags as ``Unexpected indentation``.
            "* Bullet from B that has multi-line content with a",
            "  continuation line indented two spaces.",
        ],
    }
    merged = cli.FragmentBatch._merge_sections([frag_a_sections, frag_b_sections])
    entry = cli.FragmentBatch._format_entry("1.2.3", merged)
    errors = cli.FragmentBatch._CompiledLinter.lint(entry, "<pkg compiled>")
    assert errors, "expected docutils to flag the orphan-then-bullet seam"
    assert any("indentation" in e.lower() for e in errors)


def test_validate_compile_output_catches_single_fragment_with_inline_defect(tmp_path):
    """A single fragment with an unclosed inline literal — Layer 1's regex
    accepts it (bullet shape is fine), Layer 2 rejects via docutils."""
    _write(
        tmp_path,
        "1234.rst",
        (
            "Added\n^^^^^\n\n"
            "* Added support for ``BaseArticulation.body_link_jacobian_w property\n"
            "  for task-space controllers.\n"
        ),
    )
    batch = cli.FragmentBatch.from_dir(tmp_path)
    err = batch.validate_compile_output("pkg")
    assert err is not None
    assert "pkg" in err
    assert "literal" in err.lower()


def test_validate_compile_output_error_names_package(tmp_path):
    """Error message includes the package name so authors can attribute it."""
    _write(
        tmp_path,
        "1234.rst",
        "Added\n^^^^^\n\n* First.\n* Second with ``unclosed literal\n",
    )
    batch = cli.FragmentBatch.from_dir(tmp_path)
    err = batch.validate_compile_output("my_specific_package")
    assert err is not None
    assert "'my_specific_package'" in err


def test_validate_compile_output_skip_only_batch_is_clean(tmp_path):
    """A batch with only ``.skip`` fragments has nothing to compile."""
    _write(tmp_path, "ci-only.skip", "")
    batch = cli.FragmentBatch.from_dir(tmp_path)
    assert batch.valid == []
    assert batch.validate_compile_output("pkg") is None


# ---------------------------------------------------------------------------
# Gate-orchestration integration — PRDiff.evaluate wires the new check in
# ---------------------------------------------------------------------------


def _pkg(tmp_path: Path, name: str) -> cli.Package:
    """Build a managed Package at ``tmp_path/source/<name>/``."""
    root = tmp_path / "source" / name
    (root / "config").mkdir(parents=True)
    (root / "docs").mkdir(parents=True)
    (root / "config" / "extension.toml").write_text('version = "0.0.0"\n', encoding="utf-8")
    (root / "docs" / "CHANGELOG.rst").write_text("Changelog\n---------\n\n", encoding="utf-8")
    return cli.Package(root)


def test_evaluate_passes_a_clean_pr_fragment(tmp_path):
    """Sanity: a clean PR fragment + clean existing fragments → no errors."""
    pkg = _pkg(tmp_path, "isaaclab")
    (pkg.root / "changelog.d").mkdir()
    _write(pkg.root / "changelog.d", "alice-feature.rst", "Added\n^^^^^\n\n* Alice feature.\n")
    _write(pkg.root / "changelog.d", "bob-feature.rst", "Added\n^^^^^\n\n* Bob feature.\n")
    changed = {
        "source/isaaclab/code.py",
        "source/isaaclab/changelog.d/bob-feature.rst",
    }
    added = {"source/isaaclab/changelog.d/bob-feature.rst"}
    # Monkey-patch REPO_ROOT so Fragment(REPO_ROOT / f).validate() reads our temp tree.
    import unittest.mock

    with unittest.mock.patch.object(cli, "REPO_ROOT", tmp_path):
        missing, invalid = cli.PRDiff(changed=changed, added=added).evaluate([pkg])
    assert missing == []
    assert invalid == []


def test_evaluate_flags_compiled_output_failure(tmp_path):
    """When the PR-merged fragment shape would break the doc build, the gate
    rejects with the new compile-output-failure error path."""
    pkg = _pkg(tmp_path, "isaaclab")
    (pkg.root / "changelog.d").mkdir()
    _write(
        pkg.root / "changelog.d",
        "bad-pr.rst",
        "Added\n^^^^^\n\n* Has ``unclosed literal in this bullet.\n",
    )
    changed = {
        "source/isaaclab/code.py",
        "source/isaaclab/changelog.d/bad-pr.rst",
    }
    added = {"source/isaaclab/changelog.d/bad-pr.rst"}
    import unittest.mock

    with unittest.mock.patch.object(cli, "REPO_ROOT", tmp_path):
        missing, invalid = cli.PRDiff(changed=changed, added=added).evaluate([pkg])
    # The bad fragment isn't malformed at the per-fragment level (shape is OK),
    # so it slips past Rule 2 and is caught by the new Rule 4 (compile-output lint).
    assert any("compiled fragment output" in r for _, r in invalid), (
        f"expected compile-output error in invalid_fragments, got {invalid!r}"
    )


def test_evaluate_skips_compile_lint_when_pr_does_not_touch_fragments(tmp_path):
    """Source-only PRs that don't add fragments don't trigger the compile lint
    (the existing on-base fragments are not this PR's responsibility)."""
    pkg = _pkg(tmp_path, "isaaclab")
    (pkg.root / "changelog.d").mkdir()
    # Pre-existing fragment with a defect; PR doesn't add any fragment.
    _write(
        pkg.root / "changelog.d",
        "preexisting-bad.rst",
        "Added\n^^^^^\n\n* ``unclosed.\n",
    )
    changed = {"source/isaaclab/code.py"}  # no changelog.d changes
    added = set()
    import unittest.mock

    with unittest.mock.patch.object(cli, "REPO_ROOT", tmp_path):
        missing, invalid = cli.PRDiff(changed=changed, added=added).evaluate([pkg])
    # The defective preexisting fragment is not flagged; only "missing fragment for
    # this package" is reported (Rule 5).
    assert missing == ["isaaclab"]
    assert invalid == []  # compile-lint did not run


@pytest.mark.parametrize(
    "section_body, expected_clean",
    [
        # Valid shapes
        (["* one", "* two"], True),
        (["* one", "  continuation", "* two"], True),
        (["* one with :class:`~pkg.Foo` ref."], True),
        # Invalid shapes
        (["* one", "* two with ``unclosed and", "  continuation"], False),
        (["* one", " under-indented continuation"], False),
    ],
)
def test_lint_compiled_output_parametrized(section_body, expected_clean):
    """Compact matrix of known-good / known-bad shapes against the compile lint."""
    text = "1.2.3 (2026-05-14)\n~~~~~~~~~~~~~~~~~~\n\nAdded\n^^^^^\n\n" + "\n".join(section_body) + "\n"
    errors = cli.FragmentBatch._CompiledLinter.lint(text, "test")
    if expected_clean:
        assert errors == [], f"expected clean parse, got {errors}"
    else:
        assert errors, f"expected docutils to flag, got clean parse for: {section_body}"
