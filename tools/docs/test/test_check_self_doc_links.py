# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test the self-referencing documentation link check's mapping from URLs to source files.

The repository-wide guard is the ``check-self-doc-links`` pre-commit hook, which runs on
every pull request. These tests cover the one thing the hook cannot: that the check still
reports a break rather than silently passing, so a pattern change cannot make it a no-op.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "check_self_doc_links.py"
_spec = importlib.util.spec_from_file_location("check_self_doc_links", _MODULE_PATH)
checker = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(checker)


def test_missing_page_is_reported_ignoring_anchors(tmp_path, monkeypatch) -> None:
    """A link whose page is absent is reported at its location, anchor excluded."""
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "DOCS_ROOT", tmp_path / "docs")
    (tmp_path / "README.md").write_text(
        "- [Envs](https://isaac-sim.github.io/IsaacLab/develop/source/overview/gone.html#usage)\n"
    )
    assert [(p.name, n, d) for p, n, d in checker.find_broken_links()] == [("README.md", 1, "source/overview/gone")]


def test_directory_index_source_does_not_satisfy_a_flat_page_url(tmp_path, monkeypatch) -> None:
    """``docs/<path>/index.rst`` publishes at ``<path>/index.html``, so ``<path>.html`` is broken."""
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "DOCS_ROOT", tmp_path / "docs")
    index = tmp_path / "docs" / "source" / "overview" / "environments" / "index.rst"
    index.parent.mkdir(parents=True)
    index.write_text("Environments\n============\n")
    (tmp_path / "README.md").write_text(
        "- [Envs](https://isaac-sim.github.io/IsaacLab/develop/source/overview/environments.html)\n"
    )
    assert [doc_path for _, _, doc_path in checker.find_broken_links()] == ["source/overview/environments"]


def test_links_to_refs_built_elsewhere_are_left_to_the_live_link_checker(tmp_path, monkeypatch) -> None:
    """Only develop is built from this tree; main, release and tag URLs are checked by lychee."""
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "DOCS_ROOT", tmp_path / "docs")
    (tmp_path / "README.md").write_text(
        "".join(
            f"- [Gone](https://isaac-sim.github.io/IsaacLab/{ref}/source/overview/gone.html)\n"
            for ref in ("main", "release/3.0.0", "v3.0.0-beta2")
        )
    )
    assert checker.find_broken_links() == []
