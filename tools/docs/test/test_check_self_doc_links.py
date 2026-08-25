# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test that the self-referencing documentation link check detects a missing page.

The repository-wide guard is the ``check-self-doc-links`` pre-commit hook, which runs on
every pull request. This test covers the one thing the hook cannot: that the check still
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
