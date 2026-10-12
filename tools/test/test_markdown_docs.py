# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify Markdown downloads through a real Sphinx HTML build."""

from __future__ import annotations

import io
from pathlib import Path

from sphinx.application import Sphinx


def test_markdown_mirrors(tmp_path: Path) -> None:
    """Mirrors retain resolved content and links and update on incremental builds."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "setup").mkdir()
    extensions = Path(__file__).resolve().parents[2] / "docs" / "_extensions"
    (source / "conf.py").write_text(
        f"import sys\nsys.path.insert(0, {str(extensions)!r})\n"
        "extensions = ['isaaclab_docs', 'sphinx_design', 'isaaclab_markdown']\n"
        "isaaclab_latest_branch = 'v9.8.7'\n"
        "isaaclab_markdown_pages = ['setup/guide', 'troubleshooting']\n",
        encoding="utf-8",
    )
    (source / "index.rst").write_text(
        "Docs\n====\n\n.. toctree::\n\n   setup/guide\n   troubleshooting\n   other\n", encoding="utf-8"
    )
    (source / "other.rst").write_text("Other\n=====\n", encoding="utf-8")
    (source / "troubleshooting.rst").write_text(
        ".. _fix-imports:\n\nTroubleshooting\n===============\n\nFix imports here.\n", encoding="utf-8"
    )
    fragment = source / "setup" / "commands.inc"
    fragment.write_text("Included instructions.\n\n.. isaaclab-clone-https::\n", encoding="utf-8")
    (source / "setup" / "guide.rst").write_text(
        """Setup
=====

See :ref:`fix-imports`, :doc:`/other`, and `external <https://example.com/guide.html>`__.

.. include:: commands.inc

.. tab-set::

   .. tab-item:: Linux

      .. code-block:: bash

         echo '```'

   .. tab-item:: Windows

      .. code-block:: batch

         echo Windows

.. dropdown:: Recovery

   .. warning::

      Keep this warning.

.. list-table::
   :header-rows: 1

   * - Error
     - Fix
   * - Missing module
     - Use uv
""",
        encoding="utf-8",
    )
    warnings = io.StringIO()
    output = tmp_path / "html"
    app = Sphinx(
        str(source),
        str(source),
        str(output),
        str(tmp_path / "doctrees"),
        "html",
        status=io.StringIO(),
        warning=warnings,
        freshenv=True,
    )
    app.build()
    assert not warnings.getvalue()
    mirror = (output / "setup" / "guide.md").read_text(encoding="utf-8")
    assert "# Setup" in mirror
    assert "Included instructions." in mirror
    assert "git clone https://github.com/isaac-sim/IsaacLab.git --branch v9.8.7" in mirror
    assert "````bash\necho '```'\n````" in mirror
    assert "Linux" in mirror and "Windows" in mirror
    assert "```batch\necho Windows\n```" in mirror
    assert "Recovery" in mirror and "Keep this warning." in mirror
    assert "| Error | Fix |" in mirror
    assert "| Missing module | Use uv |" in mirror
    assert "../troubleshooting.md#fix-imports" in mirror
    assert "../other.html" in mirror
    assert "https://example.com/guide.html" in mirror
    assert '<a id="fix-imports"></a>' in (output / "troubleshooting.md").read_text(encoding="utf-8")
    html = (output / "setup" / "guide.html").read_text(encoding="utf-8")
    assert 'href="guide.md" download' in html
    assert "Download this page" not in mirror
    assert not (output / "other.md").exists()
    index = (output / "markdown.md").read_text(encoding="utf-8")
    assert "[Setup](setup/guide.md)" in index
    assert "[Troubleshooting](troubleshooting.md)" in index

    fragment.write_text("Updated instructions.\n", encoding="utf-8")
    app.build()
    updated = (output / "setup" / "guide.md").read_text(encoding="utf-8")
    assert "Updated instructions." in updated
    assert "Included instructions." not in updated
    assert not warnings.getvalue()
