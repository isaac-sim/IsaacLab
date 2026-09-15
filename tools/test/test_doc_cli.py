# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for generated Docker CLI documentation."""

import os
from io import StringIO
from pathlib import Path

from sphinx.application import Sphinx


def test_cli_reference_rebuilds_when_parser_changes(tmp_path, monkeypatch):
    """An incremental build reflects parser changes without touching the guide."""
    docs = tmp_path / "docs"
    docs.mkdir()
    docker = tmp_path / "docker"
    docker.mkdir()
    monkeypatch.syspath_prepend(str(docker))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "docs/_extensions"))
    (docs / "conf.py").write_text("extensions = ['isaaclab_docs']\nmaster_doc = 'index'\n")
    (docs / "index.rst").write_text("CLI\n===\n\n.. isaaclab-container-cli::\n   :section: commands\n")
    parser = docker / "container.py"
    template = (
        "import argparse\n"
        "def build_parser():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    commands = parser.add_subparsers()\n"
        "    commands.add_parser('start', help={help!r})\n"
        "    return parser\n"
    )
    parser.write_text(template.format(help="Original command help"))
    warnings = StringIO()
    output = tmp_path / "html"
    app = Sphinx(docs, docs, output, tmp_path / "doctrees", "html", status=StringIO(), warning=warnings, freshenv=True)
    app.build()
    assert "Original command help" in (output / "index.html").read_text()

    updated_help = "Updated command help from parser dependency"
    parser.write_text(template.format(help=updated_help))
    # Ensure the dependency is newer even on filesystems with coarse timestamp resolution.
    os.utime(parser, (parser.stat().st_atime, parser.stat().st_mtime + 2))
    app.build()
    assert updated_help in (output / "index.html").read_text()
    assert not warnings.getvalue(), warnings.getvalue()
