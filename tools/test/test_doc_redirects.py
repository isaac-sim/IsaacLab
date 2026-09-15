# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for compatibility URLs after documentation moves."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def redirects():
    path = Path(__file__).resolve().parents[2] / "docs/_extensions/isaaclab_docs.py"
    spec = importlib.util.spec_from_file_location("isaaclab_docs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._write_doc_redirects


def test_redirect_preserves_old_url_and_fragment(tmp_path, redirects):
    """An old nested URL forwards to its built destination and preserves client state."""
    target = tmp_path / "source/how-to/example.html"
    target.parent.mkdir(parents=True)
    target.write_text("new guide")
    app = SimpleNamespace(
        builder=SimpleNamespace(format="html", get_outfilename=lambda doc: str(tmp_path / (doc + ".html"))),
        config=SimpleNamespace(isaaclab_doc_redirects={"source/tutorials/00_sim/example": "source/how-to/example"}),
    )
    redirects(app, None)
    html = (tmp_path / "source/tutorials/00_sim/example.html").read_text()
    assert 'href="../../how-to/example.html"' in html
    assert 'location.replace("../../how-to/example.html" + location.search + location.hash)' in html
    assert target.read_text() == "new guide"


def test_redirect_rejects_missing_destination(tmp_path, redirects):
    """Fail the build instead of publishing a redirect to a missing guide."""
    app = SimpleNamespace(
        builder=SimpleNamespace(format="html", get_outfilename=lambda doc: str(tmp_path / (doc + ".html"))),
        config=SimpleNamespace(isaaclab_doc_redirects={"old": "missing"}),
    )
    with pytest.raises(ValueError, match="target was not built: missing"):
        redirects(app, None)
    assert not (tmp_path / "old.html").exists()
