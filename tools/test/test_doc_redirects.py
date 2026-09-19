# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for compatibility URLs after documentation moves."""

import importlib.util
import json
import re
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
    assert '["../../how-to/example.html", location.hash]' in html
    assert "location.replace(target[0] + location.search + target[1])" in html
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


def test_redirect_skips_missing_destination_for_historical_version(tmp_path, redirects):
    """Do not apply current redirects to historical docs that predate their destinations."""
    app = SimpleNamespace(
        builder=SimpleNamespace(format="html", get_outfilename=lambda doc: str(tmp_path / (doc + ".html"))),
        config=SimpleNamespace(
            isaaclab_doc_redirects={"old": "missing"},
            smv_current_version="v2.0.0",
        ),
    )
    redirects(app, None)
    assert not (tmp_path / "old.html").exists()


def test_redirect_routes_split_sections_and_rejects_missing_page(tmp_path, redirects):
    """An old section reaches the page that now contains it, not the default landing page."""
    for name in ("index", "cluster"):
        (tmp_path / f"{name}.html").write_text("new guide")
    app = SimpleNamespace(
        builder=SimpleNamespace(format="html", get_outfilename=lambda doc: str(tmp_path / (doc + ".html"))),
        config=SimpleNamespace(
            isaaclab_doc_redirects={"old": "index"},
            isaaclab_doc_redirect_fragments={"old": {"clusters": "cluster#deployment-cluster"}},
        ),
    )
    redirects(app, None)
    html = (tmp_path / "old.html").read_text()
    routes = json.loads(re.search(r"const sections = (.*);", html).group(1))
    assert routes["#clusters"] == ["cluster.html", "#deployment-cluster"]
    assert "location.search" in html
    (tmp_path / "cluster.html").unlink()
    with pytest.raises(ValueError, match="target was not built: cluster"):
        redirects(app, None)
