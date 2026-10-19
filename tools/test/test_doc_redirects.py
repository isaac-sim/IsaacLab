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


@pytest.fixture
def pin_historical_versions():
    path = Path(__file__).resolve().parents[2] / "tools/docs/pin_historical_versions.py"
    spec = importlib.util.spec_from_file_location("pin_historical_versions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pin_historical_versions


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


def test_redirect_supports_versioned_documentation_url(tmp_path, redirects):
    """A retired guide can forward to the immutable documentation version that owns it."""
    target = "https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/walkthrough/index.html"
    app = SimpleNamespace(
        builder=SimpleNamespace(format="html", get_outfilename=lambda doc: str(tmp_path / (doc + ".html"))),
        config=SimpleNamespace(isaaclab_doc_redirects={"source/setup/walkthrough/index": target}),
    )
    redirects(app, None)
    html = (tmp_path / "source/setup/walkthrough/index.html").read_text()
    assert f'href="{target}"' in html
    assert f"[{json.dumps(target)}, location.hash]" in html
    assert "location.replace(target[0] + location.search + target[1])" in html


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


def test_redirect_skips_missing_fragment_destination_for_historical_version(tmp_path, redirects):
    """Keep a historical page redirect when a current fragment destination is unavailable."""
    (tmp_path / "index.html").write_text("new guide")
    app = SimpleNamespace(
        builder=SimpleNamespace(format="html", get_outfilename=lambda doc: str(tmp_path / (doc + ".html"))),
        config=SimpleNamespace(
            isaaclab_doc_redirects={"old": "index"},
            isaaclab_doc_redirect_fragments={"old": {"overview": "missing#section"}},
            smv_current_version="v2.0.0",
        ),
    )
    redirects(app, None)
    html = (tmp_path / "old.html").read_text()
    routes = json.loads(re.search(r"const sections = (.*);", html).group(1))
    assert routes == {}
    assert '["index.html", location.hash]' in html


def test_redirect_routes_page_and_sections_and_rejects_missing_page(tmp_path, redirects):
    """Old page roots and sections reach the page and section that now contain them."""
    for name in ("index", "cluster"):
        (tmp_path / f"{name}.html").write_text("new guide")
    app = SimpleNamespace(
        builder=SimpleNamespace(format="html", get_outfilename=lambda doc: str(tmp_path / (doc + ".html"))),
        config=SimpleNamespace(
            isaaclab_doc_redirects={"old": "index"},
            isaaclab_doc_redirect_fragments={
                "old": {"": "cluster#deployment-cluster", "clusters": "cluster#deployment-cluster"}
            },
        ),
    )
    redirects(app, None)
    html = (tmp_path / "old.html").read_text()
    routes = json.loads(re.search(r"const sections = (.*);", html).group(1))
    assert routes[""] == ["cluster.html", "#deployment-cluster"]
    assert routes["#clusters"] == ["cluster.html", "#deployment-cluster"]
    assert "location.search" in html
    (tmp_path / "cluster.html").unlink()
    with pytest.raises(ValueError, match="target was not built: cluster"):
        redirects(app, None)


def test_historical_docs_pin_links_and_clone_commands_to_version(tmp_path, pin_historical_versions):
    """Generated 2.x docs remain self-consistent after the default branch changes."""
    space = '<span class="w"> </span>'
    original = (
        '<a href="https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html">docs</a>'
        '<a href="https://github.com/isaac-sim/IsaacLab/blob/main/isaaclab.sh">source</a>'
        '<a href="https://github.com/isaac-sim/IsaacLab/tree/main/source">tree</a>'
        f"git{space}clone{space}https://github.com/isaac-sim/IsaacLab.git"
        "git clone git@github.com:isaac-sim/IsaacLab.git"
    )
    historical = tmp_path / "v2.3.2/page.html"
    historical.parent.mkdir()
    historical.write_text(original)
    current = tmp_path / "v3.0.0/page.html"
    current.parent.mkdir()
    current.write_text(original)

    assert pin_historical_versions(tmp_path) == 1
    html = historical.read_text()
    assert "https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/quickstart.html" in html
    assert "https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/isaaclab.sh" in html
    assert "https://github.com/isaac-sim/IsaacLab/tree/v2.3.2/source" in html
    assert f"clone{space}--branch{space}v2.3.2{space}https://github.com/isaac-sim/IsaacLab.git" in html
    assert "git clone --branch v2.3.2 git@github.com:isaac-sim/IsaacLab.git" in html
    assert current.read_text() == original
