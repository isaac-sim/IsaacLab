# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for GitHub links to API implementations."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def test_source_links_resolve_lazy_exports_and_inherited_properties(tmp_path, monkeypatch):
    """Lazy exports link to their implementation and documented ref; unresolved objects have no link."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "docs/_extensions"))
    from isaaclab_docs import _configure_source_links

    source = tmp_path / "source"
    source.mkdir()
    path = source / "source_link_example.py"
    path.write_text(
        "class Base:\n    @property\n    def value(self):\n        return 1\n\nclass Derived(Base):\n    pass\n"
        "Alias = Derived\ndef __getattr__(name):\n    if name == 'Derived':\n        return Alias\n"
        "    raise AttributeError(name)\ndel Derived\n"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, path.stem, module)
    spec.loader.exec_module(module)
    config = SimpleNamespace(isaaclab_latest_branch="develop", smv_current_version="v3.0.0")
    app = SimpleNamespace(srcdir=tmp_path / "docs")
    info = {"module": path.stem, "fullname": "Derived.value"}
    _configure_source_links(app, config)
    assert config.linkcode_resolve("py", info) == (
        "https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/source_link_example.py#L2-L4"
    )
    config.smv_current_version = ""
    _configure_source_links(app, config)
    assert config.linkcode_resolve("py", info) == (
        "https://github.com/isaac-sim/IsaacLab/blob/develop/source/source_link_example.py#L2-L4"
    )
    assert config.linkcode_resolve("py", {"module": path.stem, "fullname": "missing"}) is None
    assert config.linkcode_resolve("py", {"module": "pathlib", "fullname": "Path"}) is None
