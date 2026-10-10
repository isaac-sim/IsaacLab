# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build-boundary checks for browser simulation embeds."""

from __future__ import annotations

import gzip
import io
import json
from html.parser import HTMLParser
from pathlib import Path

import pytest
from sphinx.testing.util import SphinxTestApp


class _PageAssets(HTMLParser):
    def __init__(self, html: str):
        super().__init__()
        self.widgets = []
        self.assets = []
        self.feed(html)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "isaaclab-browser-demo":
            self.widgets.append(values)
        elif tag in ("script", "link"):
            target = values.get("src") or values.get("href") or ""
            if "browser-demo." in target:
                self.assets.append(target)


@pytest.fixture
def docs_source(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    bundle = root.parent / "contrib/browser_demos/static/browser_demos/example"
    bundle.mkdir(parents=True)
    (root / "source/_static").mkdir(parents=True)
    extension_dir = Path(__file__).resolve().parents[2] / "docs/_extensions"
    (root / "conf.py").write_text(
        f"import sys\nsys.path.insert(0, {str(extension_dir)!r})\n"
        "extensions = ['isaaclab_docs']\n"
        "html_static_path = ['source/_static', '../contrib/browser_demos/static']\n",
        encoding="utf-8",
    )
    (root / "index.rst").write_text("Examples\n========\n\n.. toctree::\n\n   guide/nested\n", encoding="utf-8")
    (root / "guide").mkdir()
    (root / "guide/nested.rst").write_text(
        "Nested\n======\n\n.. isaaclab-browser-demo:: example\n   :title: Gains & limits\n\n"
        ".. isaaclab-browser-demo:: example\n",
        encoding="utf-8",
    )
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "bundleVersion": 1,
                "abiVersion": 1,
                "module": "simulation.mjs",
                "wasm": "simulation.wasm",
                "isaacLabDemo": {
                    "policy": {"files": ["policy-0.bin", "policy-1.bin"]},
                    "visuals": [{"file": "../shared/robot.bin"}],
                },
            }
        ),
        encoding="utf-8",
    )
    (bundle / "simulation.mjs").write_text("export default () => {};\n", encoding="utf-8")
    (bundle / "simulation.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")
    for name in ("policy-0.bin", "policy-1.bin"):
        (bundle / name).write_bytes(bytes(4))
    shared = bundle.parent / "shared"
    shared.mkdir()
    (shared / "robot.bin").write_bytes(bytes(4))
    css = bundle.parents[1] / "css"
    css.mkdir()
    (css / "browser-demo.css").write_text("", encoding="utf-8")
    (css / "browser-demo.js").write_text("", encoding="utf-8")
    return root


def _build(source: Path, builder: str) -> tuple[Path, int, str]:
    warnings = io.StringIO()
    app = SphinxTestApp(
        buildername=builder,
        srcdir=source,
        builddir=source.parent / "build",
        status=io.StringIO(),
        warning=warnings,
        warningiserror=True,
        freshenv=True,
    )
    try:
        app.build()
        return app.outdir, app.statuscode, warnings.getvalue()
    finally:
        app.cleanup()


def _use_chunked_wasm(bundle: Path) -> list[str]:
    compressed = gzip.compress((bundle / "simulation.wasm").read_bytes(), mtime=0)
    files = ["simulation.wasm.gz.part0", "simulation.wasm.gz.part1"]
    split = len(compressed) // 2
    for filename, data in zip(files, (compressed[:split], compressed[split:]), strict=True):
        (bundle / filename).write_bytes(data)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest.update(wasm="simulation.wasm.gz", wasmFiles=files)
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (bundle / "simulation.wasm").unlink()
    return files


@pytest.mark.parametrize("builder", ["html", "dirhtml"])
def test_browser_embeds_resolve_assets_and_load_widget_once(docs_source: Path, builder: str):
    bundle = docs_source.parent / "contrib/browser_demos/static/browser_demos/example"
    if builder == "dirhtml":
        _use_chunked_wasm(bundle)
    output, status, warnings = _build(docs_source, builder)
    assert status == 0, warnings
    page = output / ("guide/nested.html" if builder == "html" else "guide/nested/index.html")
    assets = _PageAssets(page.read_text(encoding="utf-8"))
    assert len(assets.widgets) == 2
    assert assets.widgets[0]["demo-title"] == "Gains & limits"
    for widget in assets.widgets:
        manifest_path = (page.parent / widget["src"]).resolve()
        manifest = json.loads(manifest_path.read_text())
        files = manifest.get("wasmFiles", [manifest["wasm"]])
        binary = b"".join((manifest_path.parent / file).read_bytes() for file in files)
        if manifest["wasm"].endswith(".gz"):
            binary = gzip.decompress(binary)
        assert binary == b"\x00asm\x01\x00\x00\x00"
    assert len(assets.assets) == 2
    for target in assets.assets:
        assert (page.parent / target.split("?", 1)[0]).resolve().is_file()
    assert _PageAssets((output / "index.html").read_text(encoding="utf-8")).assets == []


@pytest.mark.parametrize(
    "problem",
    [
        "unknown-demo",
        "missing-binary",
        "missing-wasm-chunk",
        "missing-policy-chunk",
        "missing-shared-visual",
        "lfs-pointer",
        "future-abi",
    ],
)
def test_browser_embeds_reject_unusable_bundles(docs_source: Path, problem: str):
    bundle = docs_source.parent / "contrib/browser_demos/static/browser_demos/example"
    if problem == "unknown-demo":
        (bundle / "manifest.json").unlink()
    elif problem == "missing-binary":
        (bundle / "simulation.wasm").unlink()
    elif problem == "missing-wasm-chunk":
        files = _use_chunked_wasm(bundle)
        (bundle / files[-1]).unlink()
    elif problem == "missing-policy-chunk":
        (bundle / "policy-1.bin").unlink()
    elif problem == "missing-shared-visual":
        (bundle.parent / "shared/robot.bin").unlink()
    elif problem == "lfs-pointer":
        (bundle / "policy-1.bin").write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 4\n",
            encoding="utf-8",
        )
    else:
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["abiVersion"] = 2
        (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _, status, warnings = _build(docs_source, "html")
    assert status == 1
    assert "Cannot embed browser demo 'example'" in warnings
    if problem == "missing-wasm-chunk":
        assert files[-1] in warnings
    if problem == "lfs-pointer":
        assert "Git LFS pointer" in warnings


def test_browser_embeds_allow_text_documentation(docs_source: Path):
    output, status, warnings = _build(docs_source, "text")
    assert status == 0, warnings
    text = (output / "guide/nested.txt").read_text(encoding="utf-8")
    assert "Nested" in text
    assert "isaaclab-browser-demo" not in text
