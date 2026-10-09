# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for image links and deduplication in built documentation."""

import subprocess
import sys
from pathlib import Path

from bs4 import BeautifulSoup


def test_static_images_are_reused_without_breaking_image_links(tmp_path):
    """Static images have one copy; nested, scaled, and non-static images remain usable."""
    extensions = Path(__file__).resolve().parents[2] / "docs/_extensions"
    source = tmp_path / "docs"
    static = source / "source/_static"
    page = source / "source/guide"
    page.mkdir(parents=True)
    for directory, color in (("first", "red"), ("second", "blue")):
        folder = static / directory
        folder.mkdir(parents=True)
        (folder / "image.svg").write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
            f'<rect width="10" height="10" fill="{color}"/></svg>',
            encoding="utf-8",
        )
    (page / "other.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"/>', encoding="utf-8"
    )
    (source / "conf.py").write_text(
        f"import sys\nsys.path.insert(0, {str(extensions)!r})\n"
        "extensions = ['isaaclab_docs']\nhtml_static_path = ['source/_static']\n",
        encoding="utf-8",
    )
    (source / "index.rst").write_text("Images\n======\n\n.. toctree::\n\n   source/guide/images\n", encoding="utf-8")
    (page / "images.rst").write_text(
        "Images\n======\n\n"
        ".. image:: ../_static/first/image.svg\n   :width: 5px\n\n"
        ".. image:: ../_static/second/image.svg\n\n"
        ".. image:: other.svg\n\n"
        '.. raw:: html\n\n   <img src="../../_static/first/image.svg" alt="Raw preview">\n',
        encoding="utf-8",
    )
    output = tmp_path / "html"
    subprocess.run(
        [sys.executable, "-m", "sphinx", "-W", "-q", "-j", "2", str(source), str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    html_file = output / "source/guide/images.html"
    html = BeautifulSoup(html_file.read_text(encoding="utf-8"), "html.parser")
    images = html.select('img[src$=".svg"]')
    assert len(images) == 4
    for image in images:
        assert (html_file.parent / image["src"]).is_file()
    scaled = images[0]
    assert (html_file.parent / scaled.parent["href"]).resolve() == (html_file.parent / scaled["src"]).resolve()
    assert scaled["src"] != images[1]["src"]
    assert sorted(file.name for file in (output / "_images").rglob("*.svg")) == ["other.svg"]
    assert (output / "_static/first/image.svg").is_file()
    assert (output / "_static/second/image.svg").is_file()
