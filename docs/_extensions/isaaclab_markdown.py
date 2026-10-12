# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate Markdown mirrors alongside selected HTML documentation pages."""

from __future__ import annotations

import html
import posixpath
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from bs4 import BeautifulSoup, Tag
from docutils import nodes
from markdownify import MarkdownConverter
from sphinx.application import Sphinx
from sphinx.errors import ExtensionError


class _Converter(MarkdownConverter):
    """Preserve Sphinx anchors and literal code in Markdown."""

    def convert_a(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        if el.get("id") and not el.get("href"):
            return f'\n\n<a id="{html.escape(el["id"], quote=True)}"></a>\n\n'
        return super().convert_a(el, text, parent_tags)

    def convert_pre(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        code = el.get_text()
        language = ""
        for parent in el.parents:
            for cls in parent.get("class", []):
                if cls.startswith("highlight-"):
                    language = cls.removeprefix("highlight-")
                    break
            if language:
                break
        fence = "```"
        while fence in code:
            fence += "`"
        if not code.endswith("\n"):
            code += "\n"
        return f"\n\n{fence}{language}\n{code}{fence}\n\n"


def _write_page(
    app: Sphinx, pagename: str, templatename: str, context: dict[str, Any], doctree: nodes.document | None
) -> None:
    """Convert the page body after Sphinx has resolved references and directives."""
    if app.builder.name != "html" or pagename not in app.config.isaaclab_markdown_pages:
        return

    soup = BeautifulSoup(context["body"], "html.parser")
    for element in soup.select("a.headerlink, script, style, input"):
        element.decompose()
    # Explicit anchors keep :ref: links and contents links valid despite Markdown slug differences.
    for element in list(soup.select("[id]")):
        anchor = soup.new_tag("a", id=element["id"])
        element.insert_before(anchor)
        del element["id"]
    for link in soup.select("a[href]"):
        url = urlsplit(link["href"])
        if not url.scheme and not url.netloc and url.path.endswith(".html"):
            target = posixpath.normpath(posixpath.join(posixpath.dirname(pagename), url.path))[:-5]
            if target in app.config.isaaclab_markdown_pages:
                link["href"] = urlunsplit(url._replace(path=url.path[:-5] + ".md"))

    converter = _Converter(heading_style="ATX", bullets="-", escape_underscores=False)
    markdown = converter.convert_soup(soup).strip() + "\n"
    path = Path(app.outdir) / f"{pagename}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")

    filename = posixpath.basename(pagename) + ".md"
    context["body"] = f'<p><a href="{filename}" download>Download this page as Markdown</a></p>\n' + context["body"]


def _write_index(app: Sphinx, exception: Exception | None) -> None:
    """Publish a version-local index, including on incremental builds."""
    if exception is not None or app.builder.name != "html":
        return
    lines = ["# Isaac Lab Markdown documentation", "", "Generated from this documentation build.", ""]
    for page in app.config.isaaclab_markdown_pages:
        if not (Path(app.outdir) / f"{page}.md").is_file():
            raise ExtensionError(f"Markdown mirror was not generated: {page}")
        title = app.env.titles[page].astext()
        lines.append(f"- [{title}]({page}.md)")
    (Path(app.outdir) / "markdown.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def setup(app: Sphinx) -> dict[str, str | bool]:
    """Register generation of Markdown downloads for the HTML builder."""
    app.add_config_value("isaaclab_markdown_pages", [], "html")
    app.connect("html-page-context", _write_page)
    app.connect("build-finished", _write_index)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
