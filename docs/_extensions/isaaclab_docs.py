# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sphinx helpers for Isaac Lab documentation."""

from __future__ import annotations

import csv
import inspect
import json
import posixpath
import re
import sys
from html import escape
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.config import Config
from sphinx.util.docutils import SphinxDirective, SphinxRole
from sphinx.util.nodes import split_explicit_title
from sphinx.util.osutil import relative_uri

_UPSTREAM_SOURCE_REF_PATTERN = re.compile(r"^(main|develop|release/.*|v[1-9]\d*\.\d+\.\d+(-[A-Za-z0-9.]+)?)$")


class IsaacLabHTMLBuilder(StandaloneHTMLBuilder):
    """Reuse static images instead of shipping another copy under ``_images``."""

    def post_process_images(self, doctree: nodes.Node) -> None:
        original_images = {}
        for image in doctree.findall(nodes.image):
            for uri in image.get("candidates", {}).values():
                if uri.startswith("source/_static/") and uri in self.env.images and uri not in original_images:
                    original_images[uri] = self.env.images[uri]
                    docnames, _ = self.env.images[uri]
                    # Keep Sphinx's relative image paths and scaled-image links, pointing
                    # them at the file already copied by html_static_path.
                    self.env.images[uri] = (docnames, "../_static/" + uri.removeprefix("source/_static/"))
        try:
            super().post_process_images(doctree)
        finally:
            self.env.images.update(original_images)


def _branch(config) -> str:
    """Return the branch or tag pinned in installation docs."""
    current_version = getattr(config, "smv_current_version", "")
    if current_version:
        return current_version
    return getattr(config, "isaaclab_latest_branch", "main")


def _source_branch(config) -> str:
    """Return a GitHub source ref that exists in the upstream repository."""
    branch = _branch(config)
    if _UPSTREAM_SOURCE_REF_PATTERN.match(branch):
        return branch
    return getattr(config, "isaaclab_latest_branch", "main")


def _configure_source_links(app: Sphinx, config: Config) -> None:
    """Link API objects to their implementation in the documented Git ref."""
    root = Path(app.srcdir).resolve().parent
    branch = _source_branch(config)

    def resolve(domain: str, info: dict[str, str]) -> str | None:
        if domain != "py":
            return None
        try:
            name, *members = info["fullname"].split(".")
            obj = getattr(sys.modules[info["module"]], name)
            for member in members:
                obj = inspect.getattr_static(obj, member)
            if isinstance(obj, property):
                obj = obj.fget
            elif isinstance(obj, (classmethod, staticmethod)):
                obj = obj.__func__
            obj = inspect.unwrap(obj)
            filename = inspect.getsourcefile(obj)
            if filename is None:
                return None
            path = Path(filename).resolve().relative_to(root)
            if path.parts[0] != "source":
                return None
            lines, start = inspect.getsourcelines(obj)
        except (AttributeError, KeyError, OSError, TypeError, ValueError):
            # External, mocked, or generated objects may have no repository source.
            return None
        return (
            f"https://github.com/isaac-sim/IsaacLab/blob/{branch}/{path.as_posix()}#L{start}-L{start + len(lines) - 1}"
        )

    config.linkcode_resolve = resolve


def _parse_rst(directive: SphinxDirective, content: str) -> list[nodes.Node]:
    """Parse nested reST and return the generated document nodes."""
    source = directive.env.doc2path(directive.env.docname, base=False)
    lines = StringList(content.splitlines(), source=source)
    container = nodes.container()
    directive.state.nested_parse(lines, 0, container)
    return container.children


class IsaacLabBenchmarkData(SphinxDirective):
    """Embed benchmark snapshots so downloaded HTML works without an HTTP server."""

    def run(self) -> list[nodes.Node]:
        snapshots = {}
        for channel in ("release", "develop"):
            path = Path(self.env.srcdir) / "source/_static/benchmarks" / f"environment-performance-{channel}.csv"
            self.env.note_dependency(str(path))
            with path.open(encoding="utf-8", newline="") as stream:
                snapshots[channel] = list(csv.DictReader(stream))
        # Prevent CSV text from closing the script element, even inside a JSON string.
        payload = json.dumps(snapshots).replace("<", "\\u003c")
        html = f'<script type="application/json" data-environment-benchmark-rows>{payload}</script>'
        return [nodes.raw("", html, format="html")]


class IsaacLabDemos(SphinxDirective):
    """Embed the staged interactive gallery without maintaining a second catalog."""

    optional_arguments = 1
    final_argument_whitespace = True
    has_content = False
    option_spec = {
        "title": directives.unchanged_required,
        "physics": lambda value: directives.choice(value, ("rigid", "vbd", "mpm")),
        "activity": lambda value: directives.choice(value, ("tuning", "policy", "scripted")),
    }

    def run(self) -> list[nodes.Node]:
        base = self.config.isaaclab_demo_base_url
        if base:
            address = urlsplit(base)
            if address.scheme not in {"http", "https"} or not address.netloc:
                raise self.error("isaaclab_demo_base_url must be empty or an absolute HTTP or HTTPS URL.")
        else:
            base = relative_uri(self.env.app.builder.get_target_uri(self.env.docname), "../demos/interactive/")
        ids = [value.strip() for value in self.arguments[0].split(",")] if self.arguments else []
        if any(not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in ids):
            raise self.error("Demo IDs must use letters, numbers, underscores or hyphens, separated by commas.")
        address = urlsplit(base)
        parameters = dict(parse_qsl(address.query, keep_blank_values=True))
        parameters["demos"] = ",".join(ids) or "all"
        for key in ("physics", "activity"):
            if key in self.options:
                parameters[key] = self.options[key]
        gallery_url = address._replace(query=urlencode(parameters)).geturl()
        title = escape(self.options.get("title", "Interactive demos"), quote=True)
        html = (
            '<section class="isaaclab-demos" data-isaaclab-demos '
            f'data-base-url="{escape(base, quote=True)}" data-demo-ids="{escape(",".join(ids), quote=True)}" '
            f'data-physics="{self.options.get("physics", "")}" data-activity="{self.options.get("activity", "")}" '
            f'aria-label="{title}">'
            '<div class="isaaclab-demos-heading">'
            f"<strong>{title}</strong>"
            f'<a href="{escape(gallery_url, quote=True)}" target="_blank" rel="noopener noreferrer" '
            'aria-label="Open demo gallery in a new tab">Open gallery</a>'
            "</div>"
            '<p class="isaaclab-demos-status" role="status" aria-live="polite">'
            "Interactive demos load when this section is visible.</p>"
            f'<iframe class="isaaclab-demos-frame" title="{title}" loading="lazy" '
            'height="320" allow="fullscreen" aria-busy="true"></iframe>'
            "<noscript><p>Enable JavaScript to use the embedded gallery, or open the demo gallery above.</p></noscript>"
            "</section>"
        )
        return [nodes.raw("", html, format="html", isaaclab_demos=True)]


def _add_demo_assets(
    app: Sphinx, pagename: str, templatename: str, context: dict, doctree: nodes.document | None
) -> None:
    """Load the embed bridge only on pages that contain a gallery."""
    if doctree is not None and any(node.get("isaaclab_demos") for node in doctree.findall(nodes.raw)):
        # Sphinx resets page assets before emitting html-page-context.
        app.add_css_file("css/interactive-demos.css")
        app.add_js_file("css/interactive-demos.js", loading_method="defer")


class IsaacLabCloneCommands(SphinxDirective):
    """Render SSH/HTTPS clone tabs using copy-friendly ``code-block`` directives."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        branch = _branch(self.config)
        content = f"""\
.. tab-set::

   .. tab-item:: SSH

      .. code-block:: bash

         git clone git@github.com:isaac-sim/IsaacLab.git --branch {branch}
         cd IsaacLab

   .. tab-item:: HTTPS

      .. code-block:: bash

         git clone https://github.com/isaac-sim/IsaacLab.git --branch {branch}
         cd IsaacLab
"""
        return _parse_rst(self, content)


class IsaacLabSourceLink(SphinxRole):
    """Link to a source file on the GitHub branch or tag for the current docs version."""

    def run(self) -> tuple[list[nodes.Node], list[nodes.system_message]]:
        branch = _source_branch(self.config)
        has_explicit_title, title, target = split_explicit_title(self.text)
        if not has_explicit_title:
            title = target
        target = target.strip("/")
        refuri = f"https://github.com/isaac-sim/IsaacLab/blob/{branch}/{target}"
        node = nodes.reference(self.rawtext, title, refuri=refuri, **self.options)
        return [node], []


class IsaacLabCloneHttps(SphinxDirective):
    """Render an HTTPS clone command as a copy-friendly ``code-block``."""

    has_content = False
    option_spec = {"platform": directives.unchanged_required}

    def run(self) -> list[nodes.Node]:
        platform = self.options.get("platform", "linux").strip().lower()
        if platform not in {"linux", "windows"}:
            raise self.error(f"Unsupported platform '{platform}'. Use 'linux' or 'windows'.")

        branch = _branch(self.config)
        language = "batch" if platform == "windows" else "bash"
        content = f"""\
.. code-block:: {language}

   git clone https://github.com/isaac-sim/IsaacLab.git --branch {branch}
   cd IsaacLab
"""
        return _parse_rst(self, content)


class IsaacLabIsaacSimInstall(SphinxDirective):
    """Render the Isaac Sim install command used by historical documentation versions."""

    optional_arguments = 1  # installer: "pip" (default is "uv pip")

    def run(self) -> list[nodes.Node]:
        version = self.config.isaacsim_version
        if self.arguments and self.arguments[0] == "pip":
            content = f"""\
.. code-block:: bash

   python -m pip install "isaacsim[all,extscache]=={version}" --extra-index-url https://pypi.nvidia.com
"""
        else:
            content = f"""\
.. code-block:: bash

   uv pip install "isaacsim[all,extscache]=={version}" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow
"""
        return _parse_rst(self, content)


class IsaacLabUvIsaacSimWheelInstall(SphinxDirective):
    """Render the Isaac Lab wheel command for the current Isaac Sim version."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        branch = self.config.isaaclab_wheel_source_tag
        overrides_url = (
            f"https://raw.githubusercontent.com/isaac-sim/IsaacLab/{branch}/tools/wheel_builder/uv-overrides.txt"
        )
        content = f"""\
.. code-block:: bash

   uv pip install "isaaclab[isaacsim]=={self.config.isaaclab_wheel_version}" \\
     --overrides "{overrides_url}" \\
     --extra-index-url https://pypi.nvidia.com \\
     --torch-backend cu130 \\
     --index-strategy unsafe-best-match
"""
        return _parse_rst(self, content)


class IsaacLabUvImportersWheelInstall(SphinxDirective):
    """Render the Isaac Lab standalone importer command with resolver overrides."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        branch = self.config.isaaclab_wheel_source_tag
        overrides_url = (
            f"https://raw.githubusercontent.com/isaac-sim/IsaacLab/{branch}/tools/wheel_builder/uv-overrides.txt"
        )
        content = f"""\
.. code-block:: bash

   uv pip install "isaaclab[importers]=={self.config.isaaclab_wheel_version}" \\
     --overrides "{overrides_url}" \\
     --index https://pypi.nvidia.com \\
     --torch-backend cu130 \\
     --index-strategy unsafe-best-match
"""
        return _parse_rst(self, content)


class IsaacLabTorchInstall(SphinxDirective):
    """Render the pinned ``torch``/``torchvision`` install command for a CUDA build.

    Versions come from ``[tool.isaaclab.versions]`` (the single source of truth),
    exposed via the ``torch_version`` / ``torchvision_version`` config values.

    Usage::

        .. isaaclab-torch-install:: cu130
        .. isaaclab-torch-install:: cu130 pip
    """

    required_arguments = 1  # CUDA build tag, e.g. "cu128"
    optional_arguments = 1  # installer: "pip" (default is "uv pip")

    def run(self) -> list[nodes.Node]:
        cuda_tag = self.arguments[0]
        installer = "python -m pip" if len(self.arguments) > 1 and self.arguments[1] == "pip" else "uv pip"
        torch_version = self.config.torch_version
        torchvision_version = self.config.torchvision_version
        content = f"""\
.. code-block:: bash

   {installer} install -U torch=={torch_version} torchvision=={torchvision_version} --index-url https://download.pytorch.org/whl/{cuda_tag}
"""
        return _parse_rst(self, content)


class IsaacLabOvrtxInstall(SphinxDirective):
    """Render the ``pip install ovrtx`` command pinned to the pyproject spec.

    The spec comes from ``[tool.isaaclab.versions].ovrtx``, exposed via the
    ``ovrtx_spec`` config value.
    """

    has_content = False

    def run(self) -> list[nodes.Node]:
        spec = self.config.ovrtx_spec
        content = f"""\
.. code-block:: bash

   uv pip install "ovrtx{spec}"
"""
        return _parse_rst(self, content)


def _write_doc_redirects(app: Sphinx, exception: Exception | None) -> None:
    """Preserve old HTML URLs without retaining duplicate guide sources."""
    if exception is not None or app.builder.format != "html":
        return
    is_multiversion_build = bool(getattr(app.config, "smv_current_version", ""))
    for old, new in app.config.isaaclab_doc_redirects.items():
        output = Path(app.builder.get_outfilename(old))
        if is_multiversion_build and output.is_file():
            # Preserve pages built from historical refs instead of replacing them with current redirects.
            continue
        if new.startswith(("https://", "http://")):
            target = new
        else:
            destination = Path(app.builder.get_outfilename(new))
            if not destination.is_file():
                # The current config is also used to build tags that predate these redirect targets.
                if is_multiversion_build:
                    continue
                raise ValueError(f"Documentation redirect target was not built: {new}")
            target = posixpath.relpath(destination.as_posix(), output.parent.as_posix())
        sections = {}
        for fragment, route in getattr(app.config, "isaaclab_doc_redirect_fragments", {}).get(old, {}).items():
            doc, separator, anchor = route.partition("#")
            page = Path(app.builder.get_outfilename(doc))
            if not page.is_file():
                if is_multiversion_build:
                    continue
                raise ValueError(f"Documentation redirect target was not built: {doc}")
            sections[f"#{fragment}" if fragment else ""] = [
                posixpath.relpath(page.as_posix(), output.parent.as_posix()),
                separator + anchor,
            ]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            '<!doctype html><meta charset="utf-8"><title>Page moved</title>'
            f'<noscript><meta http-equiv="refresh" content="0; url={escape(target, quote=True)}"></noscript>'
            f'<link rel="canonical" href="{escape(target, quote=True)}">'
            f"<script>const sections = {json.dumps(sections)};\n"
            f"const target = sections[location.hash] || [{json.dumps(target)}, location.hash];\n"
            "location.replace(target[0] + location.search + target[1]);</script>"
            f'<p>This page moved to <a href="{escape(target, quote=True)}">{escape(new)}</a>.</p>',
            encoding="utf-8",
        )


def setup(app):
    """Register Isaac Lab documentation directives."""
    app.add_builder(IsaacLabHTMLBuilder, override=True)
    app.add_config_value("isaaclab_doc_redirects", {}, "html")
    app.add_config_value("isaaclab_doc_redirect_fragments", {}, "html")
    app.add_config_value("isaaclab_demo_base_url", "", "env")
    app.connect("build-finished", _write_doc_redirects)
    app.connect("config-inited", _configure_source_links)
    app.connect("html-page-context", _add_demo_assets)
    app.add_config_value("isaaclab_latest_branch", "main", "env")
    app.add_config_value("isaaclab_wheel_version", "", "env")
    app.add_config_value("isaaclab_wheel_source_tag", "", "env")
    app.add_config_value("isaacsim_version", "", "env")
    app.add_config_value("torch_version", "", "env")
    app.add_config_value("torchvision_version", "", "env")
    app.add_config_value("ovrtx_spec", "", "env")
    app.add_role("isaaclab-source", IsaacLabSourceLink())
    app.add_directive("isaaclab-benchmark-data", IsaacLabBenchmarkData)
    app.add_directive("isaaclab-demos", IsaacLabDemos)
    app.add_directive("isaaclab-clone-commands", IsaacLabCloneCommands)
    app.add_directive("isaaclab-clone-https", IsaacLabCloneHttps)
    app.add_directive("isaaclab-isaacsim-install", IsaacLabIsaacSimInstall)
    app.add_directive("isaaclab-uv-isaacsim-wheel-install", IsaacLabUvIsaacSimWheelInstall)
    app.add_directive("isaaclab-uv-importers-wheel-install", IsaacLabUvImportersWheelInstall)
    app.add_directive("isaaclab-torch-install", IsaacLabTorchInstall)
    app.add_directive("isaaclab-ovrtx-install", IsaacLabOvrtxInstall)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
