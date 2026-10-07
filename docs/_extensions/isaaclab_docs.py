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
from html import escape
import re
import sys
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.util.docutils import SphinxDirective, SphinxRole
from sphinx.util.nodes import split_explicit_title

_UPSTREAM_SOURCE_REF_PATTERN = re.compile(r"^(main|develop|release/.*|v[1-9]\d*\.\d+\.\d+(-[A-Za-z0-9.]+)?)$")


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
    return getattr(config, "isaaclab_latest_branch", "develop")


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
        return f"https://github.com/isaac-sim/IsaacLab/blob/{branch}/{path.as_posix()}#L{start}-L{start + len(lines) - 1}"

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
     --index-strategy unsafe-best-match
"""
        return _parse_rst(self, content)


class IsaacLabTorchInstall(SphinxDirective):
    """Render the pinned ``torch``/``torchvision`` install command for a CUDA build.

    Versions come from ``[tool.isaaclab.versions]`` (the single source of truth),
    exposed via the ``torch_version`` / ``torchvision_version`` config values.

    Usage::

        .. isaaclab-torch-install:: cu130
    """

    required_arguments = 1  # CUDA build tag, e.g. "cu128"

    def run(self) -> list[nodes.Node]:
        cuda_tag = self.arguments[0]
        torch_version = self.config.torch_version
        torchvision_version = self.config.torchvision_version
        content = f"""\
.. code-block:: bash

   uv pip install -U torch=={torch_version} torchvision=={torchvision_version} --index-url https://download.pytorch.org/whl/{cuda_tag}
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
        destination = Path(app.builder.get_outfilename(new))
        if not destination.is_file():
            # The current config is also used to build tags that predate these redirect targets.
            if is_multiversion_build:
                continue
            raise ValueError(f"Documentation redirect target was not built: {new}")
        output = Path(app.builder.get_outfilename(old))
        target = posixpath.relpath(destination.as_posix(), output.parent.as_posix())
        sections = {}
        for fragment, route in getattr(app.config, "isaaclab_doc_redirect_fragments", {}).get(old, {}).items():
            doc, separator, anchor = route.partition("#")
            page = Path(app.builder.get_outfilename(doc))
            if not page.is_file():
                raise ValueError(f"Documentation redirect target was not built: {doc}")
            sections[f"#{fragment}"] = [
                posixpath.relpath(page.as_posix(), output.parent.as_posix()), separator + anchor
            ]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            '<!doctype html><meta charset="utf-8"><title>Page moved</title>'
            f'<noscript><meta http-equiv="refresh" content="0; url={escape(target, quote=True)}"></noscript>'
            f'<link rel="canonical" href="{escape(target, quote=True)}">'
            f'<script>const sections = {json.dumps(sections)};\n'
            f'const target = sections[location.hash] || [{json.dumps(target)}, location.hash];\n'
            'location.replace(target[0] + location.search + target[1]);</script>'
            f'<p>This page moved to <a href="{escape(target, quote=True)}">{escape(new)}</a>.</p>',
            encoding="utf-8",
        )


def setup(app):
    """Register Isaac Lab documentation directives."""
    app.add_config_value("isaaclab_doc_redirects", {}, "html")
    app.add_config_value("isaaclab_doc_redirect_fragments", {}, "html")
    app.connect("build-finished", _write_doc_redirects)
    app.connect("config-inited", _configure_source_links)
    app.add_config_value("isaaclab_latest_branch", "develop", "env")
    app.add_config_value("isaaclab_wheel_version", "", "env")
    app.add_config_value("isaaclab_wheel_source_tag", "", "env")
    app.add_config_value("isaacsim_version", "", "env")
    app.add_config_value("torch_version", "", "env")
    app.add_config_value("torchvision_version", "", "env")
    app.add_config_value("ovrtx_spec", "", "env")
    app.add_role("isaaclab-source", IsaacLabSourceLink())
    app.add_directive("isaaclab-benchmark-data", IsaacLabBenchmarkData)
    app.add_directive("isaaclab-clone-commands", IsaacLabCloneCommands)
    app.add_directive("isaaclab-clone-https", IsaacLabCloneHttps)
    app.add_directive("isaaclab-uv-isaacsim-wheel-install", IsaacLabUvIsaacSimWheelInstall)
    app.add_directive("isaaclab-uv-importers-wheel-install", IsaacLabUvImportersWheelInstall)
    app.add_directive("isaaclab-torch-install", IsaacLabTorchInstall)
    app.add_directive("isaaclab-ovrtx-install", IsaacLabOvrtxInstall)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
