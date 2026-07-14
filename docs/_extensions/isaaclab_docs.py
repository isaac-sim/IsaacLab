# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sphinx helpers for Isaac Lab documentation."""

from __future__ import annotations

import re

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective
from sphinx.util.docutils import SphinxRole
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


def _parse_rst(directive: SphinxDirective, content: str) -> list[nodes.Node]:
    """Parse nested reST and return the generated document nodes."""
    source = directive.env.doc2path(directive.env.docname, base=False)
    lines = StringList(content.splitlines(), source=source)
    container = nodes.container()
    directive.state.nested_parse(lines, 0, container)
    return container.children


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

    def run(self) -> list[nodes.Node]:
        branch = _branch(self.config)
        content = f"""\
.. code-block:: bash

   git clone https://github.com/isaac-sim/IsaacLab.git --branch {branch}
   cd IsaacLab
"""
        return _parse_rst(self, content)


class IsaacLabKitlessInstallSnippet(SphinxDirective):
    """Render the kit-less clone + install commands from the installation index."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        branch = _branch(self.config)
        content = f"""\
.. code-block:: bash

   git clone https://github.com/isaac-sim/IsaacLab.git --branch {branch}
   cd IsaacLab
   ./isaaclab.sh --install   # or ./isaaclab.sh -i
"""
        return _parse_rst(self, content)


class IsaacLabQuickstartInstall(SphinxDirective):
    """Render quickstart install snippets with the pinned release branch."""

    option_spec = {
        "kitless": directives.flag,
        "isaacsim": directives.flag,
        "platform": directives.unchanged_required,
    }

    def run(self) -> list[nodes.Node]:
        branch = _branch(self.config)
        platform = self.options["platform"].strip().lower()
        if platform not in {"linux", "windows"}:
            raise self.error(f"Unsupported platform '{platform}'. Use 'linux' or 'windows'.")

        if "kitless" in self.options and "isaacsim" in self.options:
            raise self.error("Specify only one of :kitless: or :isaacsim:.")

        if "kitless" in self.options:
            content = _quickstart_kitless(branch, platform)
        elif "isaacsim" in self.options:
            content = _quickstart_isaacsim(
                branch,
                platform,
                self.config.isaacsim_version,
                self.config.torch_version,
                self.config.torchvision_version,
            )
        else:
            raise self.error("Specify either :kitless: or :isaacsim:.")

        return _parse_rst(self, content)


def _quickstart_kitless(branch: str, platform: str) -> str:
    """Return quickstart reST for kit-less installation."""
    if platform == "linux":
        return f"""\
.. code-block:: bash

   # Install uv (https://docs.astral.sh/uv/getting-started/installation/)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   git clone https://github.com/isaac-sim/IsaacLab.git --branch {branch}
   cd IsaacLab

   uv venv --python 3.12 --seed env_isaaclab
   source env_isaaclab/bin/activate
   ./isaaclab.sh -i
"""
    return f"""\
.. code-block:: batch

   :: Install uv: https://docs.astral.sh/uv/getting-started/installation/

   git clone https://github.com/isaac-sim/IsaacLab.git --branch {branch}
   cd IsaacLab

   uv venv --python 3.12 --seed env_isaaclab
   env_isaaclab\\Scripts\\activate
   isaaclab.bat -i
"""


class IsaacLabIsaacSimInstall(SphinxDirective):
    """Render the ``uv pip install isaacsim`` command pinned to the pyproject version."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        version = self.config.isaacsim_version
        content = f"""\
.. code-block:: bash

   uv pip install "isaacsim[all,extscache]=={version}" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow
"""
        return _parse_rst(self, content)


class IsaacLabTorchInstall(SphinxDirective):
    """Render the pinned ``torch``/``torchvision`` install command for a CUDA build.

    Versions come from ``[tool.isaaclab.versions]`` (the single source of truth),
    exposed via the ``torch_version`` / ``torchvision_version`` config values.

    Usage::

        .. isaaclab-torch-install:: cu128
        .. isaaclab-torch-install:: cu130 pip
    """

    required_arguments = 1  # CUDA build tag, e.g. "cu128"
    optional_arguments = 1  # installer: "pip" (default is "uv pip")

    def run(self) -> list[nodes.Node]:
        cuda_tag = self.arguments[0]
        installer = "pip" if len(self.arguments) > 1 and self.arguments[1] == "pip" else "uv pip"
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

   pip install --extra-index-url https://pypi.nvidia.com "ovrtx{spec}"
"""
        return _parse_rst(self, content)


def _quickstart_isaacsim(branch: str, platform: str, isaacsim_version: str, torch_version: str, torchvision_version: str) -> str:
    """Return quickstart reST for full Isaac Sim installation."""
    if platform == "linux":
        return f"""\
.. code-block:: bash

   git clone https://github.com/isaac-sim/IsaacLab.git --branch {branch}
   cd IsaacLab

   uv venv --python 3.12 --seed env_isaaclab
   source env_isaaclab/bin/activate
   uv pip install --upgrade pip
   uv pip install "isaacsim[all,extscache]=={isaacsim_version}" \\
     --extra-index-url https://pypi.nvidia.com \\
     --index-strategy unsafe-best-match --prerelease=allow
   uv pip install -U torch=={torch_version} torchvision=={torchvision_version} \\
     --index-url https://download.pytorch.org/whl/cu128
   ./isaaclab.sh -i
"""
    return f"""\
.. code-block:: batch

   :: Install uv: https://docs.astral.sh/uv/getting-started/installation/

   git clone https://github.com/isaac-sim/IsaacLab.git --branch {branch}
   cd IsaacLab

   uv venv --python 3.12 --seed env_isaaclab
   env_isaaclab\\Scripts\\activate
   uv pip install --upgrade pip
   uv pip install "isaacsim[all,extscache]=={isaacsim_version}" ^
     --extra-index-url https://pypi.nvidia.com ^
     --index-strategy unsafe-best-match --prerelease=allow
   uv pip install -U torch=={torch_version} torchvision=={torchvision_version} ^
     --index-url https://download.pytorch.org/whl/cu128
   isaaclab.bat -i
"""


def setup(app):
    """Register Isaac Lab documentation directives."""
    app.add_config_value("isaaclab_latest_branch", "develop", "env")
    app.add_config_value("isaacsim_version", "", "env")
    app.add_config_value("torch_version", "", "env")
    app.add_config_value("torchvision_version", "", "env")
    app.add_config_value("ovrtx_spec", "", "env")
    app.add_role("isaaclab-source", IsaacLabSourceLink())
    app.add_directive("isaaclab-clone-commands", IsaacLabCloneCommands)
    app.add_directive("isaaclab-clone-https", IsaacLabCloneHttps)
    app.add_directive("isaaclab-kitless-install-snippet", IsaacLabKitlessInstallSnippet)
    app.add_directive("isaaclab-quickstart-install", IsaacLabQuickstartInstall)
    app.add_directive("isaaclab-isaacsim-install", IsaacLabIsaacSimInstall)
    app.add_directive("isaaclab-torch-install", IsaacLabTorchInstall)
    app.add_directive("isaaclab-ovrtx-install", IsaacLabOvrtxInstall)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
