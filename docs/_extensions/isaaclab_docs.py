# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sphinx helpers for Isaac Lab documentation."""

from __future__ import annotations

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective


def _branch(config) -> str:
    """Return the branch or tag pinned in installation docs."""
    current_version = getattr(config, "smv_current_version", "")
    if current_version:
        return current_version
    return getattr(config, "isaaclab_latest_branch", "main")


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
            content = _quickstart_isaacsim(branch, platform)
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


def _quickstart_isaacsim(branch: str, platform: str) -> str:
    """Return quickstart reST for full Isaac Sim installation."""
    if platform == "linux":
        return f"""\
.. code-block:: bash

   git clone https://github.com/isaac-sim/IsaacLab.git --branch {branch}
   cd IsaacLab

   uv venv --python 3.12 --seed env_isaaclab
   source env_isaaclab/bin/activate
   uv pip install --upgrade pip
   uv pip install "isaacsim[all,extscache]==6.0.0.1" \\
     --extra-index-url https://pypi.nvidia.com \\
     --index-strategy unsafe-best-match --prerelease=allow
   uv pip install -U torch==2.10.0 torchvision==0.25.0 \\
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
   uv pip install "isaacsim[all,extscache]==6.0.0.1" ^
     --extra-index-url https://pypi.nvidia.com ^
     --index-strategy unsafe-best-match --prerelease=allow
   uv pip install -U torch==2.10.0 torchvision==0.25.0 ^
     --index-url https://download.pytorch.org/whl/cu128
   isaaclab.bat -i
"""


def setup(app):
    """Register Isaac Lab documentation directives."""
    app.add_config_value("isaaclab_latest_branch", "develop", "env")
    app.add_directive("isaaclab-clone-commands", IsaacLabCloneCommands)
    app.add_directive("isaaclab-clone-https", IsaacLabCloneHttps)
    app.add_directive("isaaclab-kitless-install-snippet", IsaacLabKitlessInstallSnippet)
    app.add_directive("isaaclab-quickstart-install", IsaacLabQuickstartInstall)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
