# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sphinx helpers for Isaac Lab documentation."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

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
    """Render the Isaac Sim install command pinned to the pyproject version."""

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
        branch = _source_branch(self.config)
        overrides_url = (
            f"https://raw.githubusercontent.com/isaac-sim/IsaacLab/{branch}/tools/wheel_builder/uv-overrides.txt"
        )
        content = f"""\
.. code-block:: bash

   uv pip install "isaaclab[isaacsim]" \\
     --overrides "{overrides_url}" \\
     --extra-index-url https://pypi.nvidia.com \\
     --index-strategy unsafe-best-match
"""
        return _parse_rst(self, content)


class IsaacLabUvImportersWheelInstall(SphinxDirective):
    """Render the Isaac Lab standalone importer command with resolver overrides."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        branch = _source_branch(self.config)
        overrides_url = (
            f"https://raw.githubusercontent.com/isaac-sim/IsaacLab/{branch}/tools/wheel_builder/uv-overrides.txt"
        )
        content = f"""\
.. code-block:: bash

   uv pip install "isaaclab[importers]" \\
     --overrides "{overrides_url}"
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

   pip install "ovrtx{spec}"
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


def _container_parser(srcdir) -> argparse.ArgumentParser:
    """Import ``docker/container.py`` and return its argument parser.

    The module is loaded by path because ``docker`` is not an installed package. Its ``utils``
    package only imports from the standard library, so this is safe during a documentation build.
    """
    docker_dir = Path(srcdir).parent / "docker"
    if str(docker_dir) not in sys.path:
        sys.path.insert(0, str(docker_dir))
    spec = importlib.util.spec_from_file_location("isaaclab_docker_container", docker_dir / "container.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_parser()


def _squash(text: str) -> str:
    """Collapse an argparse help string onto a single line."""
    return " ".join((text or "").split())


def _subparsers_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction | None:
    """Return the sub-command action, relying on argparse internals as sphinx-argparse does."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _list_table(header: str, rows: list[tuple[str, str]]) -> str:
    """Render name/description pairs as a two-column list table."""
    lines = [".. list-table::", "   :header-rows: 1", "   :widths: 25 75", "", f"   * - {header}", "     - Description"]
    for name, description in rows:
        lines.extend([f"   * - {name}", f"     - {description}"])
    return "\n".join(lines) + "\n"


class IsaacLabContainerCli(SphinxDirective):
    """Render the ``docker/container.py`` reference from its ``argparse`` definition.

    Importing the parser keeps the published tables from drifting away from the CLI. Select the
    table with ``:section: commands`` (default) or ``:section: options``.
    """

    has_content = False
    option_spec = {"section": directives.unchanged}

    def run(self) -> list[nodes.Node]:
        section = self.options.get("section", "commands")
        if section not in ("commands", "options"):
            raise self.error(f"Unknown section '{section}' for isaaclab-container-cli. Use 'commands' or 'options'.")

        action = _subparsers_action(_container_parser(self.env.srcdir))
        if action is None:
            raise self.error("Could not find sub-commands in docker/container.py.")

        if section == "commands":
            header = "Command"
            rows = [(f"``{choice.dest}``", _squash(choice.help)) for choice in action._choices_actions]
        else:
            # Every sub-command inherits the same parent parser, so one of them describes them all.
            header = "Argument"
            rows = [
                (", ".join(f"``{flag}``" for flag in arg.option_strings) or f"``{arg.dest}``", _squash(arg.help))
                for arg in action.choices["start"]._actions
                if arg.dest != "help"
            ]
        return _parse_rst(self, _list_table(header, rows))


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
    app.add_directive("isaaclab-uv-isaacsim-wheel-install", IsaacLabUvIsaacSimWheelInstall)
    app.add_directive("isaaclab-uv-importers-wheel-install", IsaacLabUvImportersWheelInstall)
    app.add_directive("isaaclab-torch-install", IsaacLabTorchInstall)
    app.add_directive("isaaclab-ovrtx-install", IsaacLabOvrtxInstall)
    app.add_directive("isaaclab-container-cli", IsaacLabContainerCli)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
