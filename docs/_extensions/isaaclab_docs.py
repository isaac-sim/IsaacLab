# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sphinx helpers for Isaac Lab documentation."""

from __future__ import annotations

from docutils import nodes
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


def setup(app):
    """Register Isaac Lab documentation directives."""
    app.add_config_value("isaaclab_latest_branch", "main", "env")
    app.add_directive("isaaclab-clone-commands", IsaacLabCloneCommands)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
