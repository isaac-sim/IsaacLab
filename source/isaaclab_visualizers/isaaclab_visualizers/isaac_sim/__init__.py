# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Sim 7 viewer visualizer backend (prototype), selected with ``--viz isaac-sim``.

One OVStage is the data layer for Isaac Lab and the Isaac Sim 7 viewer. Isaac Lab owns its ordinals: it writes body
poses and visualization markers, seals, and renders the viewer's render product with OVRTX. Two ways of attaching the
viewer are prototyped (:attr:`IsaacSimBackendCfg.host`):

* ``"viewer"`` (default): Isaac Sim's ``Viewer`` driven from Isaac Lab's loop, as in Isaac Sim's
  ``viewer.viewport_direct`` example. One navigable viewport, no scheduler.
* ``"app"``: the Isaac Sim 7 application host, adding the stage tree, property and content panes and transport controls.

Prototype requirements, not yet captured by the package dependencies:

* Isaac Sim 7 ``7.0.0a2.dev55385`` wheels (including ``isaacsim-rendering-ovgl`` and ``isaacsim-rendering-ovrtx``),
  with ``ovstage==0.4.0.1793102128``, ``ovphysx==0.8.0.41080602166`` and ``ovrtx==0.7.0.dev1793102128`` from the
  ``ct-omniverse-pypi-local`` index.
* For ``host="app"`` only: the ``sim_viewer`` application package from the Isaac Sim source tree on ``PYTHONPATH``,
  and a built ``sim_viewer`` client given by :attr:`IsaacSimBackendCfg.app_client_root` or
  ``ISAACLAB_ISAAC_SIM_APP_CLIENT_ROOT``.

This package keeps imports lazy so configuration-only imports avoid pulling the
Isaac Sim 7 libraries before they are needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .isaac_sim_visualizer_cfg import IsaacSimBackendCfg

if TYPE_CHECKING:
    from .isaac_sim_visualizer import IsaacSimVisualizer

__all__ = ["IsaacSimBackendCfg", "IsaacSimVisualizer"]


def __getattr__(name: str):
    if name == "IsaacSimVisualizer":
        from .isaac_sim_visualizer import IsaacSimVisualizer

        return IsaacSimVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
