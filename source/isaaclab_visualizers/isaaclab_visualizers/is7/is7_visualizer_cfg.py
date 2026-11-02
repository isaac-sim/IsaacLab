# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Isaac Sim 7 viewer visualizer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab.utils import configclass
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg

if TYPE_CHECKING:
    from .is7_visualizer import IS7Visualizer


@configclass
class IS7VisualizerCfg(VisualizerCfg):
    """Configuration for the Isaac Sim 7 viewer attached to an Isaac Lab simulation.

    Isaac Lab keeps its own physics and scene data. Poses are written into an OVStage that the
    Isaac Sim 7 viewer renders, so no Isaac Sim physics engine is involved.
    """

    class_type: type[IS7Visualizer] | str = "{DIR}.is7_visualizer:IS7Visualizer"
    """Visualizer implementation class."""

    visualizer_type: str = "is7"
    """Type identifier for the Isaac Sim 7 viewer."""

    cloning_contexts: tuple[type | str, ...] = ("isaaclab.cloner:UsdReplicateContext",)
    """The viewer scene is exported from the authored USD stage, so every environment is authored there."""

    ui: Literal["desktop", "browser", "headless"] = "desktop"
    """Where the viewer is presented.

    * ``"desktop"``: open the Electron window attached to the host.
    * ``"browser"``: serve the client and log a URL to open.
    * ``"headless"``: serve without opening a client, for tests and benchmarks.
    """

    resolution: tuple[int, int] = (1280, 720)
    """Initial render size of a viewport pane in pixels."""

    viewport_hz: float = 60.0
    """Rate of the host's rendering and stepping [Hz]."""

    presentation: Literal["auto", "gpu", "http"] = "http"
    """How the desktop client receives frames.

    ``"http"`` sends encoded CPU images. The GPU texture path fails with OVGL in the attached setup ("published CPU
    pixels without a native texture"), so it is not the default.
    """

    port: int = 0
    """Port of the host's HTTP service; zero picks a free one."""

    app_client_root: str | None = None
    """Directory with the built ``sim_viewer`` application client: ``renderer/`` (``index.js``, ``manifest.js``)
    and, for ``ui="desktop"``, ``main.cjs``.

    The renderer supplies the stage, property and content panes. ``None`` serves only the shared shell.
    """

    deinstance_max_points: int = 5_000_000
    """Expand USD instanceable prims in the exported scene while it has at most this many mesh points.

    The viewer's renderer re-resolves instanced geometry whenever poses change, which makes per-frame pose updates
    about 2.5x slower on instanced assets such as the cartpole. Expanding copies every instance's geometry, so
    scenes with heavy assets keep their instancing. Use ``0`` to never expand.
    """
