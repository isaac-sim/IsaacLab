# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Isaac Sim 7 viewer visualizer."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from isaaclab.utils import configclass
from isaaclab.visualizers.visualizer_cfg import VisualizerBackendCfg

if TYPE_CHECKING:
    from .isaac_sim_visualizer import IsaacSimVisualizer


@configclass
class IsaacSimBackendCfg(VisualizerBackendCfg):
    """Configuration for the Isaac Sim 7 viewer attached to an Isaac Lab simulation.

    Isaac Lab keeps its own physics. One OVStage, created by the viewer host's stage context and borrowed by Isaac Lab,
    is the data layer for both: Isaac Lab owns its ordinals, writes poses, markers and the host's edits, seals, and
    renders the viewer's products with OVRTX at most ``window.fps`` times a second. The shared
    :class:`~isaaclab.visualizers.visualizer_cfg.VisualizerCfg` supplies the viewport size (``window.size``), the frame
    rate (``window.fps``) and ``headless``, which serves the viewer without a client.
    """

    class_type: type[IsaacSimVisualizer] | str = "{DIR}.isaac_sim_visualizer:IsaacSimVisualizer"
    """Visualizer implementation class."""

    visualizer_type: str = "isaac_sim"
    """Type identifier for the Isaac Sim 7 viewer."""

    cloning_contexts: ClassVar[tuple[type | str, ...]] = ("isaaclab.cloner:UsdReplicateContext",)
    """The shared stage is populated from the authored USD stage, so every environment is authored there."""

    camera_controls: Literal["newton", "isaac_sim"] = "newton"
    """Viewport camera bindings.

    ``"newton"`` matches Isaac Lab's Newton viewer: left-drag looks, middle-drag orbits, Shift+middle-drag pans and
    Ctrl+middle-drag dollies; a left click still selects. ``"isaac_sim"`` keeps the client's Kit-style bindings:
    Alt+left-drag orbits, middle-drag pans and right-drag looks. See :mod:`.isaac_sim_camera_controls`.
    """

    ui: Literal["desktop", "browser"] = "desktop"
    """Where the viewer is presented: the Electron window attached to the host, or a logged browser URL."""

    presentation: Literal["auto", "gpu", "http"] = "http"
    """How the desktop client receives frames. ``"http"`` sends encoded CPU images, which Isaac Lab's frames are."""

    port: int = 0
    """Port of the host's HTTP service; zero picks a free one."""

    app_client_root: str | None = None
    """Directory with the built ``sim_viewer`` application client: ``renderer/`` (``index.js``,
    ``manifest.js``) and, for ``ui="desktop"``, ``main.cjs``.

    The renderer supplies the stage, property and content panes. ``None`` uses the
    ``ISAACLAB_ISAAC_SIM_APP_CLIENT_ROOT`` environment variable, or serves only the shared shell when it is unset.
    """
