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

    Isaac Lab keeps its own physics and scene data. The Isaac Sim 7 application host renders an export of the
    authored stage and receives Isaac Lab's body poses every step, so no Isaac Sim physics engine is involved.
    The shared :class:`~isaaclab.visualizers.visualizer_cfg.VisualizerCfg` supplies the viewport size
    (``window.size``), the host rate (``window.fps``) and ``headless``, which serves the viewer without a client.
    """

    class_type: type[IsaacSimVisualizer] | str = "{DIR}.isaac_sim_visualizer:IsaacSimVisualizer"
    """Visualizer implementation class."""

    visualizer_type: str = "isaac_sim"
    """Type identifier for the Isaac Sim 7 viewer."""

    cloning_contexts: ClassVar[tuple[type | str, ...]] = ("isaaclab.cloner:UsdReplicateContext",)
    """The viewer scene is exported from the authored USD stage, so every environment is authored there."""

    ui: Literal["desktop", "browser"] = "desktop"
    """Where the viewer is presented: the Electron window attached to the host, or a logged browser URL."""

    presentation: Literal["auto", "gpu", "http"] = "http"
    """How the desktop client receives frames.

    ``"http"`` sends encoded CPU images. The GPU texture path fails with OVGL in the attached setup ("published CPU
    pixels without a native texture"), so it is not the default.
    """

    port: int = 0
    """Port of the host's HTTP service; zero picks a free one."""

    stage_mode: Literal["export", "shared"] = "shared"
    """Which stage the viewer shows.

    * ``"export"``: the host builds its own stage from an export of Isaac Lab's stage and renders it with OVGL; poses
      reach it through :attr:`pose_transport`.
    * ``"shared"``: one OVStage, created by the host's stage context and borrowed by Isaac Lab, is the data layer for
      both. Isaac Lab owns its ordinals: it writes poses and the host's edits, seals, and renders the viewer's
      render products with OVRTX at most ``window.fps`` times a second. The host runs without its orchestration.
    """

    pose_transport: Literal["participant", "commands"] = "participant"
    """How Isaac Lab's body poses reach the host's stage.

    * ``"participant"``: a pose engine registered with the host writes world matrices and a transform journal at the
      host's output ordinal, before rendering. The OVGL renderer then refreshes only the journaled bodies.
    * ``"commands"``: the host's command queue writes local matrices with ``omni:resetXformStack``.
    """

    app_client_root: str | None = None
    """Directory with the built ``sim_viewer`` application client: ``renderer/`` (``index.js``, ``manifest.js``)
    and, for ``ui="desktop"``, ``main.cjs``.

    The renderer supplies the stage, property and content panes. ``None`` uses the
    ``ISAACLAB_ISAAC_SIM_APP_CLIENT_ROOT`` environment variable, or serves only the shared shell when it is unset.
    """

    deinstance_max_points: int = 5_000_000
    """Expand USD instanceable prims in the exported scene while it has at most this many mesh points.

    The viewer's renderer re-resolves instanced geometry whenever poses change, which makes per-frame pose updates
    about 2.5x slower on instanced assets such as the cartpole. Expanding copies every instance's geometry, so
    scenes with heavy assets keep their instancing. Use ``0`` to never expand.
    """
