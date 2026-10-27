# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for visualizers."""

from __future__ import annotations

import math
import warnings
from copy import deepcopy
from dataclasses import MISSING, fields
from typing import TYPE_CHECKING, ClassVar

from ..utils import configclass

if TYPE_CHECKING:
    from ..cloner import ReplicateContext
    from .base_visualizer import BaseVisualizer


VISUALIZER_TYPES = ("kit", "newton", "rerun", "viser")
"""Built-in visualizer identifiers accepted by legacy recording sources."""

VISUALIZER_ALIASES = {"newton_gl": "newton"}
"""Alternative ``--visualizer`` names and the identifiers they select."""


def reject_removed_visualizer(name: str) -> None:
    """Raise a :class:`ValueError` that points a removed visualizer name at its replacement.

    Args:
        name: Visualizer identifier supplied by the user.

    Raises:
        ValueError: If *name* is ``newton_rtx``, which the ``newton`` visualizer replaced.
    """
    if name == "newton_rtx":
        raise ValueError(
            "The 'newton_rtx' visualizer was removed. Use the 'newton' visualizer, which supports RTX rendering:"
            " switch the Renderer to RTX in its window, or set NewtonGLBackendCfg.initial_renderer='rtx' to start"
            " in RTX."
        )


def with_free_view(
    cameras: list[PerspectiveCameraCfg | SceneCameraCfg],
    eye: tuple[float, float, float],
    lookat: tuple[float, float, float],
    focal_length: float,
) -> list[PerspectiveCameraCfg | SceneCameraCfg]:
    """Return *cameras*, preceded by a free perspective view when it has task views but no free view of its own.

    A task view follows a robot or frames an environment, so the interactive view that a user moves stays
    available beside it, at the visualizer's own ``eye``, ``lookat`` and ``focal_length``.
    """
    perspective = [camera for camera in cameras if isinstance(camera, PerspectiveCameraCfg)]
    if perspective and all(camera.is_task_view for camera in perspective):
        return [PerspectiveCameraCfg(eye=eye, lookat=lookat, focal_length=focal_length), *cameras]
    return cameras


@configclass
class PerspectiveCameraCfg:
    """Pose and optics for a visualizer-owned perspective camera.

    By default the pose is in world coordinates. With :attr:`env_index` or :attr:`track_path` set, the view is tied
    to an environment or to an asset in it, so it frames that robot however the scene is laid out, and the
    visualizer renders it from the viewer camera. Nothing is created in the scene.
    """

    eye: tuple[float, float, float] = (4.0, -4.0, 3.0)
    """Eye position [m]: world coordinates, or an offset from the environment origin or the tracked asset."""

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Look-at target [m], in the same frame as :attr:`eye`."""

    focal_length: float = 12.0
    """Perspective camera focal length [mm]."""

    env_index: int | None = None
    """Environment whose origin :attr:`eye` and :attr:`lookat` are relative to; None keeps world coordinates.

    With :attr:`track_path` it is the environment of the tracked asset, and None means environment 0.
    """

    track_path: str | None = None
    """Scene asset the view follows, or None for a view fixed relative to the environment origin.

    Use ``"robot"`` for an asset root or ``"robot/base"`` for a named body, which must match exactly one body.
    """

    follow_heading: bool = False
    """Rotate :attr:`eye` and :attr:`lookat` with the tracked asset's yaw, keeping the horizon level.

    Without it the offsets stay aligned with the world axes. Only used with :attr:`track_path`.
    """

    heading_smoothing_time_constant: float = 0.0
    """Exponential heading-filter time constant [s]; zero follows the heading immediately.

    Larger values damp rapid turns more, with more lag. Only used with :attr:`follow_heading`.
    """

    @property
    def is_task_view(self) -> bool:
        """Whether the view is tied to an environment or an asset, not fixed in the world."""
        return self.env_index is not None or self.track_path is not None

    def __post_init__(self) -> None:
        if self.env_index is not None and self.env_index < 0:
            raise ValueError("env_index must be non-negative.")
        if not math.isfinite(self.heading_smoothing_time_constant) or self.heading_smoothing_time_constant < 0.0:
            raise ValueError("heading_smoothing_time_constant must be finite and non-negative.")


@configclass
class SceneCameraCfg:
    """Select an existing scene camera's output as the visualizer's main view.

    This does not create a sensor. Declare its pose, optics, and renderer in the scene's CameraCfg.
    """

    prim_path: str = MISSING
    """Scene camera's configured prim path, including ``{ENV_REGEX_NS}`` when applicable."""


@configclass
class ImageViewCfg:
    """A shared image selection consumed by windows and recorders.

    Reusing this declaration shares one runtime view within a simulation, including across copies
    of an environment configuration. Use :func:`isaaclab.utils.replace` to declare an independent view.
    """

    class_type: str = "{DIR}.image_view:ImageView"
    """Runtime image selection and composition."""

    source: str | PerspectiveCameraCfg = MISSING
    """Scene sensor name, or an interactive perspective camera declaration."""

    envs: tuple[int, ...] = (0,)
    """Sensor rows to display, in the requested order."""

    channels: tuple[str, ...] = ("rgb",)
    """Existing source outputs to display, left-to-right: rgb, depth, normals, or segmentation."""

    depth_range: tuple[float, float] = (0.1, 10.0)
    """Display color scale limits [m]. Source depth values and camera clipping planes remain unchanged."""

    def __eq__(self, other) -> bool:
        """Share explicit declarations; equal initial poses can belong to independent cameras."""
        return self is other

    def __deepcopy__(self, memo):
        """Keep this explicitly shared declaration when its consumers' configurations are copied."""
        return self

    def validate_config(self) -> None:
        """Reject unsupported channels, negative row indices, and invalid display color limits."""
        if not isinstance(self.source, (str, PerspectiveCameraCfg)) or self.source == "":
            raise ValueError("ImageViewCfg.source must name a scene camera or declare a perspective camera.")
        if any(not isinstance(i, int) or i < 0 for i in self.envs):
            raise ValueError("ImageViewCfg.envs must contain non-negative sensor row indices.")
        if not self.channels:
            raise ValueError("ImageViewCfg.channels must contain at least one display channel.")
        for channel in self.channels:
            if channel not in ("rgb", "depth", "normals", "segmentation"):
                raise ValueError(f"Unsupported image channel: {channel!r}.")
        if isinstance(self.source, PerspectiveCameraCfg) and (self.envs != (0,) or self.channels != ("rgb",)):
            raise ValueError("Perspective image views provide one RGB image; use envs=(0,) and channels=('rgb',).")
        if len(self.depth_range) != 2 or not 0 <= self.depth_range[0] < self.depth_range[1]:
            raise ValueError("ImageViewCfg.depth_range must contain increasing non-negative limits.")


@configclass
class WindowCfg:
    """Native window presentation, independent of the camera's image resolution."""

    size: tuple[int, int] = (1920, 1080)
    """Initial window width and height [px]."""

    fps: float = 30.0
    """Maximum Newton window update rate [Hz], measured in wall-clock time.

    Headless on-demand recording is independent of this limit. Kit owns its application update cadence.
    """

    def validate_config(self) -> None:
        """Require positive dimensions and a finite, positive presentation rate."""
        if len(self.size) != 2 or any(not isinstance(value, int) or value < 1 for value in self.size):
            raise ValueError("WindowCfg.size must contain positive integer width and height.")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("WindowCfg.fps must be finite and positive.")


@configclass
class VisualizerBackendCfg:
    """Implementation and backend-specific options for a visualizer."""

    class_type: type[BaseVisualizer] | str | None = None
    """Visualizer implementation class. Concrete configs must set this field."""

    cloning_contexts: ClassVar[tuple[type[ReplicateContext] | str, ...]] = ()
    """Clone contexts that build this visualizer's scene representation from the asset plan."""

    launcher_type: ClassVar[str | None] = None
    """Process runtime required by this visualizer, if any."""

    visualizer_type: str | None = None
    """Backend selector identifier, supplied by concrete backend configurations."""


@configclass
class VisualizerCfg:
    """View, window, and scene settings independent of the selected visualizer backend."""

    backend: VisualizerBackendCfg | list[VisualizerBackendCfg] = []
    """Selected implementation(s); an empty list keeps the settings without opening a viewer.

    Each declaration expands independently: two views with two selected backends create four viewers.
    """

    view: ImageViewCfg | None = None
    """Shared image view, or None to use the visualizer's camera selection."""

    window: WindowCfg = WindowCfg()
    """Native window settings used by Newton and Kit; network visualizers manage their own presentation."""

    headless: bool = False
    """Render on demand without opening a native viewer window."""

    # Primary interactive camera settings
    cameras: list[PerspectiveCameraCfg | SceneCameraCfg] | None = None
    """Camera sources available to the visualizer.

    PerspectiveCameraCfg configures the interactive view; SceneCameraCfg refers to an existing scene
    sensor. Newton GL and RTX display the first source and offer a dropdown for switching sources.
    Kit, Rerun, and Viser display the first scene source in their camera panel.
    None uses eye/lookat/focal_length and the streaming settings below.
    Every explicit scene source must provide all requested streaming_gt_types channels.
    """

    eye: tuple[float, float, float] = (4.0, -4.0, 3.0)
    """Interactive visualizer camera eye position in world coordinates."""

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Interactive visualizer camera look-at target in world coordinates."""

    focal_length: float = 12.0
    """Camera focal length in millimeters for visualizer camera views."""

    background_color: tuple[float, float, float] | None = None
    """Viewer perspective-camera background as normalized RGB values in ``[0, 1]``.

    None preserves the scene HDR background in Kit and Newton RTX, or Newton GL's procedural sky.
    An explicit color changes only the visible background, not scene lighting or reflections.
    Scene-camera images retain their sensor's :attr:`~isaaclab.sensors.CameraCfg.background_color`.
    """

    # ── Streaming view ────────────────────────────────────────────────────────
    # Reads pixels from a scene-declared camera sensor, tiles them
    # across envs and GT types, and shows the result as an image panel in interactive
    # visualizers (Newton GL, Kit) or pushes it per-step to sink-based ones (Rerun, Viser).

    streaming_view: bool = False
    """Enable the streaming camera image view (opt-in, disabled by default)."""

    streaming_sensor_prim_path: str | None = None
    """Prim path of a scene-declared :class:`~isaaclab.sensors.Camera` to display.

    Use the camera's configured prim path, including ``{ENV_REGEX_NS}`` when applicable.
    None selects the first compatible camera in the scene. Without one the panel stays empty.
    The scene owns camera construction, updates, and lifetime; visualizers only read its output.
    """

    # Shared settings
    streaming_envs: int | list[int] = 32
    """Environments to display, limited to 100 tiles.

    * ``int`` — sample this many envs once at initialization (from all visible envs).
    * ``list[int]`` — display exactly these env indices.
    """

    streaming_gt_types: tuple[str, ...] = ("rgb",)
    """GT data types displayed left-to-right per environment row.

    Valid values: ``"rgb"``, ``"depth"``, ``"segmentation"``, ``"normals"``.
    Validated at initialization when :attr:`streaming_view` is ``True``.
    """

    streaming_depth_min: float = 0.1
    """Near-clip for the turbo depth colormap [m].  Used when ``"depth"`` is in
    :attr:`streaming_gt_types`."""

    streaming_depth_max: float = 10.0
    """Far-clip for the turbo depth colormap [m].  Used when ``"depth"`` is in
    :attr:`streaming_gt_types`."""

    # Partial visualization settings
    max_visible_envs: int | None = None
    """Upper bound on how many envs are shown.

    * If visible_env_indices is not None, then this field will apply also
      to the explicit env indices set to the visible_env_indices.
    """

    visible_env_indices: list[int] | None = None
    """env indices to visualize in order (out-of-range indices are dropped)."""

    randomly_sample_visible_envs: bool = True
    """If ``max_visible_envs`` is provided, when enabled, selected visible envs are randomly sampled.
       If disabled, the first ``max_visible_envs`` envs are selected.

    * Note: ``visible_env_indices`` overrides this field.
    """

    # Visualization Markers
    enable_markers: bool = True
    """Enable visualization markers (debug drawing)."""

    # Live Plots
    enable_live_plots: bool = True
    """Stream per-step scalar data (manager terms, episode reward, episode length) into the visualizer.

    Plot windows start hidden or collapsed by default and can be toggled open at runtime.
    Set to ``False`` to disable live plots entirely and avoid any collection overhead.
    """

    live_plots_update_interval: int = 5
    """Collect and push live plot data every ``N`` simulation steps (default: every 5 steps)."""

    def validate_config(self) -> None:
        """Reject ambiguous sources and unsupported native-window configurations."""
        if self.view is not None:
            backends = self.backend if isinstance(self.backend, list) else [self.backend]
            for backend in backends:
                if backend.visualizer_type != "newton":
                    raise ValueError("VisualizerCfg.view requires the Newton visualizer.")
            if self.cameras is not None or self.streaming_sensor_prim_path is not None:
                raise ValueError("Choose view or legacy cameras/streaming_sensor_prim_path, not both.")

    def __post_init__(self) -> None:
        if self.background_color is not None:
            if len(self.background_color) != 3 or any(not 0.0 <= value <= 1.0 for value in self.background_color):
                raise ValueError("background_color must contain three normalized RGB values in [0, 1].")
            self.background_color = tuple(float(value) for value in self.background_color)

        if self.cameras is not None:
            if not self.cameras:
                raise ValueError("cameras must contain at least one display source.")
            # task views only add to the cameras the visualizer finds, so they leave the streaming choice alone
            if not all(isinstance(camera, PerspectiveCameraCfg) and camera.is_task_view for camera in self.cameras):
                self.streaming_view = any(isinstance(camera, SceneCameraCfg) for camera in self.cameras)
            if isinstance(camera := self.cameras[0], PerspectiveCameraCfg) and not camera.is_task_view:
                self.eye, self.lookat, self.focal_length = camera.eye, camera.lookat, camera.focal_length


def migrate_visualizer_cfgs(
    declarations: VisualizerCfg | list[VisualizerCfg] | None, defaults: VisualizerCfg | None = None
) -> list[VisualizerCfg]:
    """Translate deprecated combined declarations and shared defaults before launch.

    Args:
        declarations: Shared visualizer declarations or deprecated combined configurations.
        defaults: Deprecated simulation-wide defaults; explicit per-view settings take precedence.

    Returns:
        Shared configurations with backend-only options; current declarations retain their identity.
    """
    declarations = declarations if isinstance(declarations, list) else [declarations]
    if defaults is not None:
        warnings.warn(
            "default_visualizer_cfg is deprecated in Isaac Lab 3.2. Put shared settings on each VisualizerCfg instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    viewers = []
    for cfg in declarations:
        if cfg is None:
            cfg = VisualizerCfg()
        if defaults is not None:
            for source, target in ((defaults, cfg), (defaults.window, cfg.window)):
                source_defaults, target_defaults = type(source)(), type(target)()
                for field in fields(source):
                    if field.name in ("backend", "window"):
                        continue
                    value = getattr(source, field.name)
                    target_is_default = getattr(target, field.name) == getattr(target_defaults, field.name)
                    if value != getattr(source_defaults, field.name) and target_is_default:
                        setattr(target, field.name, deepcopy(value))
        if isinstance(cfg, VisualizerBackendCfg):
            warnings.warn(
                f"{type(cfg).__name__} is deprecated in Isaac Lab 3.2. "
                f"Use VisualizerCfg(backend={type(cfg.backend).__name__}(), ...) "
                "with shared camera and window settings on VisualizerCfg.",
                DeprecationWarning,
                stacklevel=2,
            )
            backend = type(cfg.backend)(**{field.name: getattr(cfg, field.name) for field in fields(cfg.backend)})
            cfg = VisualizerCfg(
                backend=backend,
                **{field.name: getattr(cfg, field.name) for field in fields(VisualizerCfg) if field.name != "backend"},
            )
        viewers.append(cfg)
    return viewers
