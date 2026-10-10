# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for visualizers."""

from __future__ import annotations

import logging
import math
import os
import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import numpy as np
import warp as wp

from ..envs.utils.camera_colorizer import sensor_key_for_gt_type
from ..envs.utils.camera_view import image_grid_columns, resolve_streaming_envs
from ..utils import validate
from ..utils.buffers import TimestampedBuffer
from ..utils.images import compose_image
from .visualizer_cfg import PerspectiveCameraCfg

if TYPE_CHECKING:
    from ..managers import ManagerBase
    from ..renderers.base_renderer import VisualMaterialBatch
    from ..sensors import Camera
    from ..sim import SimulationContext
    from .visualizer_cfg import VisualizerCfg


logger = logging.getLogger(__name__)

_USD_DEFAULT_VERTICAL_APERTURE_MM = 15.2908


class BaseVisualizer(ABC):
    """Base class for all visualizer backends.

    Lifecycle: __init__() -> initialize() -> step() (repeated) -> close()
    """

    def __init__(self, cfg: VisualizerCfg):
        """Initialize visualizer with config.

        Args:
            cfg: Visualizer configuration.
        """
        validate(cfg)
        self.cfg = cfg
        self._sim: SimulationContext | None = None
        self._cameras: list[PerspectiveCameraCfg | Camera] = []
        self._is_initialized = False
        self._is_closed = False
        self._env_ids: list[int] | None = None
        self._deferred_startup_messages: list[str] = []
        self._live_plot_sources: list = []
        self._live_plot_env_idx: int = 0
        self._live_plots_step_counter: int = 0
        self._reset_requested: bool = False
        self._sim_time = 0.0
        self._camera_sensor: Camera | None = None
        self._camera_sensor_indices: list[int] = []
        self._streaming_aspect = 1.0
        self._streaming_frame = TimestampedBuffer()
        self._streaming_host_frame = TimestampedBuffer()
        self._streaming_env_ids: wp.array | None = None
        self._streaming_depth_colors: wp.array | None = None
        self._streaming_layout: tuple | None = None
        self._streaming_view_key: tuple | None = None
        self._streaming_keys: tuple[str, ...] = ()

    @property
    def visual_material_writer(self) -> Callable[[tuple[VisualMaterialBatch, ...]], Any] | None:
        """Return the backend's shared material-writer factory, if supported.

        Its writer accepts ``None`` for a full sync or channel-to-material-offset device arrays plus
        one environment-id device array for partial writes, and provides an idempotent ``close()``.
        """
        return None

    def initialize(self, sim: SimulationContext, *, cameras: list[PerspectiveCameraCfg | Camera]) -> None:
        """Bind scene inputs and the simulation-owned resource registry.

        Args:
            sim: Simulation owner that provides the scene and backend resources.
            cameras: Resolved perspective settings and borrowed scene sensors, in display order.
        """
        scene_data_provider = sim.get_scene_data_provider()
        if scene_data_provider is None:
            raise RuntimeError(f"{self.__class__.__name__} requires a scene_data_provider.")
        self._sim = sim
        self._cameras = list(cameras)

        cfg = self.cfg
        num_envs = scene_data_provider.num_envs
        self._env_ids = None
        if num_envs > 0:
            count = num_envs if cfg.max_visible_envs is None else max(0, min(int(cfg.max_visible_envs), num_envs))
            if cfg.visible_env_indices is not None:
                self._env_ids = list(dict.fromkeys(i for i in cfg.visible_env_indices if 0 <= i < num_envs))[:count]
            elif cfg.max_visible_envs is not None and cfg.randomly_sample_visible_envs:
                self._env_ids = sorted(random.sample(range(num_envs), count))
            elif cfg.max_visible_envs is not None:
                self._env_ids = list(range(count))

    def _setup_streaming_view(
        self,
        num_envs: int,
        *,
        visible_env_ids: list[int] | None = None,
        target_aspect: float = 1.0,
    ) -> None:
        """Configure tiles and bind the first scene camera for display."""
        if not self.cfg.streaming_view:
            return
        self._streaming_aspect = target_aspect
        self._camera_sensor_indices = resolve_streaming_envs(
            num_envs, self.cfg.streaming_envs, sample_from=visible_env_ids
        )
        self._camera_sensor = next(
            (camera for camera in self._cameras if not isinstance(camera, PerspectiveCameraCfg)), None
        )

    def render_tiled_rgba_array(self) -> wp.array | None:
        """Acquire the selected camera frame and compose a device-resident display image.

        Returns:
            Reused uint8 RGBA storage of shape [H, W, 4], or None without a selected camera.
            Acquisition uses the sensor's normal lazy update. Composition itself only reads published arrays.
        """
        camera, env_ids, cfg = self._camera_sensor, self._camera_sensor_indices, self.cfg
        if camera is None or not env_ids:
            return None
        gt_types = tuple(cfg.streaming_gt_types)
        aspect, depth_min, depth_max = self._streaming_aspect, cfg.streaming_depth_min, cfg.streaming_depth_max
        view_key = (camera, tuple(env_ids), gt_types, aspect, depth_min, depth_max)
        frame = self._streaming_frame
        if view_key != self._streaming_view_key:
            available = frozenset(camera.cfg.data_types)
            self._streaming_keys = tuple(sensor_key_for_gt_type(gt, available) for gt in gt_types)
            self._streaming_layout = None
            self._streaming_view_key = view_key
            frame.timestamp = -1.0
        if frame.timestamp == self._sim_time:
            return frame.data

        outputs = camera.data.output
        sources = tuple(outputs[key].warp for key in self._streaming_keys)
        layout = tuple((source.shape, source.dtype, source.device) for source in sources)
        if self._streaming_layout != layout:
            if not sources:
                raise ValueError("Image composition requires at least one display channel.")
            for source, gt in zip(sources, gt_types, strict=True):
                channels = 3 if gt in ("rgb", "normals") else 1
                if source.ndim != 4 or min(source.shape) < 1 or source.shape[3] < channels:
                    raise ValueError(
                        f"Channel {gt!r} requires nonempty [N, H, W, C] arrays with at least {channels} channels."
                    )
            device = sources[0].device
            n, height, width, _ = sources[0].shape
            if any(source.device != device or source.shape[:3] != (n, height, width) for source in sources):
                raise ValueError("Image channels must have the same batch size, resolution, and device.")
            if min(env_ids) < 0 or max(env_ids) >= n:
                raise ValueError(f"Image row selection is outside the source batch of {n} rows.")
            columns = image_grid_columns(len(env_ids), len(sources), height, width, aspect)
            shape = (math.ceil(len(env_ids) / columns) * height, columns * len(sources) * width, 4)
            colors = np.empty((0, 3), dtype=np.uint8)
            if "depth" in gt_types:
                from matplotlib import colormaps

                colors = (colormaps["turbo"](np.arange(256) / 255.0)[..., :3] * 255).astype(np.uint8)
            self._streaming_env_ids = wp.array(env_ids, dtype=wp.int32, device=device)
            self._streaming_depth_colors = wp.array(colors, dtype=wp.uint8, device=device)
            frame.data = wp.empty(shape, dtype=wp.uint8, device=device)
            self._streaming_layout = layout
        compose_image(
            frame.data, sources, self._streaming_env_ids, gt_types, self._streaming_depth_colors,
            depth_min=depth_min, depth_max=depth_max,
        )  # fmt: skip
        frame.timestamp = self._sim_time
        self._streaming_host_frame.timestamp = -1.0
        return frame.data

    def render_tiled_rgb_array(self) -> np.ndarray | None:
        """Read back the tiled image for CPU consumers such as recording and web transports.

        Returns:
            Cached contiguous uint8 RGB image of shape [H, W, 3], or None without a selected camera.
        """
        image = self.render_tiled_rgba_array()
        if image is None:
            return None
        frame = self._streaming_host_frame
        if frame.timestamp != self._streaming_frame.timestamp:
            frame.data = np.ascontiguousarray(image.numpy()[..., :3])
            frame.timestamp = self._streaming_frame.timestamp
        return frame.data

    @abstractmethod
    def step(self, dt: float) -> None:
        """Update visualization for one step.

        Args:
            dt: Time step in seconds.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release borrowed scene references after the backend releases its native resources.

        Subclasses must call ``super().close()`` when their resource teardown finishes, including on failure.
        """
        self._camera_sensor = None
        self._cameras.clear()
        self._streaming_frame = TimestampedBuffer()
        self._streaming_host_frame = TimestampedBuffer()
        self._streaming_env_ids = self._streaming_depth_colors = None
        self._streaming_layout = self._streaming_view_key = None
        self._streaming_keys = ()
        self._sim = None
        self._is_closed = True

    @abstractmethod
    def is_running(self) -> bool:
        """Check if visualizer is still running (e.g., window not closed).

        Returns:
            ``True`` if the visualizer is running, otherwise ``False``.
        """
        raise NotImplementedError

    def is_training_paused(self) -> bool:
        """Check if training is paused by visualizer controls.

        Returns:
            ``True`` if training is paused, otherwise ``False``.
        """
        return False

    def is_rendering_paused(self) -> bool:
        """Check if rendering is paused by visualizer controls.

        Returns:
            ``True`` if rendering is paused, otherwise ``False``.
        """
        return False

    def is_reset_requested(self) -> bool:
        """Check if an episode reset was requested from visualizer controls.

        Returns:
            ``True`` if a reset was requested, otherwise ``False``.
        """
        return self._reset_requested

    def consume_reset_request(self) -> bool:
        """Return whether an episode reset was requested and clear the flag.

        Returns:
            ``True`` once when a reset was requested, then ``False`` until the next request.
        """
        requested = self._reset_requested
        self._reset_requested = False
        return requested

    @property
    def is_initialized(self) -> bool:
        """Check if initialize() has been called."""
        return self._is_initialized

    @property
    def is_closed(self) -> bool:
        """Check if close() has been called."""
        return self._is_closed

    @property
    def physics_backend(self) -> str | None:
        """Return the active physics backend name (e.g. ``'newton'``, ``'physx'``, ``'ovphysx'``).

        Returns:
            Backend name string, or ``None`` when no simulation context is active yet.
        """
        try:
            from ..sim.simulation_context import SimulationContext
            from ..utils.backend_utils import FactoryBase

            if SimulationContext.instance() is None:
                return None
            return FactoryBase._get_backend()
        except Exception:
            return None

    def supports_markers(self) -> bool:
        """Check if visualizer supports VisualizationMarkers.

        Returns:
            ``True`` if marker rendering is supported, otherwise ``False``.
        """
        return False

    def supports_live_plots(self) -> bool:
        """Check if visualizer supports live plots.

        Returns:
            ``True`` if live plots are supported, otherwise ``False``.
        """
        return False

    def add_live_plots(
        self,
        managers: dict[str, ManagerBase],
        scalars: dict[str, dict[str, Any]] | None = None,
        term_names: dict[str, list[str]] | None = None,
        env_idx: int = 0,
    ) -> None:
        """Register environment managers and direct scalars as live-plot data sources.

        Creates one :class:`~isaaclab.ui.live_plots.ManagerLivePlots` per manager and one
        :class:`~isaaclab.ui.live_plots.DirectScalarLivePlots` per scalar group, storing all
        sources for use inside :meth:`_render_live_plots`.  Does nothing when
        :meth:`supports_live_plots` returns ``False`` or ``cfg.enable_live_plots`` is ``False``.

        Args:
            managers: Mapping of manager name to manager instance.
            scalars: Optional mapping of group name to a dict of ``{term_name: callable}``.
                Each callable must take no arguments and return a numeric value.  Used to
                plot non-manager metrics such as episode reward or episode length.
            term_names: Optional per-manager allowlists of term names to include.
                ``None`` (default) collects all terms for every manager.
            env_idx: Environment index to sample each step.  Defaults to ``0``.
        """
        if not self.supports_live_plots():
            return
        if not getattr(self.cfg, "enable_live_plots", True):
            return

        if os.environ.get("ISAACLAB_DISABLE_LIVE_PLOTS", "0") == "1":
            return
        from ..ui.live_plots.manager_live_plots import DirectScalarLivePlots, ManagerLivePlots

        # Scalar groups (e.g. episode metrics) are placed first so they appear at
        # the top of every visualizer's plot list regardless of backend ordering.
        self._live_plot_sources = [DirectScalarLivePlots(name, group) for name, group in (scalars or {}).items()]
        for name, mgr in managers.items():
            # Skip managers that have no active terms — they contribute nothing to plots
            # and would create empty panels in Rerun, Viser, and the Kit live-plot window.
            active = getattr(mgr, "active_terms", None)
            if active is not None:
                has_terms = any(active.values()) if isinstance(active, dict) else bool(active)
                if not has_terms:
                    continue
            self._live_plot_sources.append(ManagerLivePlots(name, mgr, (term_names or {}).get(name)))
        self._live_plot_env_idx = env_idx

    def _render_live_plots(self) -> None:
        """Push live-plot data to the backend for the current step.

        Called from each backend's :meth:`step` implementation when live plots are active.
        The default implementation is a no-op; backends that support live plots override this
        method to forward collected term values to their native plotting API (e.g.
        ``viewer.log_scalar``).
        """
        pass

    def requires_forward_before_step(self) -> bool:
        """Whether simulation should run forward() before step().

        Returns:
            ``True`` when forward kinematics should run before stepping.
        """
        return False

    def pumps_app_update(self) -> bool:
        """Whether this visualizer calls omni.kit.app.get_app().update() in step().

        Returns True for visualizers (e.g. KitVisualizer) that already pump the Kit
        app loop, so SimulationContext.render() can skip its own app.update() call
        and avoid double-rendering.
        """
        return False

    def get_visualized_env_ids(self) -> list[int] | None:
        """Return env IDs this visualizer is displaying, if any.

        Returns:
            Visualized environment ids, or ``None`` for all environments.
        """
        return self._env_ids

    def get_rendering_dt(self) -> float | None:
        """Get rendering time step.

        Returns:
            Rendering time step override, or ``None`` to use interface default.
        """
        return None

    def set_camera_view(self, eye: tuple, target: tuple) -> None:
        """Set camera view position.

        Args:
            eye: Camera eye position.
            target: Camera target position.
        """
        pass

    def _focal_length_to_vertical_fov_degrees(self) -> float:
        """Convert cfg focal length to vertical FOV using USD's default aperture."""
        focal_length = float(self.cfg.focal_length)
        if focal_length <= 0.0:
            raise ValueError("VisualizerCfg.focal_length must be positive.")
        return math.degrees(2.0 * math.atan(_USD_DEFAULT_VERTICAL_APERTURE_MM / (2.0 * focal_length)))

    def reset(self, soft: bool = False) -> None:
        """Reset visualizer state.

        Args:
            soft: Whether to perform a soft reset.
        """
        self._streaming_frame.timestamp = -1.0
        self._streaming_host_frame.timestamp = -1.0

    def _log_initialization_table(self, logger: logging.Logger, title: str, rows: list[tuple[str, Any]]) -> None:
        """Log a compact initialization table for a visualizer.

        Args:
            logger: Logger used to emit the table.
            title: Table title.
            rows: Table row key/value pairs.
        """
        from prettytable import PrettyTable

        table = PrettyTable()
        table.title = title
        table.field_names = ["Field", "Value"]
        table.align["Field"] = "l"
        table.align["Value"] = "l"
        for key, value in rows:
            table.add_row([key, value])
        logger.debug("Visualizer initialization:\n%s", table.get_string())

    def _log_viewer_url(
        self,
        visualizer_name: str,
        viewer_url: str,
    ) -> None:
        """Queue a visible browser URL block for web-based visualizers.

        Args:
            visualizer_name: Name of the visualizer exposing the URL.
            viewer_url: Browser URL for the visualizer.
        """
        parsed_url = urlparse(viewer_url)
        visualizer_label = visualizer_name.removesuffix("Visualizer").lower()
        title = f" {visualizer_label} (listening *:{parsed_url.port}) " if parsed_url.port else f" {visualizer_label} "
        label = "URL"
        label_width = len(label)
        value_width = max(len(viewer_url), len(title) + 2, 21)
        inner_width = label_width + value_width + 9
        left_rule_width = max((inner_width - len(title)) // 2, 1)
        right_rule_width = max(inner_width - len(title) - left_rule_width, 1)

        lines = [
            f"╭{'─' * left_rule_width}{title}{'─' * right_rule_width}╮",
            f"│{' ' * (label_width + 4)}╷{' ' * (value_width + 4)}│",
            f"│   {label:<{label_width}} │ {viewer_url:<{value_width}}   │",
            f"│{' ' * (label_width + 4)}╵{' ' * (value_width + 4)}│",
            f"╰{'─' * inner_width}╯",
        ]
        self._deferred_startup_messages.append("\n" + "\n".join(lines) + "\n")

    def flush_startup_messages(self) -> None:
        """Print deferred startup messages immediately before the workflow update loop starts."""
        for message in self._deferred_startup_messages:
            print(message, flush=True)
        self._deferred_startup_messages.clear()

    def play(self) -> None:
        """Handle simulation play/start. No-op by default."""
        pass

    def pause(self) -> None:
        """Handle simulation pause. No-op by default."""
        pass

    def stop(self) -> None:
        """Handle simulation stop. No-op by default."""
        pass
