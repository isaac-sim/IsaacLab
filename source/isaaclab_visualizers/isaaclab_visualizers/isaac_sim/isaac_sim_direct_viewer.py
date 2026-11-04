# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Isaac Sim 7 viewer attached directly to Isaac Lab's loop, with no host world or scheduler.

This follows Isaac Sim's ``viewer.viewport_direct`` example: a :class:`~isaacsim.application.viewer.Viewer` borrows a
viewport and the owner calls ``control()`` and renders on its own thread. Isaac Lab renders with its own OVRTX
renderer, so the viewport here is a small adapter that serves Isaac Lab's frames and turns the client's navigation into
a camera pose that Isaac Lab writes with the next frame.

Frames are rendered on demand: only while a client shows the viewport and has room for another image, and at most
``window.fps`` times a second. Rendering, readback and encoding then share Isaac Lab's step budget.

Open points from this prototype, to discuss with the Isaac Sim team (these are experimental notes, not settled design):

* Frame source: Isaac Sim's ``Viewport`` reads frames that a renderer publishes into the stage context, and only OVGL's
  ``StageRenderer`` publishes that way outside a scheduler. :class:`FrameViewport` repeats its navigation handling to
  serve Isaac Lab's OVRTX frames instead. A ``Viewport`` that accepts a frame callable, as ``RendererOutput`` does,
  would remove that copy.
* Encoding: the viewer always encodes PNG, about 130 ms per 720p frame here. :class:`DirectViewer` swaps in the
  ``auto`` encoder through a private attribute. A ``ViewerConfig`` setting would avoid that.
* Presentation: the desktop client tries GPU textures first and logs one error per viewport before falling back to
  images, because these frames are CPU pixels. An option to offer images only would keep the log clean.
* Panes: ``Viewer`` rejects the stage, property and content panes, which need the application host's services. It is
  still to be decided whether a viewport is enough for Isaac Lab, or whether those services could work with ``Viewer``.
* Transport: ``Viewer`` has no Play, Pause, Step or Stop. Isaac Lab could add them through ``Viewer.router`` and a
  custom panel, or keep its own controls.
* Stage ownership: the stage is still created through Isaac Sim's ``UsdContext`` so the panes of ``host="app"`` can
  read it. With a viewport only, Isaac Lab could create and own the OVStage itself.
* Viewport size: the render product keeps ``window.size``; resizing the pane does not resize it yet. Clicks are off.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from .isaac_sim_shared_stage import FrameStore, Replication, SharedStage, ViewerRenderer

if TYPE_CHECKING:
    from collections.abc import Callable

    from isaacsim.viewports.adapters import Camera

VIEWER_CAMERA = "/IsaacLabViewer/Camera"
"""Camera the viewer navigates."""

VIEWER_PRODUCT = "/IsaacLabViewer/Product"
"""Render product the viewer displays."""


def viewer_layer(scene: str, resolution: tuple[int, int], up_axis: str, meters_per_unit: float) -> str:
    """Return a root layer that sublayers the exported scene and adds the viewer's camera and render product.

    Args:
        scene: Path of the exported scene.
        resolution: Render width and height [px].
        up_axis: Scene up axis, ``"Y"`` or ``"Z"``.
        meters_per_unit: Scene units [m].
    """
    near, far = 0.01 / meters_per_unit, 100000.0 / meters_per_unit
    return f"""#usda 1.0
(
    subLayers = [@{scene}@]
    upAxis = "{up_axis}"
    metersPerUnit = {meters_per_unit!r}
)
def Scope "IsaacLabViewer"
{{
    def Camera "Camera"
    {{
        float2 clippingRange = ({near}, {far})
        float focalLength = 18.147562
        float horizontalAperture = 20.955
        float verticalAperture = 15.2908
    }}
    def RenderProduct "Product"
    {{
        rel camera = <{VIEWER_CAMERA}>
        rel orderedVars = [<{VIEWER_PRODUCT}/LdrColor>]
        uniform int2 resolution = ({resolution[0]}, {resolution[1]})
        def RenderVar "LdrColor"
        {{
            uniform string sourceName = "LdrColor"
            uniform token dataType = "color4f"
        }}
    }}
}}
"""


class FrameViewport:
    """Viewport for Isaac Sim's ``Viewer`` that serves frames Isaac Lab renders.

    Isaac Sim's own ``Viewport`` reads frames that a renderer publishes into the stage context and writes the camera
    through that context; Isaac Lab's OVRTX renderer keeps frames in a :class:`FrameStore` and only Isaac Lab writes the
    stage. This class provides the same members ``Viewer`` uses.

    Args:
        frames: Frames Isaac Lab renders.
        product: Render product shown.
        camera: Initial camera placement; navigation orbits around its target.
    """

    source_id = "isaaclab"
    on_click = None

    def __init__(self, frames: FrameStore, product: str, camera: Camera):
        from isaacsim.viewports.adapters import RendererOutput
        from isaacsim.viewports.manager import ViewportManager, ViewportSource

        self._frames = frames
        self._product = product
        self._camera = camera
        self._orbit = camera.orbit
        self._lock = threading.Lock()
        self._pending = True
        self._pose: np.ndarray | None = None
        self.manager = ViewportManager()
        self.manager.register(
            ViewportSource(self.source_id, "Viewport", "ovrtx", lambda: RendererOutput(read=self._latest))
        )

    def _latest(self) -> Any:
        return self._frames.latest(self._product)

    @property
    def camera(self) -> Camera:
        return self._camera

    @property
    def visible(self) -> bool:
        """Whether a client currently shows the viewport."""
        return self.manager.is_visible(self.source_id)

    def navigation_state(self) -> dict[str, Any]:
        with self._lock:
            return self._orbit.as_dict()

    def navigate(self, operations: dict[str, Any]) -> dict[str, Any]:
        """Apply the client's orbit, pan, dolly or fly gestures; the pose reaches the stage with the next frame."""
        from isaacsim.viewports.authoring import apply_navigation

        with self._lock:
            self._orbit = apply_navigation(self._orbit, operations)
            self._pending = True
            return self._orbit.as_dict()

    def flush_camera(self, *, context: Any = None) -> bool:
        """Stage the latest camera pose for Isaac Lab; return whether it changed."""
        with self._lock:
            if not self._pending:
                return False
            self._pose = np.asarray(self._orbit.matrix(), dtype=np.float64).reshape(1, 16)
            self._pending = False
            return True

    def take_pose(self) -> np.ndarray | None:
        """Return the camera's row-major world transform if it changed since the last call."""
        with self._lock:
            pose, self._pose = self._pose, None
            return pose

    def read_publication(self, *, context: Any = None) -> tuple[int, int]:
        """Return the producer generation and the latest frame's id, or zeros before the first frame."""
        frame = self._frames.latest(self._product)
        return (1, frame.frame_id) if frame is not None else (0, 0)

    def close(self) -> None:
        """Stop serving frames once the client's reads finish."""
        self.manager.stop()
        self.manager.wait_stopped()


class DirectViewer:
    """Shared stage, OVRTX renderer and Isaac Sim viewer, all driven from Isaac Lab's thread.

    Args:
        usd: Root layer the stage is populated from; it must define :data:`VIEWER_CAMERA` and :data:`VIEWER_PRODUCT`.
        replication: Environment copies and placement applied once the stage is populated.
        camera: Initial camera placement.
        device: CUDA device ordinal for rendering.
        desktop: Whether to open the Electron window; otherwise only the browser client is served.
        port: Port of the viewer's HTTP service; zero picks a free one.
        encoder: Image codec for frames sent to the client, as accepted by Isaac Sim's ``FrameEncoder.create``.
    """

    def __init__(
        self,
        usd: str,
        replication: Replication | None,
        camera: Camera,
        device: int = 0,
        desktop: bool = True,
        port: int = 0,
        encoder: str = "auto",
    ):
        from isaacsim.application.viewer import Viewer, ViewerConfig

        self.stage = SharedStage(usd, replication)
        self.startup = dict(self.stage.startup)
        self.renderer = ViewerRenderer(self.stage, device)
        self.renderer.add_product(VIEWER_PRODUCT)
        self.frames = FrameStore()
        self.viewport = FrameViewport(self.frames, VIEWER_PRODUCT, camera)
        # Clicks would need a pick or a ray callback, and layouts only hold the one viewport.
        config = ViewerConfig(title="Isaac Lab", application_id="isaaclab", desktop=desktop, port=port, clicks=False)
        self.viewer = Viewer(self.viewport, config)
        from isaacsim.viewports.presentation import FrameEncoder

        # Open question for Isaac Sim: the viewer always encodes PNG (about 130 ms per 720p frame here, so about 10
        # frames per second reach the client) and has no setting for it, so this sets its private encoder.
        self.viewer._frames._encoder = FrameEncoder.create(encoder)
        self.viewer.start()
        self._camera_armed = False
        self.renders = 0

    @property
    def url(self) -> str:
        """Browser address of the viewer, including its launch token."""
        return self.viewer.url

    def step(self, time_ns: int, write_state: Callable[[SharedStage], None]) -> bool:
        """Apply the client's input, publish a finished frame and, when a client wants one, render the next.

        Args:
            time_ns: Isaac Lab's simulation time [ns].
            write_state: Writes Isaac Lab's state at the open ordinal.

        Returns:
            Whether the viewer is still open.
        """
        if not self.viewer.control():
            return False
        published, busy = self.renderer.poll(self.frames)
        self.renders += published
        if busy or not self.viewer.should_render(scene_changed=True):
            return True
        write_state(self.stage)
        pose = self.viewport.take_pose()
        if pose is not None:
            if not self._camera_armed:
                # The orbit pose is a world transform.
                self.stage.write([VIEWER_CAMERA], "omni:resetXformStack", np.ones(1, dtype=np.bool_))
                self._camera_armed = True
            self.stage.write([VIEWER_CAMERA], "omni:xform", pose)
        self.renderer.render(self.stage.seal(), time_ns)
        return True

    def close(self) -> None:
        """Close the viewer, then the viewport, renderer and stage."""
        try:
            self.viewer.close()
            self.viewport.close()
        finally:
            try:
                self.renderer.close()
            finally:
                self.stage.close()
