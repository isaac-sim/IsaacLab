Changelog
---------

0.1.4 (2026-06-08)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :meth:`~isaaclab_visualizers.newton.NewtonVisualizer.set_camera_view` so
  the Newton visualizer follows :meth:`~isaaclab.sim.SimulationContext.set_camera_view`
  camera updates.


0.1.3 (2026-06-06)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated the visualizer tiled camera tutorial support to keep generated Kit
  tiled camera views synchronized with their target robots.

Fixed
^^^^^

* Fixed Newton visualizer contact rendering by logging Newton contact buffers
  when available and falling back to scene contact sensors for PhysX-backed
  scenes.
* Fixed Newton visualizer HUD dependency checks by requiring
  ``typing-extensions>=4.15.0`` for the Newton visualizer extra and failing
  integration tests when Newton reports that ``imgui_bundle`` could not be
  imported. Removed the legacy ``setup.py`` for ``isaaclab_visualizers`` now that
  ``pyproject.toml`` carries the package metadata.

* Fixed Rerun and Viser visualizers rendering Newton infinite ground planes too
  small by expanding non-positive plane extents to the same large finite size
  used by Newton GL.

* Fixed Viser visualizer ground-grid flickering by reusing unchanged plane grid
  line segments instead of removing and re-adding them every frame.


0.1.2 (2026-06-05)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Bumped the ``newton[sim]`` pin used by the visualizer extras from ``v1.2.0`` to ``v1.2.1rc2``.
* Removed the legacy ``setup.py`` packaging entry point now that ``pyproject.toml`` owns the visualizers package metadata.


0.1.1 (2026-06-03)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Switched the Newton install spec to ``newton[sim]`` in the ``newton``,
  ``rerun``, and ``viser`` extras so the MuJoCo solver dependencies are
  pulled in transitively. Required because pip resolves a git-URL
  requirement once for the URL; a bare ``newton @ git+...`` here would
  shadow the ``[sim]`` extra requested elsewhere.
* Bumped the ``newton[sim]`` pin in the ``[newton]``, ``[rerun]``, and
  ``[viser]`` extras to ``v1.2.0`` (stable) so the pin matches
  :mod:`isaaclab_newton`.
* Changed Rerun and Viser visualizers to avoid opening browser tabs by default and to show browser URLs in the startup logs instead.
* Changed visualizer initialization tables to debug-level logging to reduce default startup log noise.
* Added non-interactive tiled camera image views for Kit and Newton visualizers, with generated per-env cameras or existing Camera sensor support.
* Added clearer Kit visualizer errors when tiled camera views are enabled without camera rendering support.
* Split visualizer integration coverage into separate interactive and tiled camera cases.
* Renamed the Newton visualizer tiled camera control section to ``Tiled Camera View``.

Fixed
^^^^^

* Fixed ``test_visualizer_cartpole_integration::test_cartpole_newton_visualizer_viewergl_rgb_motion``
  returning a fully-black ``ViewerGL.get_frame`` buffer on the Newton 1.2.0rc2
  + warp 1.13 cohort. ``NewtonVisualizer._apply_camera_pose`` was assigning
  ``self._viewer.camera.pos = wp.vec3(*cam_pos)``, but Newton's
  ``Camera.translate()`` adds a ``pyglet.math.Vec3`` delta with ``+=``.
  warp 1.13's strict ``__add__`` rejects ``wp.vec3 + pyglet.math.Vec3``
  with ``TypeError``; the exception was silenced by the visualizer's
  ``try/except``, which prevented ``renderer.render()`` from ever running
  -- so the framebuffer stayed empty and read back as all zeros. The fix
  assigns ``pyglet.math.Vec3`` instead, matching what Newton uses internally.
* Re-enabled ``test_cartpole_newton_visualizer_viewergl_rgb_motion`` after the
  workaround skip in https://github.com/isaac-sim/IsaacLab/pull/5538.
* Updated ``configclass`` imports in :mod:`isaaclab_visualizers.kit`,
  :mod:`isaaclab_visualizers.newton`, :mod:`isaaclab_visualizers.rerun`, and
  :mod:`isaaclab_visualizers.viser` visualizer configs to import from
  :mod:`isaaclab.utils.configclass` directly, matching the lazy-import layout
  introduced in :mod:`isaaclab.utils`.
* Updated ``test_visualizer_cartpole_integration`` to read the tiled camera
  RGB output via the ``.torch`` accessor, matching the Warp-backed camera
  data API.
* Fixed :meth:`~isaaclab_visualizers.kit.KitVisualizer._set_viewport_camera`
  raising ``Boost.Python.ArgumentError: Matrix4d.Transform(Matrix4d, NoneType)``
  during ``sim.reset()`` when ``KitVisualizerCfg.eye`` / ``lookat`` were
  configured. The call was issuing ``ViewportCameraState.set_position_world(...,
  rotate=True)`` on a freshly-initialized viewport camera, which reads
  ``omni:kit:centerOfInterest`` from the camera prim and pipes it through
  ``world_xform.Transform(...)``; on an unauthored COI the attribute getter
  returns ``None`` and the C++ binding rejects it. The position set now uses
  ``rotate=False`` -- the subsequent ``set_target_world(..., rotate=True)``
  authors the COI and rotates the camera to the configured target.
* Added Kit RTX render-product and Newton viewer warm-up steps to reduce
  cold-start visualizer integration test flakes from stale frame captures.
* Fixed Kit visualizer viewport rendering when RTX scene partitioning is enabled.


0.1.0 (2026-06-02)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Initial version of the :mod:`isaaclab_visualizers` extension, providing
  visualizer backends for Isaac Lab across Kit, Newton, Rerun, and Viser.
