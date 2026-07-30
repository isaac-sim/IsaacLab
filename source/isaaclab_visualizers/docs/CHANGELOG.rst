Changelog
---------

1.2.2 (2026-07-30)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added golden image correctness tests for :class:`~isaaclab_visualizers.kit.KitVisualizer` and
  :class:`~isaaclab_visualizers.newton.NewtonVisualizer` in both viewport and tiled-camera capture
  modes, covering PhysX and Newton MJWarp physics backends across four scenes (cartpole, shadow
  hand, AnymalD, and franka cloth).  Each combination is compared against a committed reference
  image using a dual-gate (per-pixel L2 norm + SSIM) system adapted from the renderer correctness
  tests, with per-visualizer pixel-diff and SSIM thresholds tuned to each backend's rendering
  determinism.

Fixed
^^^^^

* Fixed incomplete tiled camera renders for the PhysX physics backend in golden-image and
  tiled-camera integration tests.  :class:`~isaaclab_visualizers.newton.NewtonVisualizer` skipped
  ``_log_camera_sensor_image()`` when the Newton physics state was unavailable (PhysX backend),
  leaving all owned tiled cameras with zero renderer updates during physics warmup; only env 0
  rendered correctly.  The capture helper now pumps ``camera_sensor.update()`` for
  ``_TILED_CAMERA_SENSOR_WARMUP_UPDATES`` iterations before sampling, matching the warmup already
  applied to Kit viewport and Newton viewer paths.

* Fixed :class:`~isaaclab_visualizers.kit.KitVisualizer` silently skipping tiled camera sensor
  creation in headless mode even when ``--enable_cameras`` is active.  The camera sensor is now
  always created when camera rendering is available; only the interactive UI image window is
  suppressed in headless mode.

* Fixed cross-test contamination in golden image tests when tiled-camera and viewport captures
  run sequentially in the same process.  Three sources were addressed: (1) the stale
  :class:`~isaaclab_newton.physics.NewtonManager` shadow model from the tiled stage persisting
  into the viewport test on the PhysX backend (cleared in the between-test prepare step);
  (2) CUDA RNG state drift causing the initial cartpole pole angle to differ between isolated
  and suite runs (seed is now applied immediately before ``env.reset()``); and (3) test ordering
  in both golden test files reordered to run tiled captures before viewport captures to prevent
  RTX render-product state from contaminating tiled camera output.

* Fixed franka-cloth-with-kit-visualizer tiled and viewport golden image tests failing in heavily
  contaminated test suites (12 prior Newton tests before franka cloth) due to Newton body
  transforms never syncing to USD Fabric.  The ``_drain_until_newton_fabric_ready`` helper now
  uses a higher iteration ceiling (600 vs. 200) and more Kit app updates per iteration (4 vs. 2)
  for the tiled-camera path, and adds a ``torch.cuda.synchronize()`` before each ``SelectPrims``
  retry to help flush any queued Fabric CPU-to-GPU attribute propagation work.  The global per-pixel
  L2-norm difference threshold was raised from 10 to 20 to better reflect the magnitude of RTX
  global-illumination contamination in heavily loaded suites; the ``franka_cloth-kit-tiled``
  pixel-diff gate was correspondingly tightened from 40% to 20%.  SSIM thresholds for both
  captures remain unchanged and continue to detect structural pose regressions.

* Eliminated first-attempt flakiness in tiled-camera and Kit-viewport golden image tests by
  replacing the fixed renderer warmup loops (20 iterations) with a convergence-based pump that
  continues until two consecutive frames differ by fewer than 0.5% of pixels at L2 > 1, capped
  at 50 frames.  This ensures the RTX TAA accumulation buffer has genuinely settled before the
  golden-image comparison frame is sampled.  The ``anymal_d-kit-tiled`` SSIM threshold was also
  lowered from 0.945 to 0.910 to accommodate persistent RTX GI cold-start variability (~0.920
  observed vs warm) while preserving a large safety margin above wrong-pose regressions (~0.69
  observed for a completely missing body).  Previously ``anymal_d-kit-tiled``,
  ``shadow_hand-kit-viewport``, and ``franka_cloth-kit-tiled`` routinely required a second
  attempt; after these changes all 16 Newton tests pass on the first attempt consistently.

* Fixed false-passing golden image tests caused by ``np.frombuffer`` returning a read-only view
  of the annotator's internal buffer rather than an independent snapshot.  When Replicator reuses
  the same buffer in place, both the start-frame and end-frame captures reflected the same
  (current) buffer contents, producing zero pixel differences.  The frame capture helper now calls
  ``.copy()`` to take an independent snapshot at capture time.

* Fixed the PhysX visualizer integration test taking ~8 minutes instead of ~55 seconds.  Two
  independent bottlenecks were eliminated: (1) running Newton GL alongside Kit RTX on the PhysX
  backend caused GPU contention that pushed each ``env.step()`` from ~0.1 s to ~1.5 s across 165
  steps — the PhysX test now uses a Kit-only environment so Newton GL is absent; (2)
  ``_drain_until_newton_fabric_ready`` exhausted all 600 iterations per tiled capture on PhysX
  because ``NewtonManager._newton_fabric_ready`` is never set when Newton MJWarp physics is not
  simulating — a 20-iteration probe now detects the PhysX path and skips the remaining drain,
  letting ``_pump_tiled_until_stable`` handle frame convergence instead.


1.2.1 (2026-07-25)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added particle visualization toggle to :class:`~isaaclab_visualizers.rerun.RerunVisualizerCfg`, enabled by default.


1.2.0 (2026-07-24)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``physics_backend`` property to :class:`~isaaclab.visualizers.BaseVisualizer` and surfaced the active
  physics backend label across all visualizers: a non-interactive "Physics: <backend>" label in the Kit viewport
  menubar in :class:`~isaaclab_visualizers.kit.KitVisualizer`; a "Physics" collapsing section in the ImGui HUD
  panel in :class:`~isaaclab_visualizers.newton.NewtonVisualizer`; a sidebar markdown label in
  :class:`~isaaclab_visualizers.viser.ViserVisualizer`; and a ``TextDocumentView`` strip in the blueprint layout
  of :class:`~isaaclab_visualizers.rerun.RerunVisualizer`. When Newton MJWarp is active,
  :class:`~isaaclab_visualizers.kit.KitVisualizer` also hides the built-in "Simulation / PhysX" viewport menu
  item to avoid a misleading label.
* Added general USD mesh support for Newton/Rerun/Viser visualization markers via
  :func:`newton.usd.get_mesh`. Any :class:`~isaaclab.sim.spawners.UsdFileCfg` marker
  now loads geometry and material properties (color, texture) directly from the USD file,
  replacing the previous fallback that silently skipped unsupported USD paths.
  The hardcoded DexCube textured-box workaround has been removed.
* Added live-plot support to :class:`~isaaclab_visualizers.newton.NewtonVisualizer`,
  :class:`~isaaclab_visualizers.rerun.RerunVisualizer`, and
  :class:`~isaaclab_visualizers.viser.ViserVisualizer` via their native
  ``viewer.log_scalar`` APIs.  All three backends now return ``True`` from
  :meth:`~isaaclab.visualizers.BaseVisualizer.supports_live_plots` and override
  :meth:`~isaaclab.visualizers.BaseVisualizer._render_live_plots` to forward
  manager-term scalars each step.
* Added :meth:`~isaaclab_visualizers.kit.KitVisualizer.add_live_plots` to
  :class:`~isaaclab_visualizers.kit.KitVisualizer`, which creates
  :class:`~isaaclab.ui.widgets.ManagerLiveVisualizer` instances for the omni.ui panel path
  while also populating the general :attr:`_live_plot_sources` list.

Changed
^^^^^^^

* :class:`~isaaclab_visualizers.newton.NewtonVisualizer` now renders live plots in a dedicated
  ``Live Plots`` collapsing-header section (previously a sub-label inside ``IsaacLab Options``);
  plots are visible by default and can be toggled per manager.
* :class:`~isaaclab_visualizers.rerun.RerunVisualizer` now shows per-manager
  :class:`~rerun.blueprint.TimeSeriesView` panels as visible by default (was hidden, causing a
  blank white panel).
* :class:`~isaaclab_visualizers.viser.ViserVisualizer` now renders each scalar as its own
  collapsible folder named after the term, replacing the single shared ``Plots`` folder.  The
  viser server label is now ``Live Plots`` (was ``Isaac Lab Simulation``) and non-functional
  per-manager checkboxes have been removed.

Removed
^^^^^^^

* Removed ``config/extension.toml`` Kit extension manifest. Inter-package dependencies are now
  declared via PEP 508 ``file:`` references in ``[project.dependencies]`` of ``pyproject.toml``,
  ensuring standalone pip installs resolve local checkouts without a package index.

Fixed
^^^^^

* Fixed the contact-arrow origin fallback in the Newton visualizer to match the body-major
  ordering of the PhysX contact-sensor view (one view pattern per body).
* Fixed the Newton viewer silently running headless when no display is available: the implicit
  EGL fallback in :class:`~isaaclab_visualizers.newton.NewtonVisualizer` now prints a warning
  explaining that no window will open and how to enable one.
* Fixed Newton visualization marker cleanup during interpreter shutdown.
* Fixed headless Kit visualizer camera setup when no viewport window is available.


1.1.1 (2026-07-08)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed the ``newton[sim]`` dependency pin of the visualizer extras to Newton
  commit ``c7ae7c7648cd0717df39e5c94b95d5a02c997320`` and added the
  ``newton-usd-schemas`` dependency required by Newton's USD parsing.


1.1.0 (2026-07-02)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :meth:`~isaaclab_visualizers.newton.NewtonVisualizer.render_rgb_array` to access the latest
  Newton viewer framebuffer.
* Added :attr:`~isaaclab_visualizers.newton.NewtonVisualizerCfg.world_spacing`
  to visually separate Newton worlds without changing their simulated poses.

Fixed
^^^^^

* Fixed Newton marker filtering for environment-major marker arrays and aligned marker
  overlays with visual world spacing.


1.0.4 (2026-07-01)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed the ``newton[sim]`` dependency pin to Newton commit
  ``2064e3b79807dcc1679d1eb86ef7efd9ef0f28ee``. Projects that install Newton
  separately should use this commit with ``warp-lang==1.15.0.dev20260626``.


1.0.3 (2026-06-25)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed :class:`~isaaclab_visualizers.kit.KitVisualizer` to skip authoring the
  ``omni:scenePartition`` attribute on the viewport camera by default. Set
  ``ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=1`` to re-enable per-environment
  scene partitioning for the Kit viewport camera.


1.0.2 (2026-06-24)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the visualizer extras' ``newton[sim]`` dependency pin to use Newton
  commit ``79e95bf5571d70a0a46c8eaedc80644531d27368``, including the
  RenderContext triangle-mesh construction fix from `newton-physics/newton#3199
  <https://github.com/newton-physics/newton/pull/3199>`_.


1.0.1 (2026-06-14)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added an explicit ``pyglet>=2.1.6,<3`` dependency for the Newton visualizer
  extra so the OpenGL viewer does not rely on ambient transitive installs.


1.0.0 (2026-06-13)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the visualizer extras' ``newton[sim]`` dependency pin to use Newton commit
  ``811968bfb7cc7ff4e37b9260a2ba56930a3e605e``.


0.1.6 (2026-06-12)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* :class:`~isaaclab_visualizers.newton.NewtonVisualizer` now skips Newton's
  per-frame active-particle compaction (two device-to-host reads per render)
  when an MPM model's static particle flags are all active, and re-uploads the
  particle color buffer only when the point count grows or the configured color
  changes.


0.1.5 (2026-06-10)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added Newton visualizer configuration options for showing particles and
  setting their color.

Fixed
^^^^^

* Fixed ``set_camera_view`` updates for the Newton visualizer.


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
