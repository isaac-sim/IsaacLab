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
