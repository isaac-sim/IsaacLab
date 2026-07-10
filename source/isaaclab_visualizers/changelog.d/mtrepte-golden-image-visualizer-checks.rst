Added
^^^^^

* Added golden image correctness tests for :class:`~isaaclab_visualizers.kit.KitVisualizer` and
  :class:`~isaaclab_visualizers.newton.NewtonVisualizer` in both viewport and tiled-camera capture
  modes, covering PhysX and Newton MJWarp physics backends.  Each combination is compared against
  a committed reference image using a dual-gate (per-pixel L2 norm + SSIM) system adapted from the
  renderer correctness tests, with per-visualizer pixel-diff and SSIM thresholds tuned to each
  backend's rendering determinism.
