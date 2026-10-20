* Separated shared view, camera, window, and visibility settings in ``VisualizerCfg`` from backend-only
  ``*BackendCfg`` options. Kept the old combined ``*VisualizerCfg`` declarations and
  ``SimulationCfg.default_visualizer_cfg`` as deprecated inputs, translated once before launch.
  Migrate to ``VisualizerCfg(..., backend=NewtonGLBackendCfg(...))`` or a task-owned
  ``MultiBackendVisualizerCfg``. Old Kit declarations retained their 720p default; new shared
  declarations used the common 1080p default. Explicit per-view settings retained precedence.
* Kept multiple view declarations independent: two views with ``--viz=newton_gl,newton_rtx`` opened
  four windows, preserving each view's camera and window settings. Direct simulation construction
  also accepted a list of backend configurations.
* **Breaking:** Restricted the teapot-fill demo and its Rizon wrapper to Newton GL and removed their Kit/RTX viewer setup.
  Omit ``--visualizer`` or use ``--visualizer newton_gl``; use the surface-reconstruction tutorial for RTX rendering.
