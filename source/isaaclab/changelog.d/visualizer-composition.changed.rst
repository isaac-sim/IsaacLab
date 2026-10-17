* **Breaking:** Moved shared visualizer settings to ``VisualizerCfg`` and selected implementation options
  through its ``backend`` field. Replaced ``SimulationCfg.default_visualizer_cfg`` with a common
  ``visualizer_cfgs=VisualizerCfg(...)`` declaration. Moved window, camera, and visibility arguments out
  of backend configs; custom backend configs now inherit ``VisualizerBackendCfg``. Used the common
  1920-by-1080 window default and explicit ``streaming_view=True`` for automatic camera discovery.
* **Breaking:** Restricted the teapot-fill demo to Newton GL and removed its Kit/RTX viewer setup.
  Omit ``--visualizer`` or use ``--visualizer newton_gl``; use the surface-reconstruction tutorial for RTX rendering.
