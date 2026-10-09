* **Breaking:** Changed ``VisualizerCfg.background_color`` to default to ``None``, preserving the scene HDR in
  Kit and Newton RTX or the procedural sky in Newton GL. Set ``background_color=(0.3, 0.55, 0.82)``
  to retain the previous solid sky-blue background.
* **Breaking:** Changed custom visualizer initialization to ``initialize(sim, *, cameras)``.
  Call ``super().initialize(sim, cameras=cameras)`` to retain the simulation owner as ``self._sim`` and bind scene inputs.
  Resolve backend resources inside the visualizer; keep ``reset(soft=False)`` backend-independent.
* **Breaking:** Moved native window dimensions into shared ``WindowCfg(size=(width, height))``
  on ``VisualizerCfg.window``. Replaced Newton's simulation-frame ``update_frequency`` with
  ``WindowCfg.fps`` (30 by default); headless on-demand capture retained its independent cadence.
  Renamed the GPU array API to ``render_tiled_rgba_array()`` for consistency with ``render_tiled_rgb_array()``.
