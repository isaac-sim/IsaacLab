* **Breaking:** Changed ``VisualizerCfg.background_color`` to default to ``None``, preserving the scene HDR in
  Kit and Newton RTX or the procedural sky in Newton GL. Set ``background_color=(0.3, 0.55, 0.82)``
  to retain the previous solid sky-blue background.
* **Breaking:** Changed custom visualizer initialization to ``initialize(sim, *, cameras)``.
  Call ``super().initialize(sim, cameras=cameras)`` to bind scene inputs and the resource registry.
  Resolve backend resources inside the visualizer; keep ``reset(soft=False)`` backend-independent.
