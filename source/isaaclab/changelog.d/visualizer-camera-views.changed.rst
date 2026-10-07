* **Breaking:** Changed ``VisualizerCfg.background_color`` to default to ``None``, preserving the scene HDR in
  Kit and Newton RTX or the procedural sky in Newton GL. Set ``background_color=(0.3, 0.55, 0.82)``
  to retain the previous solid sky-blue background.
* Routed renderer-backed visualization markers through their shared renderer, so multiple views reused one marker group.
