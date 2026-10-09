* **Breaking:** Changed ``VisualizerCfg.background_color`` to default to ``None``, preserving the scene HDR in
  Kit and Newton RTX or the procedural sky in Newton GL. Set ``background_color=(0.3, 0.55, 0.82)``
  to retain the previous solid sky-blue background.
* **Breaking:** Changed custom visualizer initialization to ``initialize(sim, *, cameras)``.
  Call ``super().initialize(sim, cameras=cameras)`` to retain the simulation owner as ``self._sim`` and bind scene inputs.
  Resolve backend resources inside the visualizer; keep ``reset(soft=False)`` backend-independent.
* Added shared ``ImageViewCfg`` selections for scene cameras and perspective renders. Windows and
  ``VideoRecorderCfg(view=...)`` reused one composed device image and one host readback per frame.
  Kept legacy source strings and camera configurations; new configurations use
  ``NewtonGLVisualizerCfg(window=GLWindowCfg(view=view))`` or the RTX equivalent.
* Consolidated display and recording colorization in the device composition kernels, removing the
  duplicate CPU implementation and obsolete camera colorization, gathering, and grid helpers.
