Fixed
^^^^^

* Fixed :class:`~isaaclab_physx.renderers.IsaacRtxRenderer` rendering ``simple_shading_*``
  camera outputs through the full path-tracing pipeline. The renderer selected only the
  Minimal shading level, through the process-wide ``/rtx/minimal/mode`` carb setting, while
  leaving the render mode at ``RealTimePathTracing``. It now authors
  ``omni:rtx:rendermode = "Minimal"`` and ``omni:rtx:minimal:mode`` on the requesting render
  product, matching the OVRTX renderer. Cameras requesting different shading levels no longer
  overwrite each other, and color cameras, the Kit viewport, and deterministic rendering keep
  path tracing.
