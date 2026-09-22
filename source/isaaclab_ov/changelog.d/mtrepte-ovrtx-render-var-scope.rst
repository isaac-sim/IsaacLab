Fixed
^^^^^

* Fixed :class:`~isaaclab_ov.renderers.ovrtx_renderer.OVRTXRenderer` producing all-black frames
  for scene camera sensors on OVRTX 0.5+. Render-var lookups used a fixed, unscoped
  ``"/Render/Vars/<name>"`` key, but OVRTX 0.5+ authors each camera's render vars under its own
  scope (``"/RenderCamera_<id>/Vars/<name>"``), so the lookup silently missed every render var and
  left the output buffer at its zero-initialized state. Render-var keys are now resolved per
  camera from its actual render scope.
