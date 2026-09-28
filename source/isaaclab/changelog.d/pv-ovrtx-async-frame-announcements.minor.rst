Added
^^^^^

* Added :meth:`~isaaclab.renderers.BaseRenderer.announce_frame`. The framework announces each frame
  with the physics step count before it stages camera poses and scene state. Renderers that
  pipeline across frames use the index to group one frame's scene writes and renders. The default
  does nothing, so existing renderer implementations are unaffected.
