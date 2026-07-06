Fixed
^^^^^

* Removed overly broad ``except Exception`` handling in :class:`~isaaclab_ov.renderers.ovrtx_renderer.OVRTXRenderer`
  that downgraded failures in scene initialization, camera and object binding setup, scene partition writes,
  Newton transform syncing, and :meth:`~isaaclab_ov.renderers.ovrtx_renderer.OVRTXRenderer.render` to log
  warnings and silently continue. These now propagate so callers can decide how to handle the failure.
