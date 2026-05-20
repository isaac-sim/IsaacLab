Fixed
^^^^^

* Fixed :meth:`~isaaclab_visualizers.kit.KitVisualizer._set_viewport_camera`
  raising ``Boost.Python.ArgumentError: Matrix4d.Transform(Matrix4d, NoneType)``
  during ``sim.reset()`` when ``KitVisualizerCfg.eye`` / ``lookat`` were
  configured. The call was issuing ``ViewportCameraState.set_position_world(...,
  rotate=True)`` on a freshly-initialized viewport camera, which reads
  ``omni:kit:centerOfInterest`` from the camera prim and pipes it through
  ``world_xform.Transform(...)``; on an unauthored COI the attribute getter
  returns ``None`` and the C++ binding rejects it. The position set now uses
  ``rotate=False`` -- the subsequent ``set_target_world(..., rotate=True)``
  authors the COI and rotates the camera to the configured target.
