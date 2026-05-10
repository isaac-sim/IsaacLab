Fixed
^^^^^

* Fixed ``test_visualizer_cartpole_integration::test_cartpole_newton_visualizer_viewergl_rgb_motion``
  returning a fully-black ``ViewerGL.get_frame`` buffer on the Newton 1.2.0rc2
  + warp 1.13 cohort. ``NewtonVisualizer._apply_camera_pose`` was assigning
  ``self._viewer.camera.pos = wp.vec3(*cam_pos)``, but Newton's
  ``Camera.translate()`` adds a ``pyglet.math.Vec3`` delta with ``+=``.
  warp 1.13's strict ``__add__`` rejects ``wp.vec3 + pyglet.math.Vec3``
  with ``TypeError``; the exception was silenced by the visualizer's
  ``try/except``, which prevented ``renderer.render()`` from ever running
  -- so the framebuffer stayed empty and read back as all zeros. The fix
  assigns ``pyglet.math.Vec3`` instead, matching what Newton uses internally.
* Re-enabled ``test_cartpole_newton_visualizer_viewergl_rgb_motion`` after the
  workaround skip in https://github.com/isaac-sim/IsaacLab/pull/5538.
