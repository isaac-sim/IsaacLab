Changed
^^^^^^^

* :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView` now ships
  pass-through ``FrameViewWorldSpaceWriter`` / ``FrameViewLocalSpaceWriter``
  implementations so writes follow the new
  :meth:`~isaaclab.sim.views.BaseFrameView.xform_world_space_writer` /
  :meth:`~isaaclab.sim.views.BaseFrameView.xform_local_space_writer` context API.
  ``set_world_poses`` / ``set_local_poses`` shims still work (one-time
  ``DeprecationWarning`` per class).  The legacy ``set_scales`` /
  ``get_scales`` paths continue to operate on Newton collision-shape
  geometry sizes -- they are not routed through the writer because the
  writer's ``set_scales`` writes the transform-scale state.
