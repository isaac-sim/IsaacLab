Changed
^^^^^^^

* :class:`~isaaclab_ovphysx.sim.views.OvPhysxFrameView` now ships
  pass-through ``FrameViewWorldSpaceWriter`` / ``FrameViewLocalSpaceWriter``
  implementations so writes follow the new
  :meth:`~isaaclab.sim.views.BaseFrameView.xform_world_space_writer` /
  :meth:`~isaaclab.sim.views.BaseFrameView.xform_local_space_writer` context API.
  ``set_world_poses`` / ``set_local_poses`` shims still work (one-time
  ``DeprecationWarning`` per class).  Scale writes inside the writer scope
  delegate to the internal :class:`~isaaclab.sim.views.UsdFrameView` and
  land in the USD stage (no propagation to OVPhysX-side collision-shape
  scales).
