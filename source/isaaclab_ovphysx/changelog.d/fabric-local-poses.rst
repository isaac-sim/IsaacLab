Added
^^^^^

* Added :meth:`~isaaclab_ovphysx.sim.views.OvPhysxFrameView.get_local_scales`,
  :meth:`~isaaclab_ovphysx.sim.views.OvPhysxFrameView.set_local_scales`,
  :meth:`~isaaclab_ovphysx.sim.views.OvPhysxFrameView.get_world_scales`, and
  :meth:`~isaaclab_ovphysx.sim.views.OvPhysxFrameView.set_world_scales`, which
  delegate to the internal :class:`~isaaclab.sim.views.UsdFrameView`.

Deprecated
^^^^^^^^^^

* Deprecated :meth:`~isaaclab_ovphysx.sim.views.OvPhysxFrameView.get_scales` and
  :meth:`~isaaclab_ovphysx.sim.views.OvPhysxFrameView.set_scales` in favor of the
  explicit ``get_local_scales`` / ``set_local_scales`` (operates on
  ``xformOp:scale``) or ``get_world_scales`` / ``set_world_scales``.  The
  deprecated methods still work but emit a ``DeprecationWarning`` and default to
  local scales, preserving prior behavior.
