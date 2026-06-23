Added
^^^^^

* Added :meth:`~isaaclab_ovphysx.sim.views.OvPhysxFrameView.get_local_scales`
  and :meth:`~isaaclab_ovphysx.sim.views.OvPhysxFrameView.get_world_scales`,
  which delegate to the internal :class:`~isaaclab.sim.views.UsdFrameView`.
  Scale writes go through the writer scope (see the ``xform-space-writer``
  fragment).

Deprecated
^^^^^^^^^^

* Deprecated :meth:`~isaaclab_ovphysx.sim.views.OvPhysxFrameView.get_scales` and
  :meth:`~isaaclab_ovphysx.sim.views.OvPhysxFrameView.set_scales`.  For reads,
  use the explicit ``get_local_scales`` (operates on ``xformOp:scale``) or
  ``get_world_scales``.  For writes, use the writer scope's
  ``set_scales``.  The deprecated methods still work but emit a
  ``DeprecationWarning`` and default to local scales, preserving prior
  behavior.
