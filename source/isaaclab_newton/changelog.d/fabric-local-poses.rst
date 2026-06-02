Added
^^^^^

* Added :meth:`~isaaclab_newton.sim.views.NewtonSiteFrameView.get_local_scales`,
  :meth:`~isaaclab_newton.sim.views.NewtonSiteFrameView.set_local_scales`,
  :meth:`~isaaclab_newton.sim.views.NewtonSiteFrameView.get_world_scales`, and
  :meth:`~isaaclab_newton.sim.views.NewtonSiteFrameView.set_world_scales`.
  Newton's ``shape_scale`` is an absolute (world-space) quantity, so the local
  methods return the same value as the world methods.  Scale getters now return
  :class:`~isaaclab.utils.warp.ProxyArray`.

Deprecated
^^^^^^^^^^

* Deprecated :meth:`~isaaclab_newton.sim.views.NewtonSiteFrameView.get_scales`
  and :meth:`~isaaclab_newton.sim.views.NewtonSiteFrameView.set_scales` in favor
  of the explicit ``get_world_scales`` / ``set_world_scales`` (or their local
  equivalents).  The deprecated methods still work but emit a
  ``DeprecationWarning`` and default to world scales, preserving prior behavior.
