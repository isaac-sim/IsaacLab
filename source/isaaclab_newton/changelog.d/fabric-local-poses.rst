Added
^^^^^

* Added :meth:`~isaaclab_newton.sim.views.NewtonSiteFrameView.get_local_scales`,
  :meth:`~isaaclab_newton.sim.views.NewtonSiteFrameView.set_local_scales`,
  :meth:`~isaaclab_newton.sim.views.NewtonSiteFrameView.get_world_scales`, and
  :meth:`~isaaclab_newton.sim.views.NewtonSiteFrameView.set_world_scales` for
  transform (xform) scales.  These explicit APIs are intentionally separate from
  Newton collision shape geometry sizes.

Deprecated
^^^^^^^^^^

* Deprecated :meth:`~isaaclab_newton.sim.views.NewtonSiteFrameView.get_scales`
  and :meth:`~isaaclab_newton.sim.views.NewtonSiteFrameView.set_scales` in favor
  of the explicit xform-scale ``get_world_scales`` / ``set_world_scales`` (or
  their local equivalents).  The deprecated methods still work but emit a
  ``DeprecationWarning`` and preserve Newton's legacy collision shape
  geometry-scale behavior.
