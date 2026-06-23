Added
^^^^^

* Added :meth:`~isaaclab_newton.sim.views.NewtonSiteFrameView.get_local_scales`
  and :meth:`~isaaclab_newton.sim.views.NewtonSiteFrameView.get_world_scales`
  for reading transform (xform) scales.  Scale writes go through the writer
  scope (see the ``xform-space-writer`` fragment).  These transform-scale
  APIs are intentionally separate from Newton collision shape geometry
  sizes.

Deprecated
^^^^^^^^^^

* Deprecated :meth:`~isaaclab_newton.sim.views.NewtonSiteFrameView.get_scales`
  and :meth:`~isaaclab_newton.sim.views.NewtonSiteFrameView.set_scales` in favor
  of the explicit transform-scale getters ``get_world_scales`` /
  ``get_local_scales`` (and the writer scope's ``set_scales``).  The
  deprecated methods still work but emit a ``DeprecationWarning`` and
  preserve Newton's legacy collision shape geometry-scale behavior.
