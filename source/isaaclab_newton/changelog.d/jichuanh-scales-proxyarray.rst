Changed
^^^^^^^

* **Breaking:** :meth:`~isaaclab_newton.sim.views.NewtonSiteFrameView.get_scales`
  now returns a :class:`~isaaclab.utils.warp.ProxyArray`, matching the updated
  :class:`~isaaclab.sim.views.BaseFrameView` contract. Callers that fed the
  return value into Warp kernels or ``set_scales`` need to extract the
  underlying array via ``.warp``.
