Changed
^^^^^^^

* **Breaking:** :meth:`~isaaclab.sim.views.BaseFrameView.get_scales` now returns
  a :class:`~isaaclab.utils.warp.ProxyArray` instead of a raw ``wp.array``,
  matching :meth:`~isaaclab.sim.views.BaseFrameView.get_world_poses` and
  :meth:`~isaaclab.sim.views.BaseFrameView.get_local_poses`. Callers that
  passed the return value straight into Warp kernels or ``set_scales`` need to
  extract the underlying array via ``.warp``; callers that read ``.torch`` are
  unaffected.
