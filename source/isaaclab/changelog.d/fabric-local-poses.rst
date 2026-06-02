Added
^^^^^

* Added explicit local/world scale methods
  :meth:`~isaaclab.sim.views.BaseFrameView.get_local_scales`,
  :meth:`~isaaclab.sim.views.BaseFrameView.set_local_scales`,
  :meth:`~isaaclab.sim.views.BaseFrameView.get_world_scales`, and
  :meth:`~isaaclab.sim.views.BaseFrameView.set_world_scales` to the FrameView
  API, implemented for :class:`~isaaclab.sim.views.UsdFrameView`.  Scale getters
  now return :class:`~isaaclab.utils.warp.ProxyArray`.

* Added :func:`~isaaclab.utils.warp.fabric.decompose_indexed_fabric_transforms`,
  :func:`~isaaclab.utils.warp.fabric.compose_indexed_fabric_transforms`,
  :func:`~isaaclab.utils.warp.fabric.update_indexed_local_matrix_from_world`, and
  :func:`~isaaclab.utils.warp.fabric.update_indexed_world_matrix_from_local`
  Warp kernels operating on :class:`wp.indexedfabricarray` for reading and
  writing Fabric ``Matrix4d`` attributes (``omni:fabric:worldMatrix`` /
  ``omni:fabric:localMatrix``).

Deprecated
^^^^^^^^^^

* Deprecated :meth:`~isaaclab.sim.views.BaseFrameView.get_scales` and
  :meth:`~isaaclab.sim.views.BaseFrameView.set_scales` in favor of the explicit
  ``get_local_scales`` / ``set_local_scales`` (operates on ``xformOp:scale``) or
  ``get_world_scales`` / ``set_world_scales`` (operates on composed world-space
  scale).  The deprecated methods still work but emit a ``DeprecationWarning``;
  :class:`~isaaclab.sim.views.UsdFrameView` preserves prior behavior by
  defaulting to local scales.
