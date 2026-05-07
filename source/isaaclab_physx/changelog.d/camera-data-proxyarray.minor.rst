Changed
^^^^^^^

* **Breaking:** :class:`~isaaclab_physx.renderers.IsaacRtxRenderer` now writes
  rendered tiles into ``dict[str, wp.array]`` output buffers (was
  ``dict[str, torch.Tensor]``).
  :meth:`~isaaclab_physx.renderers.IsaacRtxRenderer.set_outputs` and
  :meth:`~isaaclab_physx.renderers.IsaacRtxRenderer.update_camera` accept
  :class:`warp.array` arguments to match the new
  :class:`~isaaclab.renderers.base_renderer.BaseRenderer` contract. The rgb
  alias of rgba is preallocated as a torch view in
  :meth:`~isaaclab.sensors.camera.CameraData.allocate` rather than rebound at
  render time.
