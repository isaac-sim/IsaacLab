* Changed :class:`~isaaclab_physx.renderers.IsaacRtxRenderer` to mark RTX sensor rendering when it is created
  and to route Gaussian HDR in :meth:`~isaaclab_physx.renderers.IsaacRtxRenderer.prepare_cameras` for cameras
  that request HDR color, instead of :class:`~isaaclab.sensors.camera.Camera` setting them.
