Changed
^^^^^^^

* Changed :class:`~isaaclab_physx.renderers.IsaacRtxRenderer` to consume
  ``wp.array`` camera output buffers and camera state arrays from
  :class:`~isaaclab.renderers.BaseRenderer`. Use :func:`warp.to_torch` on
  ``camera.data.output`` entries if Torch tensor operations are required.
* Updated PhysX PVA debug visualization to convert camera-convention
  orientation outputs with :func:`warp.to_torch`.
