Changed
^^^^^^^

* Changed :class:`~isaaclab_ov.renderers.OVRTXRenderer` to consume ``wp.array``
  camera output buffers and camera state arrays from
  :class:`~isaaclab.renderers.BaseRenderer`. Use :func:`warp.to_torch` on
  ``camera.data.output`` entries if Torch tensor operations are required.
* Changed :class:`~isaaclab_ov.renderers.OVRTXRenderer` to consume ``wp.array``
  camera output buffers and camera state arrays from
  :class:`~isaaclab.renderers.BaseRenderer`. Use :func:`warp.to_torch` on
  ``camera.data.output`` entries if Torch tensor operations are required.
