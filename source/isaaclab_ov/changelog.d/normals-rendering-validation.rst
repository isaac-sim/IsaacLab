Added
^^^^^

* Added ``"normals"`` support to :class:`~isaaclab_ov.renderers.OVRTXRenderer`. The renderer now
  declares :attr:`~isaaclab.renderers.RenderBufferKind.NORMALS` in
  :meth:`~isaaclab_ov.renderers.OVRTXRenderer.supported_output_types` (3-channel ``float32``) and
  extracts the ``NormalSD`` AOV from each rendered frame into the output buffer.
