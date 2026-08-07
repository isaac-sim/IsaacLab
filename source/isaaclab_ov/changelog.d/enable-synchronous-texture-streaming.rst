Changed
^^^^^^^

* Enabled synchronous texture streaming on the internal OVRTX ``RendererConfig`` in
  :class:`~isaaclab_ov.renderers.ovrtx_renderer.OVRTXRenderer` via
  ``texture_streaming_mode=TextureStreamingMode.SYNCHRONOUS`` to improve cross-run
  render determinism. Requires ``ovrtx>=0.4.1``.
