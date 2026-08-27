Changed
^^^^^^^

* **Breaking:** Changed :class:`~isaaclab_physx.renderers.IsaacRtxRenderer` to reject ``rgb`` or
  ``rgba`` combined with ``simple_shading_*``, or more than one distinct ``simple_shading_*``
  data type, because RTX Minimal mode applies to the entire render product. Move incompatible
  outputs to separate cameras. Repeated identical simple-shading requests remain supported.
