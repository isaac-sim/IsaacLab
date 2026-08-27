Changed
^^^^^^^

* **Breaking:** Changed :class:`~isaaclab_physx.renderers.IsaacRtxRenderer` to reject ``rgb`` or
  ``rgba`` combined with ``simple_shading_*`` because RTX Minimal mode applies to the entire
  render product. Move incompatible outputs to separate cameras.
