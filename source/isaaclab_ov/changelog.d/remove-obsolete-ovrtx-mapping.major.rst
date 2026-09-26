Removed
^^^^^^^

* **Breaking:** Removed the unused ``isaaclab_ov.renderers.ovrtx_mapping`` module and
  ``map_attribute_for_warp_writes`` export. Use a persistent caller-owned Warp buffer and pass it to
  ``binding.write(..., cuda_stream=<producing Warp stream>)``. If mapping is unavoidable, pass the
  producing stream explicitly to ``unmap(stream=...)``.
