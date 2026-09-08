Changed
^^^^^^^

* Changed ``IsaacRtxRenderer.render()`` to pass the tiled annotator buffer to
  ``reshape_tiled_image`` as a 3D array instead of flattening it to 1D. Large environment counts
  and camera resolutions no longer overflow the maximum size of a single Warp array dimension.
