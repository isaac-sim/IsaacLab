Fixed
^^^^^

* Fixed ``IsaacRtxRenderer.render()`` crashing with ``RuntimeError: Invalid indexing in slice``
  when an annotator's channel buffer has not warmed up yet (e.g. right after attach, at env
  creation) for the ``motion_vectors``, ``normals``, simple-shading, and RGB HDR data types.

Changed
^^^^^^^

* Changed ``IsaacRtxRenderer.render()`` to pass the tiled annotator buffer to
  ``reshape_tiled_image`` as a 3D array instead of flattening it to 1D. Large environment counts
  and camera resolutions no longer overflow the maximum size of a single Warp array dimension.
