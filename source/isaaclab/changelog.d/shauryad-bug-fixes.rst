Changed
^^^^^^^

* Changed the ``reshape_tiled_image`` Warp kernel to index the tiled image buffer as a 3D array
  of shape (num_tiles_y * image_height, num_tiles_x * image_width, num_channels) instead of a
  flattened 1D array. This keeps every array dimension within Warp's per-dimension size limit, so
  large environment counts and camera resolutions no longer overflow a single flattened dimension.
