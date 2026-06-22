Added
^^^^^

* Overrode :meth:`provides_temporal_camera_data` on :class:`IsaacRtxRenderer` to return ``True``
  only for the ``rgb``/``rgba`` beauty buffer (temporally accumulated by DLSS); the depth, albedo,
  simple_shading, and segmentation AOVs return ``False`` as they bypass DLSS.
