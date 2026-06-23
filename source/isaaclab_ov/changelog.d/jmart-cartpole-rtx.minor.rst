Added
^^^^^

* Overrode :meth:`provides_temporal_camera_data` on :class:`OVRTXRenderer` to return ``True``
  only for the ``rgb``/``rgba`` beauty buffer (temporally accumulated by DLSS), matching Isaac RTX;
  other AOVs return ``False``.
