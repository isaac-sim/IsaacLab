Changed
^^^^^^^

* **Breaking:** Updated :class:`~isaaclab_ov.renderers.OVRTXRenderer` to use the renamed
  ``"instance_segmentation"`` data type key (previously ``"instance_segmentation_fast"``).
  Output buffer and ``camera.data.info`` dict keys now use the new name.

Fixed
^^^^^

* Fixed :class:`~isaaclab_ov.renderers.OVRTXRenderer` to return ``int32`` instance IDs (shape
  ``(B, H, W, 1)``) when ``colorize_instance_segmentation=False``, matching the Isaac RTX renderer.
  Previously the non-colorized path incorrectly declared ``uint32``.
