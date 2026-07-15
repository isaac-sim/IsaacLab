Changed
^^^^^^^

* Removed support for ``instance_id_segmentation_fast`` from the OVRTX renderer, as it has no
  real-world sensor equivalent. Requesting this data type via
  :class:`~isaaclab_ov.renderers.OVRTXRendererCfg` will now raise an error at camera allocation
  time. Use ``instance_segmentation_fast`` or ``semantic_segmentation`` instead.
