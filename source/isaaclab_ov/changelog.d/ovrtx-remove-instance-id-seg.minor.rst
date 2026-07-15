Changed
^^^^^^^

* Removed support for ``instance_id_segmentation_fast`` from the OVRTX renderer.
  Requesting this data type via :class:`~isaaclab_ov.renderers.OVRTXRendererCfg` will now raise an
  error at camera allocation time. Use :class:`~isaaclab_physx.renderers.IsaacRtxRendererCfg` if
  ``instance_id_segmentation_fast`` is required.
