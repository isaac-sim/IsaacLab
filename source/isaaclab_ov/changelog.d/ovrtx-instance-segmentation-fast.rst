Added
^^^^^

* Added :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.colorize_instance_segmentation` and
  :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.colorize_instance_id_segmentation` config fields
  to :class:`~isaaclab_ov.renderers.OVRTXRendererCfg`.
* Added support for the ``instance_segmentation_fast`` and ``instance_id_segmentation_fast``
  data types in the OVRTX renderer, via the ``NonStableInstanceSegmentation`` and
  ``InstanceSegmentationSD`` AOVs respectively. When the corresponding
  :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.colorize_instance_segmentation` /
  :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.colorize_instance_id_segmentation` flag is
  ``True`` (default), instance IDs are colorized and returned as ``uint8`` RGBA; when ``False``,
  raw ``uint32`` instance IDs are returned.
