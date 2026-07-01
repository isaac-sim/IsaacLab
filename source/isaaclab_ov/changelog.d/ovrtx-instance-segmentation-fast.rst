Added
^^^^^

* Added :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.colorize_instance_segmentation` config field
  to :class:`~isaaclab_ov.renderers.OVRTXRendererCfg`.
* Added support for ``instance_segmentation_fast`` data type in the OVRTX renderer via the
  ``NonStableInstanceSegmentation`` AOV. When
  :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.colorize_instance_segmentation` is ``True``
  (default), instance IDs are colorized and returned as ``uint8`` RGBA; when ``False``, raw
  ``uint32`` instance IDs are returned.
