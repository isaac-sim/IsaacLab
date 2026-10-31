* Added :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.active_cuda_gpus` to select the native
  rendering GPUs and :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.geometry_publication_device` to
  publish rigid transforms and deformable points through completed copies on the rendering GPU,
  for applications using separate physics and rendering devices. Malformed values are rejected
  when the configuration is created.
