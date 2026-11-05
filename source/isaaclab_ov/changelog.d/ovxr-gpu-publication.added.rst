* Added :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.active_cuda_gpus` to select the native
  rendering GPUs and :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.geometry_publication_device` to
  publish rigid transforms and deformable points through completed copies on the rendering GPU,
  for applications using separate physics and rendering devices. Camera outputs rendered on another GPU
  are copied to the simulation device before extraction. Malformed values are rejected
  when the configuration is created; :func:`~isaaclab_ov.renderers.parse_active_cuda_gpus` and
  :func:`~isaaclab_ov.renderers.parse_cuda_device_index` apply the same checks beforehand. The
  configuration and :class:`~isaaclab_ov.renderers.OVRTXRenderer` reject a publication device that is
  not a rendering GPU: one of ``active_cuda_gpus``, or the simulation GPU when that is unset.
* Added ``python -m isaaclab_ov.renderers.check_cuda_peer``, a preflight that checks the geometry
  publication copy between two GPUs before splitting physics and rendering across them.
