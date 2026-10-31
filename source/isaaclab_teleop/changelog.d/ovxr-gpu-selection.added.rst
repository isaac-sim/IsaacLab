* Added the ``--ovrtx_active_cuda_gpus`` and ``--geometry_publication_device`` OVXR launch options
  for rendering GPU selection and geometry publication placement.
* Added ``python -m isaaclab_teleop.check_cuda_peer``, a preflight that checks the geometry
  publication copy between two GPUs before a split-GPU session.
