Changed
^^^^^^^

* Changed the lift task progress rewards and ``object_ee_distance`` to use masked ``torch.where``
  updates and a cached device index tensor instead of boolean-mask indexing and Python-list indexing,
  removing five stream synchronizations per environment step, and fused the camera observation
  normalization into a single conversion pass.
