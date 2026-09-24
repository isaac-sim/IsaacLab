Changed
^^^^^^^

* Derived OVRTX cloned camera paths from the authored absolute source path in
  ``CameraRenderSpec.camera_prim_paths``. Remove the ``camera_path_relative_to_env_0`` argument
  when constructing render specs; OVRTX validated the source path under
  ``/World/envs/env_0/`` and resolved the per-environment paths internally.
