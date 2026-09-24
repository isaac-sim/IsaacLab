Removed
^^^^^^^

* **Breaking:** Removed ``CameraRenderSpec.camera_path_relative_to_env_0``. Remove this argument
  from render-spec constructors and use the absolute paths in ``camera_prim_paths`` instead.
  OVRTX derived cloned camera paths from the authored source camera internally.
