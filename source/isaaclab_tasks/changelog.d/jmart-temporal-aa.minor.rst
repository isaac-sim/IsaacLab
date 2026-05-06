Added
^^^^^

* Added :class:`~isaaclab_tasks.utils.presets.MultiBackendCameraCfg` and
  :class:`~isaaclab_tasks.utils.presets.FrameStackPolicyCfg` for AND-conditioned
  frame stacking that activates only on the Newton + Warp combination.
* Added a runtime auto-apply step in
  :func:`~isaaclab_tasks.utils.sim_launcher.launch_simulation` that propagates
  ``frame_stack_policy`` onto ``frame_stack`` when the user has not set an
  explicit value (sentinel = 0).
* Added a runtime warning in
  :func:`~isaaclab_tasks.utils.sim_launcher.launch_simulation` when the active
  physics backend declares ``requires_temporal_camera_data`` but the renderer
  does not provide it and no camera has frame stacking enabled.
