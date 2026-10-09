* Added ``create``, ``eye``, ``lookat``, ``focal_length``, ``resolution``, ``data_types``, ``renderer_cfg``,
  ``track_path``, ``follow_heading``, and ``heading_smoothing_time_constant`` to
  :class:`~isaaclab.visualizers.SceneCameraCfg`. With ``create=True`` the visualizer declares the scene camera,
  fixed in each environment or following an asset with optional smoothed yaw, and shows it in the streaming view.
  The launcher adds it to the scene only when a visualizer or a ``streaming_view`` video uses it.
