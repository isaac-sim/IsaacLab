* **Breaking:** Required visualizer streaming cameras to be declared in the scene before cloning.
  Moved camera pose, renderer, and lifetime ownership out of visualizers. Replaced
  ``streaming_cam_target_prim_path``, ``streaming_cam_eye``, and ``streaming_cam_renderer_cfg``
  with a scene ``CameraCfg`` and ``streaming_sensor_prim_path`` selection. Removed the corresponding
  deprecated ``tiled_cam_*`` aliases and generated-camera helpers in
  ``isaaclab.envs.utils.camera_view``. Declared streaming channels must exist on the selected camera.
* Shared streaming-image composition across visualizers and batched selected camera tiles before
  transferring each channel to the host.
* Added ``PerspectiveCameraCfg`` and ``SceneCameraCfg`` for selecting visualizer display sources.
* **Breaking:** Shared ``VisualizerCfg.cameras`` across visualizers, retaining ``SceneCameraCfg`` as a reference
  to an existing sensor. Passed the source USD stage and clone plan into visualizer initialization;
  custom visualizers must accept the ``stage`` and ``clone_plan`` keyword arguments and pass them
  to ``super().initialize()``.
* **Breaking:** Removed the deprecated ``tiled_cam_view``, ``tiled_cam_num``, ``tiled_cam_env_indices``, and
  ``tiled_cam_prim_path`` aliases. Use ``streaming_view``, ``streaming_envs`` (count or list), and
  ``streaming_sensor_prim_path`` or ``cameras=[SceneCameraCfg(...)]`` instead.
