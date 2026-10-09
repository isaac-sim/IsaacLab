* **Breaking:** Required visualizer streaming cameras to be declared in the scene before cloning.
  Moved camera pose, renderer, and lifetime ownership out of visualizers. Replaced
  ``streaming_cam_target_prim_path``, ``streaming_cam_eye``, and ``streaming_cam_renderer_cfg``
  with a scene ``CameraCfg`` and ``streaming_sensor_prim_path`` selection. Removed the corresponding
  deprecated ``tiled_cam_*`` aliases and generated-camera helpers in
  ``isaaclab.envs.utils.camera_view``. Declared streaming channels must exist on the selected camera.
* Shared device-image composition across visualizers and restricted host transfers to consumers
  requesting the composed RGB image. Cached environment indices and depth colors directly in the visualizer
  and composed explicit device arrays with ``isaaclab.utils.images.compose_image``.
  Specialized RGB, depth, normals, and segmentation kernels at compilation instead of branching on the
  display channel inside each kernel launch.
* Added ``PerspectiveCameraCfg`` and ``SceneCameraCfg`` for selecting visualizer display sources.
* **Breaking:** Shared ``VisualizerCfg.cameras`` across visualizers, retaining ``SceneCameraCfg`` as a reference
  to an existing sensor. Resolved camera references before visualizer initialization;
  custom visualizers must accept the ``cameras`` and ``stage`` keyword arguments and pass them
  to ``super().initialize()``. Passed borrowed camera objects rather than cloning plans.
* Resolved the final visible environment selection in ``BaseVisualizer.initialize()`` so
  ``get_visualized_env_ids()`` returned the same selection used by viewers, markers, and camera tiles.
* **Breaking:** Removed the deprecated ``tiled_cam_view``, ``tiled_cam_num``, ``tiled_cam_env_indices``, and
  ``tiled_cam_prim_path`` aliases. Use ``streaming_view``, ``streaming_envs`` (count or list), and
  ``streaming_sensor_prim_path`` or ``cameras=[SceneCameraCfg(...)]`` instead.
