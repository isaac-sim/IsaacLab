Changed
^^^^^^^

* **Breaking:** Required visualizer streaming cameras to be declared in the scene before cloning.
  Moved camera pose, renderer, and lifetime ownership out of visualizers. Replaced
  ``streaming_cam_target_prim_path``, ``streaming_cam_eye``, and ``streaming_cam_renderer_cfg``
  with a scene ``CameraCfg`` and ``streaming_sensor_prim_path`` selection. Removed the corresponding
  ``tiled_cam_eye`` and ``tiled_cam_target_prim_path`` aliases and generated-camera helpers in
  ``isaaclab.envs.utils.camera_view``. Declared streaming channels must exist on the selected camera.
* Shared streaming-image composition across visualizers and batched selected camera tiles before
  transferring each channel to the host.
* Changed ``VisualizerCfg.background_color`` to default to ``None``, preserving the scene HDR in
  Kit and Newton RTX or the procedural sky in Newton GL. Set ``background_color=(0.3, 0.55, 0.82)``
  to retain the previous solid sky-blue background.
* Added ``PerspectiveCameraCfg`` and ``SceneCameraCfg`` for selecting visualizer display sources.
* Selected visualizers through ``SimulationCfg.visualizer_cfgs`` presets before launch, removing
  runtime viewer-name factories and settings-based selection. Used ``visualizer=NAME`` or its
  ``--viz`` / ``--visualizer`` aliases; declared custom alternatives with ``PresetCfg``.
* Moved config-only ``PresetCfg``, ``preset``, and ``resolve_presets`` into ``isaaclab.utils``;
  retained the existing ``isaaclab_tasks.utils`` imports.

Fixed
^^^^^

* Invalidated camera images after explicit pose writes so lazy reads refreshed pixels even without
  advancing simulation time.
* Applied distributed and runtime-selected devices to standalone ``SimulationCfg`` inputs as well
  as environment configs in ``launch_simulation``.
