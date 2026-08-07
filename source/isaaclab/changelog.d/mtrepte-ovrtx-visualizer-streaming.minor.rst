Added
^^^^^

* Added ``newton_gl`` and ``newton_rtx`` as canonical ``--viz`` / ``--visualizer`` CLI values,
  replacing the old ``newton`` alias (which now emits a deprecation warning and resolves to
  ``newton_gl``).
* Added :class:`~isaaclab.visualizers.VisualizerCfg` streaming camera fields
  (``streaming_view``, ``streaming_gt_types``, ``streaming_envs``,
  ``streaming_cam_target_prim_path``, ``streaming_cam_eye``, ``streaming_cam_renderer``,
  ``streaming_sensor_prim_path``, ``streaming_depth_min``, ``streaming_depth_max``)
  replacing the removed ``tiled_cam_*`` fields.
* Added :mod:`~isaaclab.envs.utils.camera_colorizer` with
  :class:`~isaaclab.envs.utils.camera_colorizer.CameraFrameColorizer` for converting raw
  depth, segmentation, and normals tensors into displayable RGB frames.
* Added :func:`~isaaclab.envs.utils.camera_view.camera_gt_batch`,
  :func:`~isaaclab.envs.utils.camera_view.compose_streaming_grid`,
  :func:`~isaaclab.envs.utils.camera_view.resolve_streaming_envs`, and
  :func:`~isaaclab.envs.utils.camera_view.create_visualizer_camera` to
  :mod:`~isaaclab.envs.utils.camera_view`.

Changed
^^^^^^^

* :class:`~isaaclab.envs.utils.video_recorder.VideoRecorder` ``source`` field now accepts
  ``"visualizer:newton"`` as an alias for ``"visualizer:newton_gl"`` to ease migration.
* :func:`~isaaclab.envs.utils.camera_view.compose_streaming_grid` now prioritises balanced
  grid rows (minimal empty cells in the last row) over aspect-ratio optimisation, preventing
  ragged layouts such as 3+1 for 4 environments.
* :meth:`~isaaclab.sim.SimulationContext._apply_default_visualizer_cfg` now only propagates
  fields that were explicitly set in ``default_visualizer_cfg`` (i.e. differ from the base
  :class:`~isaaclab.visualizers.VisualizerCfg` defaults), preventing base-class defaults such
  as ``streaming_view=False`` from overriding backend-specific defaults like
  :attr:`~isaaclab_visualizers.newton.NewtonGLVisualizerCfg.streaming_view` ``= True``.

Deprecated
^^^^^^^^^^

* ``--viz newton`` is deprecated; use ``--viz newton_gl`` instead.
* :class:`~isaaclab.envs.common.ViewerCfg` is deprecated; configure visualizers via
  :class:`~isaaclab.visualizers.VisualizerCfg` in ``SimulationCfg.visualizer_cfgs`` instead.
* ``tiled_cam_*`` fields on :class:`~isaaclab.visualizers.VisualizerCfg` are deprecated;
  they now emit :class:`DeprecationWarning` and forward to the equivalent ``streaming_*``
  field for one release before removal.
