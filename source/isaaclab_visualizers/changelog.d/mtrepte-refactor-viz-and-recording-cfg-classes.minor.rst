Added
^^^^^

* Added ``origin_type``, ``origin_env_index``, and ``origin_track_path`` fields to
  :class:`~isaaclab_visualizers.kit.KitVisualizerCfg`. These replace the removed
  :class:`~isaaclab.envs.common.ViewerCfg` / ``ViewportCameraController`` and allow the Kit
  viewport camera to track a world origin, an environment origin, or an asset root / body
  across simulation steps.  Tracking path format: ``"robot"`` for asset root,
  ``"robot/panda_hand"`` for a specific body.
