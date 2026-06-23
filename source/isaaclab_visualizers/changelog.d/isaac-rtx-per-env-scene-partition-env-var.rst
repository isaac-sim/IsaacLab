Changed
^^^^^^^

* Changed :meth:`~isaaclab_visualizers.kit.KitVisualizer._apply_viewport_camera_scene_partition` to
  skip tagging the viewport camera with an ``omni:scenePartition`` token by default. Set the
  environment variable ``ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=1`` to re-enable
  per-environment scene partitioning for the Kit viewport camera.
