Changed
^^^^^^^

* Changed :class:`~isaaclab_visualizers.kit.KitVisualizer` to skip authoring the
  ``omni:scenePartition`` attribute on the viewport camera by default. Set
  ``ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=1`` to re-enable per-environment
  scene partitioning for the Kit viewport camera.
