Changed
^^^^^^^

* Changed :meth:`~isaaclab_physx.renderers.IsaacRtxRenderer.prepare_stage` to skip authoring
  ``primvars:omni:scenePartition`` and ``omni:scenePartition`` by default. Set the environment
  variable ``ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=1`` to re-enable
  per-environment scene partitioning for Isaac RTX rendering.
