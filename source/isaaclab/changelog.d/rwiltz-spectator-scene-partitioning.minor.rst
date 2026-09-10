Fixed
^^^^^

* Fixed partitioned environments disappearing from the Kit viewport and the XR view. RTX culls
  geometry carrying ``omni:scenePartition`` from cameras without a matching token, and the Kit
  builds shipped with the pinned Isaac Sim do not implement
  ``/rtx/scenePartitioning/showAllPartitionsByDefault``. Launches that render such a view now
  default per-environment scene partitioning off. Set
  ``ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=1`` or
  :attr:`~isaaclab_physx.renderers.IsaacRtxRendererCfg.enable_scene_partitioning` to keep
  partitioning for these launches.

Added
^^^^^

* Added :func:`~isaaclab.utils.renderers.set_isaac_rtx_per_env_scene_partition_default` to set the
  Isaac RTX scene-partitioning default used when
  ``ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION`` is unset.
