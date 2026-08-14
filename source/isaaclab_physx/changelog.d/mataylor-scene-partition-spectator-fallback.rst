Fixed
^^^^^

* Fixed an empty Kit viewport when per-environment scene partitioning ran on an Isaac Sim
  release whose RTX renderer does not implement the all-partitions spectator view
  (``/rtx/scenePartitioning/showAllPartitionsByDefault``, added in Isaac Sim 6.1). Interactive
  viewport cameras inherit no ``omni:scenePartition`` token, so on those runtimes they matched
  no partition and rendered nothing. :meth:`~isaaclab_physx.renderers.IsaacRtxRenderer.prepare_stage`
  now leaves the stage unpartitioned when the spectator view is requested but unsupported. Set
  ``IsaacRtxRendererCfg.global_settings.show_all_partitions_by_default`` to ``False`` to keep
  per-environment isolation on those runtimes; the viewport is then bound to a single environment.
