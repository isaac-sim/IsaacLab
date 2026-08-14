Added
^^^^^

* Added ``IsaacRtxRendererCfg.enable_scene_partitioning`` and
  ``IsaacRtxRendererGlobalSettingsCfg.show_all_partitions_by_default`` settings.
  The latter enables an all-environment spectator view by default and requires
  spatially separated environments.

Changed
^^^^^^^

* Changed :meth:`~isaaclab_physx.renderers.IsaacRtxRenderer.prepare_stage` to
  author per-environment scene-partition attributes according to
  ``IsaacRtxRendererCfg.enable_scene_partitioning``, which defaults to enabled.
