Added
^^^^^

* Added ``IsaacRtxRendererCfg.enable_scene_partitioning`` and
  ``IsaacRtxRendererGlobalSettingsCfg.show_all_partitions_by_default`` settings.
  The latter optionally overrides AppLauncher's visualization-scoped spectator
  setting and requires spatially separated environments when enabled.

Changed
^^^^^^^

* Changed :meth:`~isaaclab_physx.renderers.IsaacRtxRenderer.prepare_stage` to
  author per-environment scene-partition attributes according to
  ``IsaacRtxRendererCfg.enable_scene_partitioning``, which defaults to enabled.
  Set ``IsaacRtxRendererCfg(enable_scene_partitioning=False)`` to preserve the
  previous unpartitioned behavior.
