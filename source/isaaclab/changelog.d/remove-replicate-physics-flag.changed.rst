* ``randomize_rigid_body_scale`` now rejects multi-environment rigid bodies because their per-environment
  USD scale edits do not reach replicated physics. Declare size variants with ``MultiAssetSpawnerCfg``.
