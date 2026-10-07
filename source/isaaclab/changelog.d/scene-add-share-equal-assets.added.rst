* Added ``share_equal_assets`` to :func:`isaaclab.scene.add`. With ``False``, an asset equal to one of the target
  scene keeps its own binding instead of spanning both scenes' environments. The heterogeneous-scene example uses it
  on Newton FeatherPGS, which needs each asset's joints and bodies regularly spaced between environments.
