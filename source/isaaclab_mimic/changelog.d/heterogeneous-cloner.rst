Fixed
^^^^^

* Fixed ``SceneAsset`` pose queries in the locomanipulation SDG utilities to
  build their frame view on demand, since static scene assets no longer carry
  a runtime view in :class:`~isaaclab.scene.InteractiveScene`.
