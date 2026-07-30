Fixed
^^^^^

* Fixed a quadratic path lookup in :class:`~isaaclab.scene_data.SceneDataProvider` transform
  mapping that stalled setup at high rigid-body counts (thousands of environments). The
  per-item ``list.index`` scan is now an ``O(N)`` dictionary lookup.
