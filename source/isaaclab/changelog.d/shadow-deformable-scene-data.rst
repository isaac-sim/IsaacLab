Added
^^^^^

* Added :class:`~isaaclab.scene_data.SceneDataFormat.Points` and geometry mapping helpers on
  :class:`~isaaclab.scene_data.SceneDataProvider` for deformable nodal position sync.

Fixed
^^^^^

* Fixed :meth:`~isaaclab.scene_data.SceneDataProvider.get_points` overflowing consumer
  particle buffers by clamping entity copies to destination capacity, and hardened
  deformable vertex-count path resolution used by PhysX/OVPhysX scene-data backends.
