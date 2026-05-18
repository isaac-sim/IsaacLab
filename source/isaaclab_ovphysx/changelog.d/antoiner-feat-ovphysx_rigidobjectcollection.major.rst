Added
^^^^^

* Added :class:`~isaaclab_ovphysx.assets.RigidObjectCollection` and
  :class:`~isaaclab_ovphysx.assets.RigidObjectCollectionData` for the
  OVPhysX backend, completing the rigid-body asset surface alongside
  :class:`~isaaclab_ovphysx.assets.RigidObject` and
  :class:`~isaaclab_ovphysx.assets.Articulation`. Supports
  ``(env, body)`` dual indexing and per-body property setters. Uses the
  ovphysx 0.4.3+ native fused multi-prim binding API
  (``create_tensor_binding(prim_paths=[...])``) so one binding spans all
  ``num_instances * num_bodies`` prims per tensor type, mirroring the
  strided-view reshape pattern used by the PhysX collection.
