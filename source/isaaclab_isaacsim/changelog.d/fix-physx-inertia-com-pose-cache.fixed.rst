* Fixed :meth:`~isaaclab_physx.assets.Articulation.set_inertias_index`,
  :meth:`~isaaclab_physx.assets.Articulation.set_inertias_mask`,
  :meth:`~isaaclab_physx.assets.RigidObject.set_inertias_index` and
  :meth:`~isaaclab_physx.assets.RigidObject.set_inertias_mask` leaving the cached center-of-mass pose stale.
  PhysX recomputes the principal-axis rotation when inertia is set, so
  :attr:`~isaaclab_physx.assets.ArticulationData.body_com_quat_b`, :attr:`~isaaclab_physx.assets.RigidObjectData.body_com_quat_b`
  and the world-frame center-of-mass quantities derived from them kept the old principal axes after writing a
  rotated, non-diagonal inertia. The writes now invalidate the cache, as
  :class:`~isaaclab_physx.assets.RigidObjectCollection` already did.
