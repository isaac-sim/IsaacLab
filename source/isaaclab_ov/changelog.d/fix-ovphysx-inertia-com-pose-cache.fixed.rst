* Fixed :meth:`~isaaclab_ov.assets.Articulation.set_inertias_index`,
  :meth:`~isaaclab_ov.assets.Articulation.set_inertias_mask`,
  :meth:`~isaaclab_ov.assets.RigidObject.set_inertias_index` and
  :meth:`~isaaclab_ov.assets.RigidObject.set_inertias_mask` leaving the cached center-of-mass pose stale.
  OVPhysX recomputes the principal-axis rotation when inertia is set, so ``body_com_quat_b`` and the
  world-frame center-of-mass quantities derived from it kept the old principal axes after writing a rotated,
  non-diagonal inertia. The writes now invalidate the cache.
