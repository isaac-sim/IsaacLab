* Fixed :meth:`~isaaclab_physx.assets.RigidObjectCollection.set_coms_index` and
  :meth:`~isaaclab_physx.assets.RigidObjectCollection.set_coms_mask` raising a ``KeyError`` because the
  center-of-mass poses were passed to the PhysX tensor API as ``wp.transformf`` instead of ``float32``.
* Fixed :meth:`~isaaclab_physx.assets.RigidObjectCollection.set_coms_index` and
  :meth:`~isaaclab_physx.assets.RigidObjectCollection.set_coms_mask` leaving the unselected entries of
  :attr:`~isaaclab_physx.assets.RigidObjectCollectionData.body_com_pose_b` zeroed when they were called for a
  subset of bodies or environments before the property was first read.
* Fixed :meth:`~isaaclab_physx.assets.RigidObjectCollection.set_inertias_index` and
  :meth:`~isaaclab_physx.assets.RigidObjectCollection.set_inertias_mask` failing in the PhysX backend because the
  inertias were passed to the PhysX tensor API with the wrong layout.
