Fixed
^^^^^

* Fixed :meth:`~isaaclab_physx.assets.RigidObjectCollection.set_coms_index` raising a ``KeyError`` because the
  center-of-mass poses were passed to the PhysX tensor API as ``wp.transformf`` instead of ``float32``.
* Fixed :meth:`~isaaclab_physx.assets.RigidObjectCollection.set_coms_index` leaving the unselected entries of
  :attr:`~isaaclab_physx.assets.RigidObjectCollectionData.body_com_pose_b` zeroed when it was called for a subset of
  bodies or environments before the property was first read.
* Fixed :meth:`~isaaclab_physx.assets.RigidObjectCollection.set_inertias_index` failing in the PhysX backend because
  the inertias were passed to the PhysX tensor API with the wrong layout.
