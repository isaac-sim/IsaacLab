Fixed
^^^^^

* Fixed :meth:`~isaaclab_physx.assets.RigidObjectCollection.write_body_link_pose_to_sim_mask`,
  :meth:`~isaaclab_physx.assets.RigidObjectCollection.write_body_com_pose_to_sim_mask`,
  :meth:`~isaaclab_physx.assets.RigidObjectCollection.write_body_com_velocity_to_sim_mask`, and
  :meth:`~isaaclab_physx.assets.RigidObjectCollection.write_body_link_velocity_to_sim_mask` not accepting the
  ``body_mask`` argument that the base class and the other backends declare.

Deprecated
^^^^^^^^^^

* Deprecated the ``body_ids`` argument of the :class:`~isaaclab_physx.assets.RigidObjectCollection` mask writers
  listed above. Pass a boolean ``body_mask`` of shape (num_bodies,) instead, or use the ``*_index`` writers with
  ``body_ids``.
