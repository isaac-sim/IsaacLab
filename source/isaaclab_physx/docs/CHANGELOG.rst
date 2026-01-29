Changelog
---------

0.1.0 (2026-01-28)
~~~~~~~~~~~~~~~~~~~

This is the initial release of the ``isaaclab_physx`` extension, which provides PhysX-specific
implementations of Isaac Lab asset classes. This extension enables a multi-backend architecture
where simulation backend-specific code is separated from the core Isaac Lab API.

Added
^^^^^

* Added :mod:`isaaclab_physx.assets` module containing PhysX-specific asset implementations:

  * :class:`~isaaclab_physx.assets.Articulation` and :class:`~isaaclab_physx.assets.ArticulationData`:
    PhysX-specific implementation for articulated rigid body systems (e.g., robots). Extends
    :class:`~isaaclab.assets.BaseArticulation` with PhysX tensor API integration for efficient
    GPU-accelerated simulation of multi-joint systems.

  * :class:`~isaaclab_physx.assets.RigidObject` and :class:`~isaaclab_physx.assets.RigidObjectData`:
    PhysX-specific implementation for single rigid body assets. Extends
    :class:`~isaaclab.assets.BaseRigidObject` with PhysX tensor API for efficient rigid body
    state queries and writes.

  * :class:`~isaaclab_physx.assets.RigidObjectCollection` and :class:`~isaaclab_physx.assets.RigidObjectCollectionData`:
    PhysX-specific implementation for collections of rigid objects. Extends
    :class:`~isaaclab.assets.BaseRigidObjectCollection` with batched ``(env_ids, object_ids)``
    API for efficient multi-object state management.

  * :class:`~isaaclab_physx.assets.DeformableObject`, :class:`~isaaclab_physx.assets.DeformableObjectCfg`,
    and :class:`~isaaclab_physx.assets.DeformableObjectData`: PhysX-specific implementation for
    soft body simulation using finite element methods (FEM). Moved from ``isaaclab.assets``.

  * :class:`~isaaclab_physx.assets.SurfaceGripper` and :class:`~isaaclab_physx.assets.SurfaceGripperCfg`:
    PhysX-specific implementation for surface gripper simulation using contact APIs. Moved from
    ``isaaclab.assets``.

* Added backward-compatible wrapper methods in :class:`~isaaclab_physx.assets.RigidObjectCollection`
  and :class:`~isaaclab_physx.assets.RigidObjectCollectionData` that delegate to the new
  ``body_*`` naming convention.

Deprecated
^^^^^^^^^^

* Deprecated the ``root_physx_view`` property on :class:`~isaaclab_physx.assets.Articulation`,
  :class:`~isaaclab_physx.assets.RigidObject`, :class:`~isaaclab_physx.assets.RigidObjectCollection`,
  and :class:`~isaaclab_physx.assets.DeformableObject` in favor of the ``root_view`` property.
  The ``root_physx_view`` property will be removed in a future release.

* Deprecated the ``object_*`` naming convention in :class:`~isaaclab_physx.assets.RigidObjectCollection`
  and :class:`~isaaclab_physx.assets.RigidObjectCollectionData` in favor of ``body_*``. The following
  methods and properties are deprecated and will be removed in a future release:

  **RigidObjectCollection methods:**

  * ``write_object_state_to_sim()`` → use ``write_body_state_to_sim()``
  * ``write_object_link_state_to_sim()`` → use ``write_body_link_state_to_sim()``
  * ``write_object_pose_to_sim()`` → use ``write_body_pose_to_sim()``
  * ``write_object_link_pose_to_sim()`` → use ``write_body_link_pose_to_sim()``
  * ``write_object_com_pose_to_sim()`` → use ``write_body_com_pose_to_sim()``
  * ``write_object_velocity_to_sim()`` → use ``write_body_com_velocity_to_sim()``
  * ``write_object_com_velocity_to_sim()`` → use ``write_body_com_velocity_to_sim()``
  * ``write_object_link_velocity_to_sim()`` → use ``write_body_link_velocity_to_sim()``
  * ``find_objects()`` → use ``find_bodies()``

  **RigidObjectCollectionData properties:**

  * ``default_object_state`` → use ``default_body_state``
  * ``object_names`` → use ``body_names``
  * ``object_link_pose_w`` → use ``body_link_pose_w``
  * ``object_link_vel_w`` → use ``body_link_vel_w``
  * ``object_com_pose_w`` → use ``body_com_pose_w``
  * ``object_com_vel_w`` → use ``body_com_vel_w``
  * ``object_state_w`` → use ``body_state_w``
  * ``object_link_state_w`` → use ``body_link_state_w``
  * ``object_com_state_w`` → use ``body_com_state_w``
  * ``object_com_acc_w`` → use ``body_com_acc_w``
  * ``object_pose_w`` → use ``body_pose_w``
  * ``object_pos_w`` → use ``body_pos_w``
  * ``object_quat_w`` → use ``body_quat_w``
  * ``object_vel_w`` → use ``body_vel_w``
  * ``object_lin_vel_w`` → use ``body_lin_vel_w``
  * ``object_ang_vel_w`` → use ``body_ang_vel_w``
  * ``object_acc_w`` → use ``body_acc_w``
  * And all other ``object_*`` properties (see :ref:`migrating-to-isaaclab-3-0` for complete list).

Migration
^^^^^^^^^

* See :ref:`migrating-to-isaaclab-3-0` for detailed migration instructions.
