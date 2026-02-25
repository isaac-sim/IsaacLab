Changelog
---------

0.5.1 (2026-02-25)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated ContactSensor ``body_names`` property to use ``num_sensors`` instead of
  deprecated ``num_bodies``.


0.5.0 (2026-02-24)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Aligned asset API with the multi-backend architecture. Base class abstract methods
  in :class:`~isaaclab.assets.BaseArticulation` and :class:`~isaaclab.assets.BaseRigidObject`
  have been refined so that PhysX and Newton backends share a consistent interface.

* Improved docstrings across all asset classes with precise shape and dtype annotations
  for warp array properties and write methods.

* Migrated tests to use the new ``_index`` / ``_mask`` write method APIs, removing
  usage of deprecated write methods.


0.4.1 (2026-02-18)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed a bug in :meth:~isaaclab_physx.assets.Articulation.process_actuators_cfg where explicit actuator joints could receive non-zero PhysX stiffness/damping, causing double PD control.


0.4.0 (2026-02-13)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Migrated all PhysX asset classes to warp backend:
  :class:`~isaaclab_physx.assets.Articulation`,
  :class:`~isaaclab_physx.assets.RigidObject`,
  :class:`~isaaclab_physx.assets.RigidObjectCollection`,
  :class:`~isaaclab_physx.assets.DeformableObject`, and
  :class:`~isaaclab_physx.assets.SurfaceGripper`.
  Internal state buffers now use ``wp.array`` with structured warp types
  (``wp.vec3f``, ``wp.quatf``, ``wp.transformf``, ``wp.spatial_vectorf``).

* Migrated all PhysX sensor classes to warp backend:
  :class:`~isaaclab_physx.sensors.ContactSensor`,
  :class:`~isaaclab_physx.sensors.Imu`, and
  :class:`~isaaclab_physx.sensors.FrameTransformer`.

* Split all write methods into ``_index`` and ``_mask`` variants for explicit
  sparse-index vs. boolean-mask semantics.

Added
^^^^^

* Added warp kernel modules for fused GPU computations:

  * :mod:`isaaclab_physx.assets.kernels` — shared kernels for root state extraction,
    velocity transforms, and data write-back.
  * :mod:`isaaclab_physx.assets.articulation.kernels` — articulation-specific kernels
    for joint state, body properties, and COM computations.
  * :mod:`isaaclab_physx.assets.deformable_object.kernels` — nodal state and mean
    vertex computations.
  * :mod:`isaaclab_physx.assets.rigid_object_collection.kernels` — 2D indexed kernels
    for multi-body collections.
  * :mod:`isaaclab_physx.sensors.contact_sensor.kernels` — contact force aggregation
    and history buffer management.
  * :mod:`isaaclab_physx.sensors.imu.kernels` — fused IMU update combining acceleration,
    gyroscope, and gravity projection.
  * :mod:`isaaclab_physx.sensors.frame_transformer.kernels` — frame transform computations.

* Added warp-based mock PhysX views for unit testing:
  ``MockArticulationViewWarp``, ``MockRigidBodyViewWarp``, ``MockRigidContactViewWarp``.


0.3.0 (2026-02-11)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Refactored :class:`~isaaclab_physx.physics.PhysxManager` to properly handle physics initialization
  order. ``attach_stage()`` is now called before ``start_simulation()`` to ensure GPU buffers are
  correctly allocated.
* Removed ``device`` field from :class:`~isaaclab_physx.physics.PhysxManagerCfg`. Device is now
  inherited from :attr:`SimulationCfg.device`.

Added
^^^^^

* Added :class:`~isaaclab_physx.physics.PhysxManager` as the concrete PhysX backend implementation
  of :class:`~isaaclab.physics.PhysicsManager`.
* Added :class:`~isaaclab_physx.physics.IsaacEvents` enum for PhysX-specific simulation events.
* Added monkey-patching of ``isaacsim.core.simulation_manager.SimulationManager`` in package init
  to ensure Isaac Sim uses :class:`~isaaclab_physx.physics.PhysxManager` for callback handling.


0.2.0 (2026-02-05)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated all PhysX benchmarks in :mod:`isaaclab_physx.benchmark` to use the new
  :class:`~isaaclab.test.benchmark.BaseIsaacLabBenchmark` framework from ``isaaclab.test.benchmark``.

* Added support for configurable output backends via ``--benchmark_backend`` argument.
  Supported backends: ``json``, ``osmo``, ``omniperf``.


0.1.4 (2026-02-05)
~~~~~~~~~~~~~~~~~~

Removed
^^^^^^^

* Removed all the deprecated properties and shorthands in the assets. They now live in the base classes.


0.1.3 (2026-02-03)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`isaaclab_physx.benchmark` module containing performance micro-benchmarks for
  PhysX asset classes. Includes:

  * ``benchmark_articulation.py``: Benchmarks for setter/writer methods on
    :class:`~isaaclab_physx.assets.Articulation` including root state, joint state,
    joint parameters, and body property operations.
  * ``benchmark_articulation_data.py``: Benchmarks for property accessors on
    :class:`~isaaclab_physx.assets.ArticulationData` covering root link/COM properties,
    joint properties, and body link/COM properties.
  * ``benchmark_rigid_object.py``: Benchmarks for setter/writer methods on
    :class:`~isaaclab_physx.assets.RigidObject` including root state and body property operations.
  * ``benchmark_rigid_object_data.py``: Benchmarks for property accessors on
    :class:`~isaaclab_physx.assets.RigidObjectData`.
  * ``benchmark_rigid_object_collection.py``: Benchmarks for setter/writer methods on
    :class:`~isaaclab_physx.assets.RigidObjectCollection` including body state, pose,
    velocity, and property operations.
  * ``benchmark_rigid_object_collection_data.py``: Benchmarks for property accessors on
    :class:`~isaaclab_physx.assets.RigidObjectCollectionData`.

  All benchmarks support configurable iterations, warmup steps, instance counts, multiple
  input modes (torch list, torch tensor), and output to JSON/CSV formats with hardware
  information capture.


0.1.2 (2026-02-03)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`isaaclab_physx.test.mock_interfaces` module providing mock PhysX view implementations
  for unit testing without requiring Isaac Sim. Includes:

  * :class:`MockRigidBodyView`: Mock for ``physx.RigidBodyView`` with transforms, velocities,
    accelerations, and mass properties.
  * :class:`MockArticulationView`: Mock for ``physx.ArticulationView`` with root/link states,
    DOF properties, and joint control.
  * :class:`MockRigidContactView`: Mock for ``physx.RigidContactView`` with contact forces,
    positions, normals, and friction data.
  * Factory functions including pre-configured quadruped and humanoid views.
  * Patching utilities and decorators for easy test injection.


0.1.0 (2026-01-28)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`isaaclab_physx.sensors` module containing PhysX-specific sensor implementations:

  * :class:`~isaaclab_physx.sensors.ContactSensor` and :class:`~isaaclab_physx.sensors.ContactSensorData`:
    PhysX-specific implementation for contact force sensing. Extends
    :class:`~isaaclab.sensors.contact_sensor.BaseContactSensor` with PhysX tensor API for contact
    force queries, contact filtering, and contact point tracking.

  * :class:`~isaaclab_physx.sensors.Imu` and :class:`~isaaclab_physx.sensors.ImuData`:
    PhysX-specific implementation for inertial measurement unit simulation. Extends
    :class:`~isaaclab.sensors.imu.BaseImu` with PhysX rigid body views for velocity and
    acceleration computation.

  * :class:`~isaaclab_physx.sensors.FrameTransformer` and :class:`~isaaclab_physx.sensors.FrameTransformerData`:
    PhysX-specific implementation for coordinate frame transformations. Extends
    :class:`~isaaclab.sensors.frame_transformer.BaseFrameTransformer` with PhysX rigid body views
    for efficient frame pose queries.

* Added PhysX-specific sensor tests moved from ``isaaclab/test/sensors/``:

  * ``test_contact_sensor.py``
  * ``test_imu.py``
  * ``test_frame_transformer.py``
  * ``check_contact_sensor.py``
  * ``check_imu_sensor.py``

Deprecated
^^^^^^^^^^

* Deprecated the ``pose_w``, ``pos_w``, and ``quat_w`` properties on
  :class:`~isaaclab_physx.sensors.ContactSensorData` and :class:`~isaaclab_physx.sensors.ImuData`.
  These properties will be removed in a future release. Please use a dedicated sensor (e.g.,
  :class:`~isaaclab.sensors.FrameTransformer`) to measure sensor poses in world frame.


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
