Changelog
---------

0.5.5 (2026-03-10)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed dtype mismatch in :class:`~isaaclab_newton.assets.RigidObjectCollection`
  where ``write_body_com_pose_to_sim_index`` and ``write_body_link_velocity_to_sim_index``
  passed ``body_com_pose_b`` (``wp.transformf``) instead of ``body_com_pos_b``
  (``wp.vec3f``) to the underlying warp kernels.


0.5.4 (2026-02-28)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added contact sensor support via :class:`newton.sensors.SensorContact` with
  Isaac Lab pattern conversion (``.*`` to fnmatch, USD path normalization)
  inlined in :meth:`~isaaclab_newton.physics.NewtonManager.add_contact_sensor`.

Changed
^^^^^^^

* Changed :class:`~isaaclab_newton.sensors.contact_sensor.ContactSensor` to
  flatten Newton's per-world nested ``sensing_objs`` and ``counterparts``
  attributes.

Fixed
^^^^^

* Fixed ``RigidObjectData.body_inertia`` shape from ``(N, B, 3, 3)`` to ``(N, B, 9)``.


0.5.3 (2026-03-09)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :attr:`~isaaclab_newton.assets.RigidObjectData.body_inertia` to return a
  ``(num_instances, num_bodies, 9)`` float32 strided view, matching the articulation fix in 0.5.2.

* Fixed non-contiguous array handling in ``RigidObjectData`` position, quaternion, and
  spatial-vector extraction helpers. The ``source`` buffer shape and kernel dispatch ``dim``
  now use the input array's shape instead of the (possibly uninitialized) output shape.


0.5.2 (2026-03-06)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :attr:`~isaaclab_newton.assets.ArticulationData.body_inertia` in
  :class:`~isaaclab_newton.assets.ArticulationData` to return a ``(num_instances, num_bodies, 9)``
  float32 array as documented, instead of a ``(num_instances, num_bodies, 3, 3)`` array. The
  ``(N, B, 3, 3)`` shape caused a broadcasting error in
  :class:`~isaaclab.envs.mdp.events.randomize_rigid_body_mass` and a dimension mismatch when the
  ``write_body_inertia_to_buffer_*`` kernels were called via
  :meth:`~isaaclab_newton.assets.Articulation.set_inertias_index` and
  :meth:`~isaaclab_newton.assets.Articulation.set_inertias_mask`. The fix creates a ``(N, B, 9)``
  view over the same memory using explicit strides, collapsing the two contiguous trailing
  dimensions without copying data.

* Fixed ``AttributeError: 'NoneType' object has no attribute 'device'`` in
  :meth:`~isaaclab_newton.physics.NewtonManager.step` when ``use_cuda_graph=True`` but the CUDA
  graph was not captured (e.g., when RTX/Fabric USD sync is active). The step condition now
  checks ``cls._graph is not None`` directly instead of repeating the capture-time heuristic.


0.5.1 (2026-03-06)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_newton.assets.RigidObjectCollection` and
  :class:`~isaaclab_newton.assets.RigidObjectCollectionData` for managing
  collections of independent rigid bodies. Uses a single
  ``ArticulationView`` with a combined fnmatch pattern to get direct
  ``(num_envs, num_bodies)`` bindings into Newton's state, avoiding the
  scatter/gather overhead needed by PhysX.

* Added :class:`~isaaclab_newton.test.mock_interfaces.views.MockNewtonCollectionView`
  for unit testing the collection data class without simulation.

* Added Newton backend to the rigid object collection interface conformance
  tests (``test_rigid_object_collection_iface.py``).


0.5.0 (2026-03-06)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added full Newton articulation test suite (``test_articulation.py``) — 194 passed,
  8 skipped, 4 xfailed — adapted from PhysX tests with Newton-specific imports, sim
  config, and solver tolerance adjustments.

* Added full Newton rigid body test suite (``test_rigid_object.py``) — 74 passed,
  25 skipped — adapted from PhysX tests with Newton-specific mass/COM APIs and
  ``_newton_sim_context()`` helper for device/gravity/dt configuration.

Fixed
^^^^^

* Fixed ``ArticulationData`` and ``RigidObjectData`` to rebind simulation pointers
  on full sim reset via ``PHYSICS_READY`` callback, preventing stale warp array
  references after ``sim.reset()`` recreates the Newton model.

* Fixed ``ArticulationData`` to force ``eval_fk`` after joint state writes so that
  link poses are consistent with joint positions before the next ``sim.step()``.

* Fixed lazy initialization of ``TimestampedBuffer`` properties in
  ``RigidObjectData`` (velocity-in-body-frame and deprecated state properties)
  that were left as ``None`` and caused ``AttributeError`` on first access.

* Fixed ``None`` guards for timestamp invalidation in ``RigidObject`` write methods
  (``write_root_pose_to_sim``, ``write_root_velocity_to_sim``) to avoid
  ``AttributeError`` when optional buffers have not been initialized.

* Fixed ``is_contiguous`` usage in ``RigidObjectData`` — warp 1.12.0rc2 exposes it
  as a property, not a method.

* Fixed ``body_com_pose_b`` → ``body_com_pos_b`` kernel input naming in
  ``RigidObjectData`` for ``root_com_pose_w`` and ``root_link_vel_w`` properties.

* Fixed ``wp.from_torch()`` called on warp arrays in ``RigidObjectData`` body
  inertia binding — replaced with direct ``.view()``/``.reshape()`` on warp arrays.

* Improved CPU support in ``NewtonManager``: added device guards for CUDA graph
  operations that are not available on CPU.

* Fixed explicit mask resolution in asset write methods to correctly handle both
  index-based and mask-based sparse writes.


0.4.1 (2026-03-03)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fix asset writer methods in :class:`~isaaclab_newton.assets.Articulation` and
  :class:`~isaaclab_newton.assets.RigidObject` to use public data properties
  instead of internal timestamped buffer ``.data`` fields, removing redundant
  manual timestamp updates.


0.4.0 (2026-03-01)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_newton.physics.NewtonManager` physics backend with
  MuJoCo-Warp, XPBD, and Featherstone solvers, CUDA-graph support, and
  backend-agnostic callback dispatch via :class:`~isaaclab.physics.PhysicsEvent`.

Changed
^^^^^^^

* Implemented ``newton_replicate`` to build per-environment worlds from USD
  prototypes using Newton's ``ModelBuilder``.

* Renamed ``NewtonContactSensorCfg`` to ``ContactSensorCfg`` and made it
  backend-agnostic with lazy ``class_type`` resolution.

* Pinned ``mujoco-warp==3.5.0`` and ``warp-lang==1.12.0rc2`` in ``setup.py``.


0.3.0 (2026-02-25)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`isaaclab_newton.test.mock_interfaces` test infrastructure module with
  structured mock views, factory functions, and unit tests — mirroring the
  ``isaaclab_physx`` mock interface pattern:

  * :class:`~isaaclab_newton.test.mock_interfaces.views.MockNewtonArticulationView`:
    extracted from monolithic ``mock_newton.py`` into its own module with lazy
    initialization, individual ``set_mock_*`` methods, ``_noop_setters`` flag,
    and numpy-based ``set_random_mock_data()``.

  * Factory functions: ``create_mock_articulation_view()``,
    ``create_mock_quadruped_view()``, ``create_mock_humanoid_view()`` for
    convenient test setup.

* Added unit tests for mock interfaces:
  ``test_mock_articulation_view.py`` and ``test_factories.py``.

Changed
^^^^^^^

* Restructured ``mock_newton.py``: moved ``MockNewtonArticulationView`` to
  ``views/mock_articulation_view.py`` and removed ``torch`` dependency from
  the mock module (replaced with ``numpy`` for random data generation).


0.2.3 (2026-02-27)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added runtime shape and dtype validation to all write methods in
  :class:`~isaaclab_newton.assets.Articulation` and
  :class:`~isaaclab_newton.assets.RigidObject` using
  :meth:`~isaaclab.assets.AssetBase.assert_shape_and_dtype` and
  :meth:`~isaaclab.assets.AssetBase.assert_shape_and_dtype_mask`.


0.2.2 (2026-02-26)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added runtime shape and dtype validation to all write methods in
  :class:`~isaaclab_newton.assets.Articulation` and
  :class:`~isaaclab_newton.assets.RigidObject` using
  :meth:`~isaaclab.assets.AssetBase.assert_shape_and_dtype` and
  :meth:`~isaaclab.assets.AssetBase.assert_shape_and_dtype_mask`.


0.2.1 (2026-02-25)

Removed
^^^^^^^

imgui-bundle dependency.

0.2.0 (2026-02-24)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`isaaclab_newton.assets` module containing Newton-specific asset implementations:

  * :class:`~isaaclab_newton.assets.Articulation` and :class:`~isaaclab_newton.assets.ArticulationData`:
    Newton-specific implementation for articulated rigid body systems (e.g., robots). Extends
    :class:`~isaaclab.assets.BaseArticulation` with Newton's ``ArticulationView`` API for
    GPU-accelerated simulation of multi-joint systems.

  * :class:`~isaaclab_newton.assets.RigidObject` and :class:`~isaaclab_newton.assets.RigidObjectData`:
    Newton-specific implementation for single rigid body assets. Extends
    :class:`~isaaclab.assets.BaseRigidObject` with Newton's simulation API for efficient
    rigid body state queries and writes.

* Added warp kernel modules for fused GPU computations:

  * :mod:`isaaclab_newton.assets.kernels` — shared kernels for root state extraction,
    velocity transforms, COM/link frame conversions, and data write-back.
  * :mod:`isaaclab_newton.assets.articulation.kernels` — articulation-specific kernels
    for joint state, soft limits, actuator state updates, and friction properties.

* All ``.data.*`` properties use ``wp.array`` with structured warp types
  (``wp.vec3f``, ``wp.quatf``, ``wp.transformf``, ``wp.spatial_vectorf``),
  matching the same convention used by ``isaaclab_physx``.

* All write methods follow the ``_index`` / ``_mask`` split for explicit
  sparse-index vs. boolean-mask semantics.


0.1.0 (2026-02-16)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added empty package
