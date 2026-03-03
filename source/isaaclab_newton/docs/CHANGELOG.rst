Changelog
---------

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
