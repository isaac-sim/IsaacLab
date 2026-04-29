Changelog
---------

0.2.13 (2026-04-29)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Aligned ``test/assets/test_rigid_object.py`` 1-to-1 with
  :mod:`isaaclab_physx.test.assets.test_rigid_object`: same set of 20 test
  functions, identical names, parametrizations, and assertions.  PhysX-style
  ``cube_object.root_view.set_X(...)`` / ``get_X(...)`` calls are adapted to
  OVPhysX by going through the public setters
  (:meth:`~isaaclab_ovphysx.assets.RigidObject.set_masses_index`,
  :meth:`~isaaclab_ovphysx.assets.RigidObject.set_coms_index`) and the
  data-class properties (``cube_object.data.body_mass``, ``body_com_pose_b``).
  The five material-property tests
  (``test_rigid_body_set_material_properties``,
  ``test_set_material_properties_via_view``, ``test_rigid_body_no_friction``,
  ``test_rigid_body_with_static_friction``, ``test_rigid_body_with_restitution``)
  remain xfailed pending the wheel-side ``RIGID_BODY_MATERIAL`` TensorType
  (see ``docs/superpowers/specs/2026-04-28-ovphysx-wheel-gaps-for-marco.md``).
  Dropped the OVPhysX-only extras that were artifacts of the earlier
  mock-based suite (``test_initialization_body_names``,
  ``test_initialization_data_not_none``,
  ``test_initialization_wrench_composers``,
  ``test_external_force_buffer_composition``,
  ``test_set_rigid_object_state_physics``, ``test_rigid_body_set_inertia``,
  ``test_gravity_vec_w_direction``, ``test_gravity_vec_w_body_acc``,
  ``test_body_root_state_properties_shapes``,
  ``test_body_root_state_properties_physics``,
  ``test_root_link_vel_w_buffer_differs_from_root_com_vel_w``,
  ``test_root_link_vel_w_lever_arm_physics``,
  ``test_ovphysx_manager_step_exists``, ``test_warmup_and_load_cpu``,
  ``test_stage_load_cpu``, ``test_warmup_and_load_gpu``).  Renamed
  ``test_warmup_gpu_not_called_for_cpu`` to ``test_warmup_attach_stage_not_called_for_cpu``
  to match the PhysX analogue and use a
  :class:`~unittest.mock.MagicMock` spy on
  :attr:`~isaaclab_ovphysx.physics.OvPhysxManager._physx` to assert the
  CPU-mode ``warmup_gpu()`` guard is in place.

0.2.12 (2026-04-29)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Unified the CPU and GPU paths in
  :meth:`~isaaclab_ovphysx.physics.OvPhysxManager._configure_physx_scene_prim`.
  ``PhysxSceneAPI`` schema and ``enableSceneQuerySupport`` are now applied
  on both CPU and GPU; the GPU-only attributes (``enableGPUDynamics``,
  ``broadphaseType="GPU"``, the ``gpu*`` capacity attrs from
  :class:`~isaaclab_ovphysx.physics.OvPhysxCfg`) remain gated on
  ``device == "gpu"``. Previously the CPU path silently skipped the
  schema apply, so user-set ``SimulationCfg.enable_scene_query_support``
  did not propagate.

0.2.11 (2026-04-27)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Polished ``test/assets/test_rigid_object.py`` following PR #5426 review
  comments: dropped wheel-gate ``pytest.importorskip`` and ``hasattr`` soft-skips
  (the ovphysx wheel reliably exposes these symbols; an ``ImportError`` at import
  time is the correct failure mode if missing); stripped the
  ``"Real-backend port of PhysX's test_X"`` preamble from all 16 test
  docstrings; dropped the ``sim_ctx_cpu`` fixture and inlined
  ``build_simulation_context(device=device, ...)`` per test, mirroring the
  PhysX/Newton pattern; added ``@pytest.mark.parametrize("device", ["cuda:0",
  "cpu"])`` to all 29 parameterisable tests, providing GPU coverage; tightened
  docstrings on ``test_initialization_with_articulation_root`` and
  ``test_initialization_with_no_rigid_body`` to make explicit these are
  rigid-object error-handling tests (not articulation tests), with actionable
  xfail reasons. The ``live_manager_cpu`` fixture and its three warmup/lifecycle
  tests remain CPU-only because they explicitly verify CPU-mode manager
  behaviour.

0.2.10 (2026-04-29)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Added Google-style docstrings to every kernel and helper function in
  :mod:`isaaclab_ovphysx.assets.kernels`. Each ``@wp.kernel`` and ``@wp.func``
  now has a summary line, an ``Args:`` block with shape, dtype, and SI unit
  annotations, and (where non-obvious) a ``Formula:`` or inline formula block
  explaining the mathematical convention. No behavior changes.

0.2.9 (2026-04-29)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Stripped ``Task <N>`` planning markers from section headers and inline
  comments in :class:`~isaaclab_ovphysx.assets.RigidObject`. These were
  development artifacts and carried no runtime meaning.
* Polished public-method docstrings on
  :class:`~isaaclab_ovphysx.assets.RigidObject` to match the structural
  style of the PhysX and Newton ``RigidObject`` references — Google-style
  ``Args:`` blocks, the ``"This method expects partial/full data."`` note,
  the ``"Sets the velocity of the root's center of mass rather than the
  root's frame."`` caveat on velocity writers, and consistent shape/dtype
  wording across the 12 root-state writers, the 6 mass/COM/inertia setters,
  and the 3 deprecated state writers.
* Renamed the private helper
  :meth:`~isaaclab_ovphysx.assets.RigidObject._write_root_state` to
  :meth:`~isaaclab_ovphysx.assets.RigidObject._write_body_state` to better
  reflect that a rigid object has no articulation root — it has a single
  body. The 14 in-file call sites are updated; the public writer names
  (``write_root_pose_to_sim_*``, ``set_masses_*``, etc.) are unchanged
  because they mirror the :class:`~isaaclab.assets.BaseRigidObject`
  contract.

0.2.8 (2026-04-29)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Reorganised :class:`~isaaclab_ovphysx.assets.RigidObjectData` to mirror
  the section layout of the PhysX and Newton ``RigidObjectData`` modules
  rather than the existing OVPhysX articulation coding style. Replaced
  ``# --- section ---`` comment dividers with ``"""section"""`` triple-quote
  blocks and reordered the file top-down to: defaults → root state →
  body state → body properties → derived → sliced → internal helpers →
  deprecated state-concat properties. Extracted the per-instance buffer
  and ProxyArray cache attribute initialisation out of ``__init__`` into
  a dedicated :meth:`_create_buffers` method, mirroring PhysX. Public API,
  property bodies, kernel launches, and the lazy ``_ensure_*`` allocation
  pattern are unchanged.

0.2.7 (2026-04-29)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Tightened the :attr:`~isaaclab_ovphysx.assets.RigidObject.root_view`
  docstring to explicitly document the OVPhysX dict-of-bindings architecture.
  Callers needing low-level binding access should use
  :meth:`~isaaclab_ovphysx.assets.RigidObject._get_binding`; for high-level
  state access use the :attr:`~isaaclab_ovphysx.assets.RigidObject.num_instances`,
  :attr:`~isaaclab_ovphysx.assets.RigidObject.body_names`, and
  :attr:`~isaaclab_ovphysx.assets.RigidObjectData.root_link_pose_w` accessors
  directly.
* Demoted :attr:`~isaaclab_ovphysx.assets.RigidObjectData.device`,
  :attr:`~isaaclab_ovphysx.assets.RigidObjectData.num_instances`, and
  :attr:`~isaaclab_ovphysx.assets.RigidObjectData.num_bodies` from
  ``@property`` accessors backed by ``_device``, ``_num_instances``, and
  ``_num_bodies`` to plain instance attributes, matching the PhysX and Newton
  backends. Downstream code that read ``RigidObjectData._device`` should now
  use ``RigidObjectData.device``; same for ``num_instances`` and ``num_bodies``.

Removed
^^^^^^^

* Removed :attr:`~isaaclab_ovphysx.assets.RigidObjectData.body_link_acc_w`.
  This OVPhysX-only convenience alias for
  :attr:`~isaaclab_ovphysx.assets.RigidObjectData.body_com_acc_w` was not
  present on the base contract or the PhysX/Newton backends. Use
  :attr:`~isaaclab_ovphysx.assets.RigidObjectData.body_com_acc_w` directly —
  for a single rigid body the link and COM accelerations are equivalent.

0.2.6 (2026-04-27)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Implemented seven deprecated state-concat properties on
  :class:`~isaaclab_ovphysx.assets.RigidObjectData` that were previously
  ``NotImplementedError`` stubs:
  :attr:`~isaaclab_ovphysx.assets.RigidObjectData.default_root_state`,
  :attr:`~isaaclab_ovphysx.assets.RigidObjectData.root_state_w`,
  :attr:`~isaaclab_ovphysx.assets.RigidObjectData.root_link_state_w`,
  :attr:`~isaaclab_ovphysx.assets.RigidObjectData.root_com_state_w`,
  :attr:`~isaaclab_ovphysx.assets.RigidObjectData.body_state_w`,
  :attr:`~isaaclab_ovphysx.assets.RigidObjectData.body_link_state_w`, and
  :attr:`~isaaclab_ovphysx.assets.RigidObjectData.body_com_state_w`.
  Each emits a ``DeprecationWarning`` recommending the split
  pose/velocity properties (e.g. ``root_link_pose_w`` + ``root_com_vel_w``)
  and lazily populates a ``vec13f`` buffer via the
  ``concat_root_pose_and_vel_to_state`` kernel, matching PhysX and Newton.
* Added ``vec13f`` dtype and ``concat_root_pose_and_vel_to_state`` kernel to
  :mod:`isaaclab_ovphysx.assets.kernels`, vendored from the shared PhysX
  kernel module.  Cache invalidation in
  :meth:`~isaaclab_ovphysx.assets.RigidObjectData._invalidate_caches` now
  covers the three new ``TimestampedBuffer`` objects.

0.2.5 (2026-04-27)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed :attr:`~isaaclab_ovphysx.assets.RigidObjectData.root_link_vel_w` to derive
  the link-frame velocity from the COM velocity via the lever-arm transform
  ``get_root_link_vel_from_root_com_vel``, matching the PhysX and Newton backends.
  ``RIGID_BODY_VELOCITY`` is assumed to return COM-frame velocity (standard PhysX
  convention); :attr:`~isaaclab_ovphysx.assets.RigidObjectData.root_com_vel_w`
  continues to read the binding directly.

Added
^^^^^

* Added ``get_root_link_vel_from_root_com_vel`` kernel to
  :mod:`isaaclab_ovphysx.assets.kernels`, vendored from the PhysX shared-kernel
  module.  The kernel recovers root link spatial velocity from COM spatial velocity
  using a lever-arm correction: ``link_lin = com_lin + omega x lever_arm``.

0.2.4 (2026-04-27)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed :meth:`~isaaclab_ovphysx.assets.RigidObject.reset` to match the
  PhysX and Newton backends: the method now only resets the wrench composers
  and no longer auto-writes the default pose and velocity to the simulation.
  Callers that want to restore initial state must explicitly call
  :meth:`~isaaclab_ovphysx.assets.RigidObject.write_root_pose_to_sim_index`
  and
  :meth:`~isaaclab_ovphysx.assets.RigidObject.write_root_velocity_to_sim_index`
  (or the mask variants) after calling ``reset``.

0.2.3 (2026-04-27)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :meth:`~isaaclab_ovphysx.assets.RigidObject._initialize_impl` where
  ``hasattr(root_pose, "body_names")`` only suppresses ``AttributeError`` but
  the real ovphysx ``TensorBinding.body_names`` raises ``TypeError`` for
  non-articulation tensor types (e.g. ``RIGID_BODY_POSE``), propagating the
  exception instead of falling back to ``["base_link"]``. Replaced the
  ``hasattr`` guard with a ``try/except (AttributeError, TypeError)`` block.
* Fixed :meth:`~isaaclab_ovphysx.assets.RigidObject._initialize_impl` where
  ``self._device`` was derived from ``self._ovphysx.device`` (a property that
  the real ovphysx ``PhysX`` object does not expose), causing a silent fallback
  to ``"cuda:0"`` even when the simulation runs on CPU. The device is now read
  from :meth:`~isaaclab_ovphysx.physics.OvPhysxManager.get_device`, which
  mirrors ``SimulationContext.cfg.device`` and is always correct.

0.2.2 (2026-04-27)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed a shape mismatch in :meth:`~isaaclab_ovphysx.assets.RigidObject._write_root_state`
  where a full write with more rows than ``num_instances`` produced a ``ValueError`` inside
  the binding instead of the expected ``RuntimeError``. Added an explicit row-count guard
  on the full-write path so callers receive a clear ``RuntimeError`` on bad shapes.
* Fixed :meth:`~isaaclab_ovphysx.assets.RigidObject._write_root_state` for 1-D bindings
  (e.g. ``RIGID_BODY_MASS``) on the index/mask sub-write paths: the source array is now
  normalised to 1-D so that boolean-mask scatter in :class:`MockTensorBinding` and the
  real OVPhysX binding receive a flat buffer rather than a ``(K, 1)`` 2-D array.
* Fixed :meth:`~isaaclab_ovphysx.assets.RigidObject.write_root_com_pose_to_sim_index` and
  :meth:`~isaaclab_ovphysx.assets.RigidObject.write_root_com_pose_to_sim_mask` to raise
  ``RuntimeError`` on full-write calls when the input has more rows than ``num_instances``
  (previously the extra rows were silently truncated by ``_com_pose_to_link_pose``).
* Implemented :attr:`~isaaclab_ovphysx.assets.RigidObjectData.default_root_pose` and
  :attr:`~isaaclab_ovphysx.assets.RigidObjectData.default_root_vel` properties that were
  left as ``NotImplementedError`` stubs; they now return the :class:`~isaaclab.utils.ProxyArray`
  wrappers populated from ``RigidObjectCfg.init_state``.

0.2.1 (2026-04-27)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed a stale-buffer bug in :meth:`~isaaclab_ovphysx.assets.RigidObject._com_pose_to_link_pose`
  where the ``RIGID_BODY_COM_POSE`` binding was read once by :class:`~isaaclab.utils.wrench_composer.WrenchComposer`
  during construction (via a ``hasattr`` property probe) and then cached with
  timestamp equal to the initial ``_sim_time``. Subsequent writes through
  :meth:`~isaaclab_ovphysx.assets.RigidObject.write_root_com_pose_to_sim_index` used the stale
  body-frame COM offset, producing an incorrect frame conversion. The buffer is
  now unconditionally refreshed at write time.


0.2.0 (2026-04-27)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_ovphysx.assets.RigidObject` and
  :class:`~isaaclab_ovphysx.assets.RigidObjectData` for single-actor rigid
  body simulation against the OVPhysX backend, satisfying the
  :class:`~isaaclab.assets.BaseRigidObject` and
  :class:`~isaaclab.assets.BaseRigidObjectData` contracts.
* Added ``RIGID_BODY_*`` :class:`TensorType` aliases (``RIGID_BODY_POSE``,
  ``RIGID_BODY_VELOCITY``, ``RIGID_BODY_ACCELERATION``,
  ``RIGID_BODY_WRENCH``, ``RIGID_BODY_MASS``, ``RIGID_BODY_INV_MASS``,
  ``RIGID_BODY_COM_POSE``, ``RIGID_BODY_INERTIA``, ``RIGID_BODY_INV_INERTIA``)
  in :mod:`isaaclab_ovphysx.tensor_types`. Three of these
  (``RIGID_BODY_ACCELERATION``, ``RIGID_BODY_INV_MASS``,
  ``RIGID_BODY_INV_INERTIA``) require an ``ovphysx`` wheel update
  exposing the matching :class:`TensorType` enum values; the remaining
  six already ship with the current wheel.
* Added ``asset_kind="rigid_object"`` mode to
  ``isaaclab_ovphysx.test.mock_interfaces.views.MockOvPhysxBindingSet``
  for kitless mock-based testing of the new asset.

Changed
^^^^^^^

* Moved shared frame-conversion and wrench-composition Warp kernels from
  ``isaaclab_ovphysx.assets.articulation.kernels`` to a new
  ``isaaclab_ovphysx.assets.kernels`` module. Articulation imports were
  updated to point at the new location; downstream code referencing the
  articulation-private kernels module needs the same import update.
  Newly-added kernel ``_compose_root_link_pose_from_com`` for the
  COM-pose write path also lives in the shared module.


0.1.2 (2026-04-23)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Wrapped :attr:`~isaaclab_ovphysx.assets.ArticulationData.GRAVITY_VEC_W` and
  :attr:`~isaaclab_ovphysx.assets.ArticulationData.FORWARD_VEC_B` in
  :class:`~isaaclab.utils.warp.ProxyArray` to match the PhysX and Newton
  backends. Public observations such as
  :func:`~isaaclab.envs.mdp.observations.projected_gravity` access
  ``asset.data.GRAVITY_VEC_W.torch``; the previous raw ``wp.array`` lacked
  ``.torch`` and raised ``AttributeError`` on the ovphysx backend.


0.1.1 (2026-04-21)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Replaced private ``_find_names`` (fnmatch + regex) with the standard
  :func:`~isaaclab.utils.string.resolve_matching_names` for all finder
  methods, unifying name-resolution behavior across backends. Fnmatch-style
  glob patterns (e.g. ``joint_*``) are no longer supported; use regex
  equivalents (e.g. ``joint_.*``). ``find_fixed_tendons`` and
  ``find_spatial_tendons`` now raise ``ValueError`` on empty tendon lists,
  matching the PhysX backend.
* Changed ``find_joints`` ``joint_subset`` parameter from ``list[int]``
  (indices) to ``list[str]`` (names) to match the ``BaseArticulation``
  interface. Callers passing indices should convert to names first.


0.1.0 (2026-04-20)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Initial release of the ``isaaclab_ovphysx`` extension.
