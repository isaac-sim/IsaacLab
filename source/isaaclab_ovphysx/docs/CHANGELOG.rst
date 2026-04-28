Changelog
---------

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
