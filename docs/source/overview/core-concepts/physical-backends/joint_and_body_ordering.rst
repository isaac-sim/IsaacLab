Joint and Body Ordering
=======================

PhysX and MJWarp may order an articulation's joints and bodies differently. Set
``joint_ordering`` and ``body_ordering`` to keep names mapped to the same tensor elements
across backends. This page explains the supported conventions, conversion costs, and direct
backend-view access. For backend selection and capabilities, see the :doc:`Physics Backends
overview <index>`.

.. seealso::

    For policy transfer instructions and validated examples, see
    :doc:`/source/how-to/transfer_policies_between_physx_and_newton`.


Why Articulation Orders Differ
------------------------------

Three facts explain why a checkpoint can need an ordering convention even when
both backends load the same USD asset:

1. USD names identify physical joints and bodies, but they do not impose one
   universal tensor-axis order across solvers.
2. PhysX and MJWarp construct native articulation views with different topology
   traversal and internal representation choices.
3. Isaac Lab resolves the requested names once during articulation
   initialization, then exposes the selected public order through its
   high-level API.

The backend selection described in :doc:`Multi-Backend Architecture
<../multi_backend_architecture>` controls which native view is created. The
ordering selection controls how the high-level API presents that view.


Public and Backend Order
------------------------

Set ``joint_ordering`` and ``body_ordering`` on
:class:`~isaaclab.assets.ArticulationCfg`. Each field accepts one of:

* ``None`` -- backend-native order and the zero-conversion default (see below).
* ``"physx"`` -- PhysX or OVPhysX articulation-view order.
* ``"mjwarp"`` -- Newton or MJWarp articulation-view order.
* ``"robot_schema"`` -- the order authored on the asset's
  ``isaac:physics:robotJoints`` (joints) or ``isaac:physics:robotLinks``
  (bodies) relationships.
* an explicit, complete name permutation -- a ``list`` or ``tuple`` naming every
  joint or body exactly once.

See :attr:`~isaaclab.assets.ArticulationCfg.joint_ordering` for the
authoritative list of accepted values.

For Python configs, prefer
:func:`~isaaclab.assets.apply_articulation_ordering_preset` to set both fields
to the same convention in a single call, which keeps joint and body order
consistent:

.. code-block:: python

    from isaaclab.assets import apply_articulation_ordering_preset

    robot_cfg = apply_articulation_ordering_preset(robot_cfg, "mjwarp")

.. warning::
    When overriding from the CLI or Hydra, set **both** ``joint_ordering`` and
    ``body_ordering``. Setting only ``joint_ordering`` silently leaves bodies in
    backend order, which mismatches a checkpoint whose body vectors follow the
    source convention.

Once initialized, the articulation and its
:class:`~isaaclab.assets.ArticulationNameMap` objects establish this contract:

.. list-table::
    :header-rows: 1
    :widths: 42 58

    * - Surface
      - Ordering contract
    * - ``joint_names`` and ``body_names``
      - Public order
    * - :class:`~isaaclab.assets.ArticulationData` joint and body properties
      - Public order
    * - Articulation command and property writers
      - Public input order
    * - ``backend_joint_names`` and ``backend_body_names``
      - Backend order
    * - ``root_view`` metadata and arrays
      - Backend order
    * - ``joint_ordering`` and ``body_ordering`` maps
      - Bridge between public and backend order

``None`` is the zero-conversion default. Public names follow the active
backend, no ordering map is installed, no reorder staging is allocated, and no
reorder kernel is launched. An explicit convention or name sequence that
resolves to backend order is normalized to ``None`` at initialization after a
one-time name resolution, so it reaches the exact same zero-conversion state:
a non-``None`` ordering map always denotes an actual permutation.

.. tip::
    After configuring an ordering, confirm the resolved public axis by comparing
    :attr:`~isaaclab.assets.Articulation.joint_names` with
    :attr:`~isaaclab.assets.Articulation.backend_joint_names` (and ``body_names``
    with ``backend_body_names``). Cross-backend conventions are resolved by
    emulation, so spot-check the result against the order your checkpoint expects.

High-Level MDP Terms
^^^^^^^^^^^^^^^^^^^^

Standard MDP terms that consume high-level articulation data use public
indices. This includes terms that resolve joint or body selections by name and
then index public-order :class:`~isaaclab.assets.ArticulationData` properties
or call high-level articulation writers.

Material randomization crosses the backend boundary explicitly: it converts
selected public body IDs to backend body IDs before deriving the corresponding
backend shape ranges. Custom or backend-specific MDP code that accesses
``root_view`` bypasses these high-level conversions and must convert its own
indices and tensors.


Conversion Cost
---------------

Convention resolution and map construction are one-time initialization
work. For a nonidentity map, affected reads and writes can require persistent
staging memory plus gather/scatter kernel launches. Identity maps avoid those
ongoing conversion paths.

On the Newton backend, a nonidentity ordering additionally records a fixed
per-step reorder of the core state buffers -- joint positions and velocities,
body poses and velocities -- inside the stepped and CUDA-graph-captured region.
This publishes backend-order state into the public-order buffers every step, so
a small baseline per-step cost exists independent of how often properties are
accessed.

The runtime and memory cost scales with environment count, joint or body count,
and how often affected properties and writers are accessed. Measure the
specific task and access pattern; there is no hardware-independent
steps-per-second number or fixed percentage overhead.


Direct Backend-View Access
--------------------------

.. warning::
    Arrays returned by the raw solver view (``root_view``) are always in
    backend solver order, regardless of the configured ``joint_ordering`` or
    ``body_ordering``. Indices from ``joint_names``, ``body_names``,
    ``find_joints``, or ``find_bodies`` are in public order and must not be
    used to index ``root_view`` arrays directly. Use the asset's ``data``
    buffers and write APIs, which already operate in public order, or
    translate indices through the asset's ``joint_ordering``/``body_ordering``
    maps first.

Prefer the high-level articulation API when possible; its data and writer
contracts already use public order. Direct ``root_view`` access uses backend
order even when ``joint_names`` or ``body_names`` uses another convention.
When a small set of indices needs to cross into a view array, translate them
with :meth:`~isaaclab.assets.Articulation.map_joint_ids_to_backend` or
:meth:`~isaaclab.assets.Articulation.map_body_ids_to_backend` instead of
indexing the ordering maps by hand; both return the input unchanged under
identity ordering.

Torch Conversion
^^^^^^^^^^^^^^^^

To gather a backend-order joint tensor into public order, enumerate public
output columns and use ``user_to_backend_indices`` to select the matching
backend source columns:

.. code-block:: python

    ordering = robot.joint_ordering
    if ordering is None:
        joint_pos_public = joint_pos_backend
    else:
        joint_pos_public = joint_pos_backend[:, list(ordering.user_to_backend_indices)]

For the opposite direction, enumerate backend output columns and use
``backend_to_user_indices`` to select the matching public source columns:

.. code-block:: python

    ordering = robot.joint_ordering
    if ordering is None:
        joint_target_backend = joint_target_public
    else:
        joint_target_backend = joint_target_public[:, list(ordering.backend_to_user_indices)]

Use ``robot.body_ordering`` in the same way for body-indexed axes. Keep the
``None`` guard because it avoids an unnecessary gather and means no map object
exists.

Warp Conversion
^^^^^^^^^^^^^^^

The elementwise reorder kernels in
``isaaclab.assets.articulation.ordering_kernels`` translate raw-view arrays
between backend and public order. For example,
``reorder_2d_backend_to_user`` gathers one ``(environment, joint)`` array into
public order:

.. code-block:: python

    import warp as wp

    from isaaclab.assets.articulation.ordering_kernels import reorder_2d_backend_to_user


    ordering = robot.joint_ordering
    if ordering is None:
        joint_pos_public = joint_pos_backend
    else:
        joint_pos_public = wp.empty(
            (robot.num_instances, robot.num_joints),
            dtype=wp.float32,
            device=robot.device,
        )
        wp.launch(
            reorder_2d_backend_to_user,
            dim=(robot.num_instances, robot.num_joints),
            inputs=[joint_pos_backend, ordering.user_to_backend],
            outputs=[joint_pos_public],
            device=robot.device,
        )

The caller owns output allocation, launch dimensions, data type, and every
non-articulation axis. A public-to-backend gather uses
``ordering.backend_to_user``. Treat both device maps as read-only.

The ``reorder_2d`` and ``reorder_3d`` kernels, in both the ``*_backend_to_user``
and ``*_user_to_backend`` directions, form this public elementwise family. All
other kernels in ``isaaclab.assets.articulation.ordering_kernels`` are internal
and may change without deprecation.

Joint maps cover named joints, not floating-base generalized coordinates.
When converting raw Jacobians or mass matrices, preserve the leading
``robot.num_base_dofs`` coordinates and offset mapped joint indices by that
count. Apply the joint permutation to both generalized-coordinate axes of a
mass matrix and leave all other axes unchanged.

The public floating-base Jacobian body rows use the full public body order; to
convert a raw backend Jacobian, gather with the full body map. Fixed-base raw
backend Jacobians omit the fixed root, so do not apply the full body map
directly. Omit public/root body index 0 and convert each remaining mapped
backend body ID to a Jacobian row by subtracting 1. The fixed-root-first
invariant makes this well-defined. See
:attr:`~isaaclab.assets.BaseArticulationData.body_link_jacobian_w` for the
authoritative body-axis convention. High-level articulation data performs
these conversions automatically.


What Ordering Does Not Solve
----------------------------

Ordering compatibility keeps names attached to the same vector elements;
it does not make simulated trajectories match. Policy behavior can still
diverge because of:

* contact generation and resolution
* friction
* restitution
* :ref:`actuator models and configuration <overview-actuators>`
* integration method
* timestep and substeps
* solver convergence

Use :doc:`Solver Comparison <solver-comparison>` to diagnose and tune these
differences rather than treating them as ordering failures.


Verification and Troubleshooting
--------------------------------

When a transferred policy behaves unexpectedly, check these items in
order:

1. Compare public ``joint_names`` and ``body_names`` with
   ``backend_joint_names`` and ``backend_body_names``.
2. Confirm both joint and body source conventions when the policy or task uses
   both kinds of vector.
3. Verify observation and action dimensions against the training run.
4. Audit custom code for direct ``root_view`` access.
5. Compare source and target values by physical name rather than by raw column.
6. When name-to-vector semantics are stable but motion still diverges, classify
   the problem as a solver-dynamics issue and continue with
   :doc:`Solver Comparison <solver-comparison>`.
