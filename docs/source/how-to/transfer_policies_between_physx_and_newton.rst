:orphan:

Transfer Policies Between PhysX and Newton
===========================================

.. seealso::

   This guide is the source of truth for the
   ``isaaclab-transferring-policies-sim-to-sim`` agent skill
   (`skill source
   <../../../skills/user/isaaclab-transferring-policies-sim-to-sim/SKILL.md>`__).
   When you change this page, update the skill so agent guidance stays in sync. See
   :doc:`/source/developer-tools/agent_skills`.

   First make every robot and object MJWarp-clean by following
   :doc:`/source/how-to/prepare_asset_for_newton`
   and the ``isaaclab-preparing-assets-for-newton``
   `skill
   <https://github.com/isaac-sim/IsaacLab/blob/develop/skills/user/prepare-assets-for-newton/SKILL.md>`__.

Sim-to-sim transfer evaluates one policy checkpoint in a physics backend different from the one
used for training. This guide covers both PhysX-trained policies deployed in Newton and
Newton-trained policies deployed in PhysX.

The checkpoint maps ordered observations to actions; it does not include the physics engine.
Transfer works when both backends expose the same policy inputs and outputs. Expect similar
behavior, not identical trajectories. Successful transfer can be a first step toward sim-to-real
deployment.


Task readiness and checkpoint compatibility
-------------------------------------------

The same registered task should describe the same Markov decision process (MDP) in both physics
engines. Selecting ``physics=isaacsim_physx`` or ``physics=newton_mjwarp`` resolves a backend alternative
through :class:`~isaaclab_tasks.utils.PresetCfg`. Use that mechanism for intentional
backend-specific physics, asset, and control configuration. A physics preset should not silently
change policy-facing action, observation, reward, termination, command, or reset terms. If a
``PresetCfg`` used by an MDP term does select different behavior, treat the resolved configurations
as different tasks: restore one checkpoint contract or retrain for the new contract.

Before attempting transfer, ensure that the same task can be trained successfully in both engines.
Then resolve each backend configuration and audit the policy interface. The following values must
match exactly:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Contract
     - Required equality
   * - Actions
     - Term order, ordered joint or body names, width, target type, scale, offset, and clipping.
   * - Observations
     - Group and term order, tensor width, history length, units, frames, clipping, and corruption
       behavior.
   * - Policy state
     - Normalization statistics, recurrent-state shape and reset, commands, and privileged inputs
       used by the actor.
   * - Timing
     - Physics ``dt``, decimation, policy period, and action-hold behavior. Newton substeps may
       differ inside the same policy period.
   * - Mechanism
     - Ordered bodies and joints, active degrees of freedom, and mimic or equality coupling.
   * - Episode
     - Reset and command distributions, reward and termination meanings, horizon, and success
       definition.


Mimic-joint action nuance
~~~~~~~~~~~~~~~~~~~~~~~~~

PhysX and Newton MJWarp preserve the leader and follower joint coordinates, but they do not create
the same drive graph. Newton imports the authored mimic relation as a mimic constraint, and
``SolverMuJoCo`` lowers it to a MuJoCo ``mjEQ_JOINT`` equality constraint for MJWarp. The Franka
finger pair therefore has one active joint drive: ``panda_finger_joint1`` is driven and the
constraint moves ``panda_finger_joint2``. PhysX creates a native two-way articulation mimic
constraint, but the mimic follower still counts as a driveable joint. The constraint does not
disable a drive authored on that joint.

This distinction matters when an actuator expression such as ``panda_finger_joint.*`` assigns
nonzero stiffness and damping to both fingers. PhysX then has two active PD drives in addition to
the mimic coupling. When one logical gripper command is written to both finger targets, PhysX
applies drive effort through both joints, effectively applying the command twice relative to
MJWarp's single driven finger. For the franka asset, we removed that discrepancy by driving only
the leader and explicitly making the follower passive:

.. code-block:: python

   "panda_hand": ImplicitActuatorCfg(
       joint_names_expr=["panda_finger_joint1"],
       # physical limits, gains, and armature
   ),
   "panda_finger2_passive": ImplicitActuatorCfg(
       joint_names_expr=["panda_finger_joint2"],
       stiffness=0.0,
       damping=0.0,
       # retain the follower's limits and armature
   ),

Zero stiffness and damping are what disable the second PD drive. The follower remains in the
articulation so the mimic constraint can move it.


Articulation joint and body ordering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PhysX and MJWarp may order joints and bodies differently for branched robots. For cross-backend
playback, set **both** ``joint_ordering`` and ``body_ordering`` to the backend used for training;
``physics=`` still selects the target backend. The task table below lists which tasks need the
overrides and which do not.

See :ref:`joint-and-body-ordering` for the full
ordering contract, accepted values, and troubleshooting.

A scrambled axis has a distinctive signature: a locomotion policy falls within a few dozen steps
in every environment rather than degrading gracefully. Rule ordering out before attributing such a
collapse to contacts, friction, or actuator response. Once the axes agree, the transferred policy
should survive on a timescale comparable to its source baseline, and whatever gap remains is
solver dynamics.


Transferring control behavior
-----------------------------

Match the nominal actuator response before tuning the policy:

* distinguish the physical ``actuator_velocity_limit`` from the solver ``joint_velocity_limit``;
* use per-joint effort, stiffness, damping, friction, and armature;
* preserve ``dt * decimation`` and action hold;
* keep targets away from hard joint stops;
* monitor saturation and consecutive action sign changes.

Increased damping is often necessary to prevent bang-bang control. With too little damping, a
position policy can alternate saturated commands and exploit one solver's drive integration or
limit response. Armature is equally important in MJWarp: it adds reflected inertia to the
generalized mass matrix and prevents small contact or drive impulses from producing excessive
joint or angular velocity. Retune damping after increasing armature because the effective natural
frequency and damping ratio change. See :ref:`prepare-asset-for-newton` for the actuator audit and
physical sourcing guidance.


Introducing domain randomization
--------------------------------

Domain randomization is a useful technique for preventing policies from overfitting to a specific solver.
Adding randomization to solver-relevant attributes, such as joint gains and friction, improves the
policy's ability to adapt to variations in solver behavior.

.. list-table::
   :header-rows: 1
   :widths: 23 34 43

   * - Family
     - Transfer purpose
     - Important nuance
   * - Robot and object friction
     - Covers material and contact-model uncertainty.
     - Current Newton event behavior uses one friction coefficient. PhysX static/dynamic values and
       buckets do not map one-to-one.
   * - Object mass and inertia
     - Covers payload and geometry variation.
     - Keep inertia positive and physically consistent. Decide explicitly whether changing mass
       recomputes inertia.
   * - Joint gains and friction
     - Covers actuator identification and loss uncertainty.
     - Accommodate variations in solver behavior even when the same attributes are used across engines.
   * - Joint armature
     - Covers reflected-inertia uncertainty and MJWarp-sensitive acceleration.
     - Use positive, physically supported ranges and randomize coupled mechanisms coherently.
   * - Gravity
     - Eases lift learning and covers load variation.
     - Progress to full nominal gravity and evaluate there. Zero gravity is not the deployment
       condition.
   * - Actuator response
     - Covers gripper closing speed or motor response.
     - Randomize the drive to accommodate differences in solver behavior.
   * - Reset pose and geometry
     - Covers grasp and contact diversity.
     - Re-run collision-valid reset checks for every geometry variant.
   * - Observation noise
     - Reduces dependence on backend-specific state estimates.
     - Match a plausible sensor and never change tensor shape or ordering.

Domain randomization should span plausible modeling uncertainty, not arbitrarily wide values. If a
distribution must become extreme for transfer to work, revisit the nominal model and the feature
the policy is exploiting.

Use curriculum when the final distribution prevents early learning. Zero gravity, low observation
noise, tighter reset ranges, or easier termination bounds can form the initial stage, but the
curriculum must promote to the final deployment distribution. Keep a separate deterministic
nominal evaluation so random draws do not obscure backend differences.

Validate the full matrix
------------------------

Evaluate every training/deployment combination:

.. list-table::
   :header-rows: 1

   * - Training backend
     - Deployment backend
     - Label and purpose
   * - PhysX
     - PhysX
     - PP: PhysX source baseline.
   * - PhysX
     - Newton
     - PN: PhysX-to-Newton transfer.
   * - Newton
     - Newton
     - NN: Newton source baseline.
   * - Newton
     - PhysX
     - NP: Newton-to-PhysX transfer.

The exact entry point can vary by RL library. With the unified Isaac Lab entry point:

**1. Train in PhysX:**

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task TRAIN_TASK physics=isaacsim_physx

**PP: reproduce source baseline in PhysX**

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl --task PLAY_TASK \
       --checkpoint logs/rsl_rl/EXPERIMENT_DIRECTORY/RUN_DIRECTORY/model_ITERATION.pt physics=isaacsim_physx

**PN: deploy PhysX checkpoint in Newton**

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl --task PLAY_TASK \
       --checkpoint logs/rsl_rl/EXPERIMENT_DIRECTORY/RUN_DIRECTORY/model_ITERATION.pt physics=newton_mjwarp \
       env.scene.robot.joint_ordering=physx env.scene.robot.body_ordering=physx

**2. Train in Newton:**

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task TRAIN_TASK physics=newton_mjwarp

**NN: reproduce source baseline in Newton**

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl --task PLAY_TASK \
       --checkpoint logs/rsl_rl/EXPERIMENT_DIRECTORY/RUN_DIRECTORY/model_ITERATION.pt physics=newton_mjwarp

**NP: deploy Newton checkpoint in PhysX**

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl --task PLAY_TASK \
       --checkpoint logs/rsl_rl/EXPERIMENT_DIRECTORY/RUN_DIRECTORY/model_ITERATION.pt physics=isaacsim_physx \
       env.scene.robot.joint_ordering=mjwarp env.scene.robot.body_ordering=mjwarp

Drop the two ordering overrides when the task's joint and body order already agrees across
backends. See `Articulation joint and body ordering`_.


Validated transfer examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following tasks have been validated for cross-backend transfer. Substitute ``TRAIN_TASK`` /
``PLAY_TASK`` with the task ID and ``EXPERIMENT_DIRECTORY`` with the directory below into the
generic commands above.

.. list-table::
   :header-rows: 1
   :widths: 36 22 42

   * - Task
     - Experiment directory
     - Notes
   * - ``Isaac-Lift-Franka``
     - ``lift_franka``
     - No ordering override needed; joint and body order is identical in both backends. The
       ``play`` entry point applies ``play_mode`` overrides automatically.
   * - ``Isaac-Velocity-Rough-G1``
     - ``g1_rough``
     - Branched topology: add ``env.scene.robot.joint_ordering=physx
       env.scene.robot.body_ordering=physx`` for PN, and the ``mjwarp`` equivalents for NP.
   * - ``Isaac-Velocity-Rough-AnymalD``
     - ``anymal_d_rough``
     - Branched topology: same ordering overrides as G1.

.. _sim-to-sim-transfer-demonstrations:

Transfer demonstrations
~~~~~~~~~~~~~~~~~~~~~~~

These videos demonstrate PhysX-trained policies running in Newton MJWarp without retraining.
They do not represent full PP/PN/NN/NP validation. The backends are shown side by side.

**ANYmal-D rough-terrain locomotion** (``Isaac-Velocity-Rough-AnymalD``)

.. raw:: html

   <video controls preload="metadata" style="width:100%; max-width:960px; margin-bottom:1.5em;">
     <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/sim2sim_anymal_d_transfer_10s_trimmed.mp4" type="video/mp4">
   </video>

**Allegro hand cube reorientation** (``Isaac-Reorient-Cube-Allegro``, PhysX-to-Newton direction only)

.. raw:: html

   <video controls preload="metadata" style="width:100%; max-width:960px; margin-bottom:1.5em;">
     <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/sim2sim_allegro_transfer_trimmed.mp4" type="video/mp4">
   </video>

.. tip::

   For Allegro transfer, contact stiffness (``ke``), contact damping (``kd``), contact friction,
   joint damping, and joint friction are the key parameters to tune. Increasing solver substeps
   to 4–16 significantly improves contact stability during in-hand manipulation.


.. _joint-and-body-ordering:

Implementation detail: joint and body ordering
----------------------------------------------

PhysX and MJWarp may order an articulation's joints and bodies differently. Set
``joint_ordering`` and ``body_ordering`` to keep names mapped to the same tensor elements
across backends. This section explains the supported conventions, conversion costs, and direct
backend-view access. For backend selection and capabilities, see :ref:`physics-backends`.


Why Articulation Orders Differ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three facts explain why a checkpoint can need an ordering convention even when
both backends load the same USD asset:

1. USD names identify physical joints and bodies, but they do not impose one
   universal tensor-axis order across solvers.
2. PhysX and MJWarp construct native articulation views with different topology
   traversal and internal representation choices.
3. Isaac Lab resolves the requested names once during articulation
   initialization, then exposes the selected public order through its
   high-level API.

The backend selection described in :ref:`backend-architecture` controls which
native view is created. The
ordering selection controls how the high-level API presents that view.


Public and Backend Order
~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Use :ref:`solver-differences` to diagnose and tune these
differences rather than treating them as ordering failures.


Verification and Troubleshooting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
   :ref:`solver-differences`.


See also
--------

* :doc:`/source/concepts/reinforcement_learning`
* :doc:`/source/features/hydra`
* :doc:`/source/how-to/solver_tuning_mjwarp`
* :ref:`physics-backends-newton`
