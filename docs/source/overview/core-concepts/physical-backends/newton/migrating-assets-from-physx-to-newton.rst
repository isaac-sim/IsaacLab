Migrating Assets from PhysX to Newton with MJWarp
==================================================

.. seealso::

   This page is the source of truth for the ``isaaclab-preparing-assets-for-newton`` agent skill
   (`skill source
   <https://github.com/isaac-sim/IsaacLab/blob/develop/skills/user/prepare-assets-for-newton/SKILL.md>`__).
   When you
   change this page, update the skill so agent guidance stays in sync. See
   :doc:`/source/overview/developer-guide/agent_skills`.

   After the asset and task run under both backends, follow
   :doc:`/source/how-to/transfer_policies_between_physx_and_newton` and its
   ``isaaclab-transferring-policies-sim-to-sim``
   `skill
   <../../../../../../skills/user/isaaclab-transferring-policies-sim-to-sim/SKILL.md>`__
   to transfer a policy checkpoint.

   Use Newton's
   `Simulation Tuning guide <https://newton-physics.github.io/newton/latest/concepts/simulation_tuning.html>`__
   for the current solver-specific diagnosis workflow and configuration references. The guide was
   introduced in `newton-physics/newton#3224
   <https://github.com/newton-physics/newton/pull/3224>`__.


This guide outlines general considerations when migrating an asset from PhysX to Newton/MJWarp
for the first time. Missing inertia, implicit
collision assumptions, ambiguous joint coupling, or placeholder actuator parameters can
produce a model that loads in both paths but represents two different systems.

This guide covers the robot, rigid objects, collision geometry, articulation topology, actuators,
and task integration required to make a PhysX asset MJWarp-ready.
Policy training and evaluation are
intentionally covered in the separate sim-to-sim how-to.


Multi-backend Asset Importing Pipeline
--------------------------------------

Assets provided by Isaac Lab work with both ``physics=physx`` and
``physics=newton_mjwarp``. Many of these assets still author USD Physics and PhysX schemas; Newton
parses the supported authored properties into its model so a separate Newton-only copy is not
required. Confirm each property against the supported-feature and schema documentation because a
PhysX attribute can remain present without being consumed by MJWarp.

When importing a new URDF or MJCF asset, use the updated Isaac Lab importers to create a
multi-physics asset. They generate a layered asset with neutral physics, PhysX, and MuJoCo payloads
when ``run_asset_transformer`` and ``run_multi_physics_conversion`` are enabled; both
converter-config options default to ``True``. The new converter produces a nested rigid-body
structure instead of the flat USD structure from earlier releases. See
:doc:`/source/how-to/import_new_asset`, :class:`~isaaclab.sim.converters.UrdfConverterCfg`, and
:class:`~isaaclab.sim.converters.MjcfConverterCfg`.


Use per-solver asset configuration classes
------------------------------------------

Asset configuration classes have been structured to allow for solver-common parameters and
solver-specific parameters. Put common
USD Physics properties in the solver-common classes such as ``RigidBodyBaseCfg`` and
``JointDriveBaseCfg``. Put backend-only properties in the matching subclasses:

* Use ``MujocoRigidBodyPropertiesCfg`` and ``MujocoJointDrivePropertiesCfg`` for properties
  implemented specifically by Newton's MJWarp solver.
* Use the ``Newton*PropertiesCfg`` classes for supported Newton collision, material, articulation,
  and other Newton-native properties.
* Keep PhysX-only damping, stabilization, solver-iteration, friction-patch, and compliant-contact
  parameters in the corresponding ``Physx*PropertiesCfg`` classes.

The class hierarchy and parameter-to-USD routing are documented in
:doc:`../../schema_cfgs`. Check the
:doc:`Newton/MuJoCo schema API </source/api/lab_newton/isaaclab_newton.sim.schemas>` and
:doc:`PhysX schema API </source/api/lab_physx/isaaclab_physx.sim.schemas>` for the currently
supported fields. A value being present in the PhysX asset or imported Newton model does not prove
that MJWarp uses it during stepping.

This setup is designed to help prevent scenarios where parameters are silently ignored when unsupported
by a solver. For backwards compatibility, we still allow using a default configuration that may include
solver-specific attributes. However, some attributes may not be parsed by specific solvers.


Audit the authored mechanical model
-----------------------------------

Inspect every dynamic link and contact-relevant object:

* Author positive mass [kg], center of mass [m], and positive-definite inertia [kg*m^2]. Treat a
  placeholder inertia warning as a modeling failure.
* Confirm the inertia frame and center-of-mass frame. A positive tensor in the wrong frame still
  produces incorrect joint and contact acceleration.
* Apply ``UsdPhysics.CollisionAPI`` only to intended collision geometry. Do not let an importer
  silently fall back to render meshes.
* Compare collision approximation, mesh scale, contact offset or margin, material binding,
  restitution, and self-collision filtering. Visual parity is not collision parity.
* Verify the articulation root, fixed-base representation, fixed-joint merging, nested rigid
  bodies, joint types, axes, and limits.
* Check that gravity is enabled or disabled intentionally for each rigid body. A source asset can
  carry a stale body-level override that makes the task behave differently from the scene gravity.


Match contact and friction behavior
-----------------------------------

The same nominal friction coefficient does not reproduce the same grasp in PhysX and MJWarp.
MJWarp often exhibits more slip because contact dimensionality, material combination, contact
geometry and count, and the global friction-cone formulation differ. Tune the target contact path
instead of copying PhysX friction settings numerically:

* First confirm that both backends use the intended collision shapes and material bindings and
  generate contacts at the expected locations. Also confirm that the gripper can supply the
  intended normal force.
* Inspect the resolved MJWarp contact dimensionality, ``condim``. ``condim=1`` is frictionless,
  ``3`` adds tangential friction, ``4`` adds torsional friction, and ``6`` also adds rolling
  friction. Select the smallest model that represents the required contact physics and verify that
  the active importer and contact path preserve it.
* Tune the material friction against measured tangential slip. PhysX static and dynamic
  coefficients and combination modes do not map one-to-one to MJWarp's resolved coefficient.
* Tune the global ``MJWarpSolverCfg.cone`` and ``MJWarpSolverCfg.impratio`` after the contact model
  and coefficients are valid. ``cone="elliptic"`` can suppress grasp slip more faithfully than
  ``"pyramidal"`` at additional cost. For a validated grasp, ``impratio=10`` is a useful starting
  point; recheck convergence and runtime after changing either setting.

``condim`` is a per-collision-shape MuJoCo attribute and defaults to ``3`` when ``mjc:condim`` is
not authored. ``impratio`` and ``cone`` are global MJWarp solver options; ``impratio`` defaults to
``1.0``. The following task configuration writes ``mjc:condim=4`` on every collision shape below
``object_spawn_cfg`` and selects the grasping solver starting point:

.. code-block:: python

    from typing import ClassVar, Literal

    import isaaclab.sim as sim_utils
    from isaaclab.utils import configclass
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg


    @configclass
    class MujocoCondimCfg(sim_utils.CollisionFragment):
        """Task-local fragment that writes the MuJoCo contact dimension."""

        _usd_namespace: ClassVar[str | None] = "mjc"
        _usd_applied_schema: ClassVar[str | None] = None

        condim: Literal[1, 3, 4, 6] | None = None


    object_spawn_cfg = sim_utils.UsdFileCfg(
        usd_path="path/to/object.usd",
        collision_props=[
            sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
            MujocoCondimCfg(condim=4),
        ],
    )

    newton_mjwarp_cfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            cone="elliptic",
            impratio=10.0,
        ),
    )

Assign ``object_spawn_cfg`` to the asset's ``spawn`` field and use ``newton_mjwarp_cfg`` as the
``newton_mjwarp`` physics preset. A file spawner applies ``collision_props`` recursively, so do not
attach this override to an entire robot when only its fingertips need ``condim=4``. Instead, author
``mjc:condim`` on the intended collider prims in USD, preserve the source MJCF ``geom`` ``condim``
during conversion, or apply separate spawn configurations to the contacting assets. Set the value
on every collider whose contact model must be controlled; then inspect the exported MJCF or
resolved model to confirm it. ``impratio`` remains global for the entire MJWarp simulation.

Record object displacement, contact count, gripper effort, penetration, and success rate for the
same fixed grasp. Do not increase friction, ``condim``, or ``impratio`` to hide missing contacts,
incorrect collision geometry, or insufficient actuator effort. See :ref:`mjwarp-solver-tuning` for
the complete tuning sequence.


Velocity limits distinction
---------------------------

``velocity_limit`` is the actuator's physical rated speed. Isaac Lab can use it in actuator or
task logic, observations, rewards, and terminations, but MJWarp does not parse it into the solver
model and does not enforce it during stepping.

``velocity_limit_sim`` requests a solver-side hard clamp and has no direct hardware counterpart.
Newton's importer can store the authored value in ``Model.joint_velocity_limit``, but the
MJWarp solver drops that field instead of consuming it. Consequently, neither
``velocity_limit`` nor ``velocity_limit_sim`` prevents a joint from exceeding the requested speed
under ``physics=newton_mjwarp``.

Do not treat these values as an MJWarp safety mechanism. When a task requires a rated-speed
boundary, check it explicitly in observations or terminations and use physically justified effort
limits, damping, armature, action scaling, rate limits, or controller clipping to keep the response
well behaved. PhysX does consume its supported solver clamp; an overly tight PhysX clamp can make a
velocity-limit termination unreachable and create a hidden transfer difference.
``effort_limit_sim`` is the simulated effort limit; use per-joint actuator or gearbox limits
rather than one oversized value for the whole robot.


Why MJWarp often needs more armature
------------------------------------

Joint armature represents motor and transmission inertia reflected into generalized coordinates.
For a geared revolute actuator, a useful physical starting point is

.. math::

   a \approx J_\mathrm{rotor} N^2,

where :math:`J_\mathrm{rotor}` is rotor inertia and :math:`N` is the reduction ratio. The effective
joint-space inertia becomes

.. math::

   M_\mathrm{eff}(q) = M(q) + \operatorname{diag}(a).

An impulse :math:`p` then produces approximately

.. math::

   \Delta \dot{q} = M_\mathrm{eff}^{-1}p.

Consequently, a light distal link, finger, or free rotational coordinate with very small effective
inertia can convert a modest drive or contact impulse into a very large velocity. PhysX and MJWarp
differ in contact regularization, constraint stabilization, drive integration, and clamp behavior;
a nearly singular or very stiff model that appeared stable in PhysX can expose much larger angular
velocities in MJWarp. Adding physically justified armature improves the conditioning of the
generalized mass matrix, bounds acceleration, and makes drives and contacts less sensitive to
time-step and iteration choices. This is why many assets need larger, per-joint armature values
when they are calibrated for MJWarp.

Zero-gravity curricula make this problem especially visible. Zero gravity does not directly damp
rotation. Instead, an unsupported object can remain in free flight or sustained manipulator
contact, without settling onto a support surface. Repeated constraint and contact impulses can
therefore excite low-inertia rotational coordinates and produce large angular velocities. During
the Franka migration, this zero-gravity behavior motivated increased armature on the low-inertia
generalized coordinates for which armature was available. Do not infer from that observation that
the grasped rigid body itself exposes an armature parameter.

Apply that remedy according to the representation:

* For a robot or an object represented as an articulation, set joint or free-joint armature to the
  smallest value supported by reflected rotor inertia, identified response, or controlled impulse
  tests.
* A plain :class:`~isaaclab.assets.RigidObject` has no actuator armature. Give it correct body mass
  and inertia first. Represent any remaining rotational energy loss according to its physical
  source: use contact and material friction for contact losses, or apply an explicit drag torque
  for bearing or fluid drag, then validate the response with free-spin and impact-decay tests.
  Enforce a required angular-speed operating bound explicitly in task or control logic. Reserve
  armature for inertia reflected through articulated coordinates; describe rigid-body drag and
  speed bounds by their actual mechanisms.

Armature is not a universal stability knob. Do not use it to hide incorrect units, missing body
inertia, penetrating resets, excessive effort or action scale, an incorrect control period, or
undersized contact capacity. Compare the source and target step and impulse responses after each
change.

Retune damping with armature
----------------------------

Insufficient damping lets a position policy alternate large positive and negative commands,
creating bang-bang control that exploits one solver's drive or joint-limit response. For one joint,

.. math::

   \omega_n = \sqrt{\frac{k_p}{I_\mathrm{eff}}}, \qquad
   \zeta = \frac{k_d}{2\sqrt{k_p I_\mathrm{eff}}}.

Increasing armature increases :math:`I_\mathrm{eff}` and, with fixed :math:`k_d`, reduces damping
ratio. Retune armature, stiffness, and damping together from an open-loop step response. Use enough
damping for a plausible, non-oscillatory response, keep action scales conservative, and keep targets
away from hard stops. Do not mask a bad inertia tensor with extreme damping.

For a stiffness-dominated relative position action, the approximate steady closing speed is

.. math::

   v \approx \frac{k_p s}{k_d},

where :math:`s` is action scale. This relation is useful for choosing gripper damping from a rated
closing speed and for defining a physically meaningful randomization range.


Choose an MJWarp starting profile
---------------------------------

PhysX and MJWarp solver parameters often do not have one-to-one mapping. PhysX scene and per-actor iteration
counts are not MJWarp nonlinear-solver iterations, and PhysX GPU contact buffers are not MJWarp's
per-environment ``njmax`` or ``nconmax`` budgets. Start from the nearest profile in
:ref:`mjwarp-solver-tuning`, then validate the task's worst-case randomized state.

For a new articulation with light contact, the common starting point is
``integrator="implicitfast"``, ``njmax=50``, ``nconmax=20``, a pyramidal cone,
``impratio=1``, and one substep. Maintained locomotion tasks commonly need constraint/contact
budgets near ``100``/``40``. Dexterous hand tasks commonly start near ``200``/``70`` with two
substeps, an elliptic cone, and ``impratio=10``. Dense manipulation can require ``300``/``200``.
These are capacity and formulation profiles, not fidelity guarantees.

Keep ``iterations=100``, ``ls_iterations=50``, and ``tolerance=1e-6`` for the first explicit
baseline. Enable ``debug_mode`` and change convergence work only after capacity is sufficient and
the iteration report or a recorded task metric identifies a convergence problem. Some dense
manipulation tasks reduce ``ls_iterations`` to ``15`` for measured performance; do not copy that
optimization before validating the larger default.

Use MuJoCo contacts by default. Switch to ``use_mujoco_contacts=False`` only when the task needs
Newton's collision pipeline, such as the rough-terrain, SDF or hydroelastic, and some
dense-manipulation patterns. Only then configure ``collision_cfg``, shape margins,
``max_triangle_pairs``, or ``rigid_contact_max``. See :doc:`mjwarp-solver` for the complete PhysX
mapping table, starting profiles, parameter semantics, and tuning sequence.

Run task-level smoke tests to compare behavior between PhysX and MJWarp:

.. code-block:: bash

   uv run python scripts/environments/zero_agent.py \
       --task TASK --num_envs 4 --headless physics=physx
   uv run python scripts/environments/zero_agent.py \
       --task TASK --num_envs 4 --headless physics=newton_mjwarp
   uv run python scripts/environments/random_agent.py \
       --task TASK --num_envs 4 --headless physics=physx
   uv run python scripts/environments/random_agent.py \
       --task TASK --num_envs 4 --headless physics=newton_mjwarp

Let each continuous agent run through multiple resets, then interrupt it. Check for non-finite
state, first-step impulses, unexpected joint saturation, excessive angular velocity, contact loss,
and importer or solver warnings.

Reset validity is part of asset readiness. Reject robot-object and robot-support penetrations,
impossible mimic states, and invalid randomized geometry before the first physics step. If valid
states are cached, inspect explicit collision meshes, cover every heterogeneous clone group,
exclude a fixed base from ground-clearance tests, store positions relative to environment origins,
and rebuild the cache after topology or geometry changes.


Diagnose MJWarp-only failures
-----------------------------

A scene can remain finite in PhysX but produce a NaN, overflow, or unstable contact in MJWarp.
Treat that difference as evidence to localize, not proof that one solver setting is wrong. MJWarp
can expose invalid or ill-conditioned asset data, unsupported features, different contact
behavior, or insufficient solver capacity that PhysX tolerated.

#. Reproduce the first failing step with one environment, a fixed seed and reset state, domain
   randomization disabled, and the same command or action sequence in both backends.
#. For a failure at initialization or the first step, inspect mass and inertia, collision scale,
   reset overlap, joint and mimic topology, drive import, and unsupported features before changing
   solver parameters.
#. For a failure that begins at contact, inspect contact locations and counts, ``nconmax`` and
   ``njmax`` warnings, margins, ``condim``, friction, cone choice, and extreme mass or inertia
   ratios.
#. For a failure under control, inspect effort and gain limits, action scale and discontinuities,
   ``dt`` and substeps, damping, armature, and joint-limit impacts.
#. For failures that appear only with dense scenes or many environments, compare the busiest
   environment against the per-environment contact and constraint capacities.

Enable ``debug_mode`` to inspect iteration-cap usage. Increase capacity when contacts or
constraints overflow; sweep iterations, line-search work, or tolerance only after the asset,
reset, controller, contact model, and capacities are valid. Keep the smallest fixed-state
reproduction and record the first non-finite quantity so later changes can be compared one at a
time.

See also
--------

* :doc:`index`
* :doc:`supported-features`
* :doc:`mjwarp-solver`
* :doc:`../../schema_cfgs`
* `Newton Simulation Tuning guide
  <https://newton-physics.github.io/newton/latest/concepts/simulation_tuning.html>`__
* `MuJoCo-Warp Contact Tuning guide
  <https://newton-physics.github.io/newton/latest/concepts/simulation_tuning_mujoco.html>`__
* :doc:`../../multi_backend_architecture`
* :doc:`/source/how-to/import_new_asset`
* :doc:`/source/features/hydra`
